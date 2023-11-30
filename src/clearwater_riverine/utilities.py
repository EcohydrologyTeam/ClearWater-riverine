import warnings

import numba
import pandas as pd
import numpy as np
import xarray as xr 

from clearwater_riverine import variables

UNIT_DETAILS = {'Metric': {'Length': 'm',
                            'Velocity': 'm/s',
                            'Area': 'm2', 
                            'Volume' : 'm3',
                            'Load': 'm3/s',
                            },
                'Imperial': {'Length': 'ft', 
                            'Velocity': 'ft/s', 
                            'Area': 'ft2', 
                            'Volume': 'ft3',
                            'Load': 'ft3/s',
                            },
                'Unknown': {'Length': 'L', 
                            'Velocity': 'L/t', 
                            'Area': 'L^2', 
                            'Volume': 'L^3',
                            'Load': 'L^3/t',
                            },
                }

CONVERSIONS = {'Metric': {'Liters': 0.001},
            'Imperial': {'Liters': 0.0353147},
            'Unknown': {'Liters': 0.001},
            }


def _determine_units(mesh) -> str:
    """ Determines units of RAS output file. 

    Returns:
        units (str):         Either 'Metric' or 'Imperial'

    """ 
    u = mesh[variables.EDGE_VELOCITY].Units
    if u == 'm/s':
        units = 'Metric'
    elif u == 'ft/s':
        units = 'Imperial'
    else:
        warnings.warn(f'Unknown units ({u}). Generic units will be stored in xarray.')
        units = 'Unknown'
    return units



class UnitConverter:
    """Handles unit conversions for concentration."""
    def __init__(self,
        mesh: xr.Dataset,
        input_mass_units: str,
        input_volume_units: str,
        input_liter_conversion: float
    ):
        """Determine the units and print unit assumptions for user.

        Args:
            input_mass_units (str): User-defined mass units for concentration timeseries used in model set-up. Assumes mg if no value
                is specified. 
            input_volume_units (str): User-defined volume units for concentration timeseries. Assumes L if no value
                is specified.
            input_liter_conversion (float): If concentration inputs are not in mass/L, supply the conversion factor to 
                convert the volume unit to liters.
        """
        self.mesh = mesh
        self.input_mass_units = input_mass_units
        self.input_volume_units = input_volume_units
        self.input_liter_conversion = input_liter_conversion
        self.units = _determine_units(mesh)
        self._print_assumptions()
    
    def _print_assumptions(self):
        print(f" Assuming concentration input has units of {self.input_mass_units}/{self.input_volume_units}...")
        print("     If this is not true, please re-run the wq simulation with input_mass_units, input_volume_units, and liter_conversion parameters filled in appropriately.")

    def _conversion_factor(self):
        """Determine conversion factor. If the input volume units are the same
            as the model units, then there is no conversion necessary. 
            If the input_volume_units are liters, then we know the conversion factor,
            stored in a dictionary. If any other unit, the user must define the conversion factor.
        """
        if self.input_volume_units == UNIT_DETAILS[self.units]['Volume']:
            self.conversion_factor = 1 
        elif self.input_volume_units == 'L':
            self.conversion_factor = CONVERSIONS[self.units]['Liters']
        else:
            self.conversion_factor = self.input_liter_conversion * CONVERSIONS[self.units]['Liters']
    
    def _convert_units(
        self,
        input_array: np.ndarray,
        convert_to: bool = True
    ) -> np.ndarray:
        """Converts an array from input units to model units or vice versa
        
        Args:
            input_array (np.ndarray): array to be converted
            convert_to (bool): True if converting from input units to model units, 
                otherwise False 
        Returns:
            np.ndarray: converted array
        """
        self._conversion_factor()
        if convert_to:
            return input_array * self.conversion_factor
        else:
            return input_array / self.conversion_factor


@numba.njit
def _linear_interpolate(x0: float, x1: float, y0: float, y1: float, xi: float):
    """ Linear interpolation:
    
    Args:
        x0 (float): Lower x value
        x1 (float): Upper x value
        y0 (float): Lower y value
        y1 (float): Upper y value
        xi (float): x value to interpolate
    Returns:
        y1 (float): interpolated y value
    """
    m = (y1 - y0)/(x1 - x0)
    yi = m * (xi - x0) + y0
    return yi

@numba.njit
def _compute_cell_volumes(
    water_surface_elev_arr: np.ndarray,
    cells_surface_area_arr: np.ndarray,
    starting_index_arr: np.ndarray,
    count_arr: np.ndarray,
    elev_arr: np.ndarray,
    vol_arr: np.ndarray,
    ) -> np.ndarray:
    """Compute the volumes of the RAS cells using lookup tables"""
    ntimes, ncells = water_surface_elev_arr.shape
    cell_volumes = np.zeros((ntimes, ncells))

    for time in range(ntimes):
        for cell in range(ncells):
            water_surface_elev = water_surface_elev_arr[time, cell]
            surface_area = cells_surface_area_arr[cell]
            index = starting_index_arr[cell] # Start index in the volume-elevation table for this cell
            count = count_arr[cell] # Number of points in the table for this cell

            # A number of cells have an index that is just past the end of the array. According to Mark Jensen, 
            # these are ghost cells and have a volume of 0.0. The count for these cells should also always be zero. 
            # The code checks for either condition.
            if index >= len(elev_arr) or count == 0:
                cell_volumes[time, cell] = 0.0
            else:
                elev = elev_arr[index:index + count] # Get the water surface elevation array for this cell
                vol = vol_arr[index:index + count] # Get the volume array for this cell

                if water_surface_elev > elev[-1]:
                    '''
                    Compute the net volume: the max volume in the lookup table plus the volume of the water above the max 
                    elevation in the lookup table.
                    
                    Note: this assumes a horizontal water surface, i.e., that the slope of the water surface across a cell
                    is negligible. The error increases with cell size.
                    
                    The validity of this method was confirmed by Mark Jensen on Jul 29, 2022.
                    '''
                    cell_volumes[time, cell] = vol[-1] + (water_surface_elev - elev[-1]) * surface_area
                elif water_surface_elev == elev[-1]:
                    cell_volumes[time, cell] = vol[-1]
                elif water_surface_elev <= elev[0]:
                    cell_volumes[time, cell] = vol[0]
                else:
                    # Interpolate
                    cell_volumes[time, cell] = 0.0 # Default
                    npts = len(elev)
                    for i in range(npts-1, -1, -1):
                        if elev[i] < water_surface_elev:
                            cell_volumes[time, cell] = _linear_interpolate(elev[i], elev[i+1], vol[i], vol[i+1], water_surface_elev)

    return cell_volumes

@numba.njit
def _compute_face_areas(
    water_surface_elev_arr: np.ndarray,
    faces_lengths_arr: np.ndarray,
    faces_cell_indexes_arr: np.ndarray,
    starting_index_arr: np.ndarray,
    count_arr: np.ndarray,
    elev_arr: np.ndarray,
    area_arr: np.ndarray,
    ) -> np.ndarray:
    """Compute the areas of the RAS cell faces using lookup tables"""
    ntimes, _ = water_surface_elev_arr.shape
    nfaces = len(faces_lengths_arr)
    face_areas = np.zeros((ntimes, nfaces))
    for time in range(ntimes):
        for face in range(nfaces):
            cell = faces_cell_indexes_arr[face]
            water_surface_elev = water_surface_elev_arr[time, cell]
            index = starting_index_arr[face] # Start index in the area-elevation table for this cell (note: this is indexed by faces)
            count = count_arr[face] # Number of points in the table for this cell (note: this is indexed by faces)

            # A number of cells have an index that is just past the end of the array. According to Mark Jensen, 
            # these are ghost cells and have a volume of 0.0. The count for these cells should also always be zero. 
            # The code checks for either condition.
            if index >= len(elev_arr) or count == 0:
                face_areas[time, face] = 0.0
            else:
                elev = elev_arr[index:index + count] # Get the water surface elevation (Z) array for this face
                area = area_arr[index:index + count] # Get the face area array for this face

                if water_surface_elev > elev[-1]:
                    """
                    Compute the net face surface area: the max face area in the lookup table plus the face area of 
                    the water above the max elevation in the lookup table.
                    
                    Note: this assumes a horizontal water surface, i.e., that the slope of the water surface across a cell face
                    is negligible. The error increases with cell size.
                    
                    The validity of this method was confirmed by Mark Jensen on Jul 29, 2022.
                    """
                    face_areas[time, face] = area[-1] + (water_surface_elev - elev[-1]) * faces_lengths_arr[face]
                elif water_surface_elev == elev[-1]:
                    face_areas[time, face] = area[-1]
                elif water_surface_elev <= elev[0]:
                    face_areas[time, face] = area[0]
                else:
                    # Interpolate
                    face_areas[time, face] = 0.0 # Default
                    npts = len(elev)
                    for i in range(npts-1, -1, -1):
                        if elev[i] < water_surface_elev:
                            x = water_surface_elev
                            face_areas[time, face] = _linear_interpolate(elev[i], elev[i+1], area[i], area[i+1], water_surface_elev)
                            # print('i, x, m, x1, y1, y: ', i, x, m, x1, y1, y)
                            if face_areas[time, face] < 0:
                                print('Computed face area = ', face_areas[time, face])
                                print('Time Step: ', time)
                                print('Cell number: ', cell)
                                print('Face number: ', face)
                                print('water_surface_elev: ', water_surface_elev)
                                print('elev: ', elev)
                                print('area: ', area)
                                # msg = 'Negative face area: computed face area = ' + str(face_areas[time, face])
                                raise(ValueError('Negative face area'))

    return face_areas


def _calc_distances_cell_centroids(mesh: xr.Dataset) -> np.array:
    """ Calculate the distance between cell centroids

    Args:
        mesh (xr.Dataset): Clearwater Model mesh

    Returns:
        dist_data (np.array):   Array of distances between all cell centroids 
    """
    # Get northings and eastings of relevant faces 
    x1_coords = mesh['face_x'][mesh['edges_face1']]
    y1_coords = mesh['face_y'][mesh['edges_face1']]
    x2_coords = mesh['face_x'][mesh['edges_face2']]
    y2_coords = mesh['face_y'][mesh['edges_face2']]

    # calculate distance 
    dist_data = np.sqrt((x1_coords - x2_coords)**2 + (y1_coords - y2_coords)**2)
    return dist_data

def _calc_coeff_to_diffusion_term(mesh: xr.Dataset) -> np.array:
    """ Calculate the coefficient to the diffusion term. 

    For each edge, this is calculated as:
    (Edge vertical area * diffusion coefficient) / (distance between cells) 
    
    Args:
        mesh (xr.Dataset):   Mesh created by the populate_ugrid function

    Returns:
        diffusion_array (np.array):     Array of diffusion coefficients associated with each edge

    """
    # diffusion coefficient: ignore diffusion between cells in the mesh and ghost cells
    diffusion_coefficient = np.zeros(len(mesh['nedge']))

    # identify ghost cells: 
    # ghost cells are only in the second element of a pair or cell indices that denote an edge together
    f2_ghost = np.where(mesh['edges_face2'] <= mesh.attrs['nreal'])

    # set diffusion coefficients where NOT pseusdo cell 
    diffusion_coefficient[np.array(f2_ghost)] = mesh.attrs['diffusion_coefficient']

    # diffusion_array =  mesh['edge_vertical_area'] * diffusion_coefficient / mesh['face_to_face_dist']
    diffusion_array =  mesh['edge_vertical_area'] * mesh.attrs['diffusion_coefficient'] / mesh['face_to_face_dist']
    return diffusion_array

def _sum_vals(mesh: xr.Dataset, face: np.array, time_index: float, sum_array: np.array) -> np.array:
    """ Sums values associated with a given cell. 
    
    Developed this function with help from the following Stack Overflow thread:  
    https://stackoverflow.com/questions/67108215/how-to-get-sum-of-values-in-a-numpy-array-based-on-another-array-with-repetitive
    
    Args:
        mesh (xr.Dataset):      Clearwater Riverine model mesh
        face (np.array):        Array containing all cells on one side of an edge connection
        time_index (float):     Timestep for calculation 
        sum_array (np.array):   Empty array to populate with sum values

    Returns:
        sum_array (np.array):   Array populated with sum values     
    """
    # _, idx, _ = np.unique(face, return_counts=True, return_inverse=True)
    nodal_values = np.bincount(face.values, mesh['coeff_to_diffusion'][time_index])
    sum_array[0:len(nodal_values)] = nodal_values
    return sum_array

def _calc_sum_coeff_to_diffusion_term(mesh: xr.Dataset) -> np.array:
    """Calculates the sum of all coefficients to diffusion terms. 

    Sums all coefficient to the diffusion term values associated with each individual cell
    (i.e., transfers values associated with EDGES to relevant CELLS)
    These values fall on the diagonal of the LHS matrix when solving the transport equation. 

    Args:
        mesh (xr.Dataset):  Clearwater Riverine model mesh

    Returns:
        sum_diffusion_array: Array containing the sum of all diffusion coefficients 
                                associated with each cell. 
    """
    # initialize array
    sum_diffusion_array = np.zeros((len(mesh['time']), len(mesh['nface'])))
    for t in range(len(mesh['time'])):
        # initialize arrays
        f1_sums = np.zeros(len(mesh['nface'])) 
        f2_sums = np.zeros(len(mesh['nface']))

        # calculate sums for all values
        f1_sums = _sum_vals(mesh, mesh['edges_face1'], t, f1_sums)
        f2_sums = _sum_vals(mesh, mesh['edges_face2'], t, f2_sums)

        # add f1_sums and f2_sums together to get total 
        # need to do this because sometimes a cell is the first cell in a pair defining an edge
        # and sometimes a cell is the second cell in a pair defining an edge
        sum_diffusion_array[t] = f1_sums + f2_sums
    return sum_diffusion_array

def _calc_ghost_cell_volumes(mesh: xr.Dataset) -> np.array:
    """
    In RAS2D output, all ghost cells (boundary cells) have a volume of 0. 
    However, this results in an error in the sparse matrix solver
    because nothing can leave the mesh. 
    Therefore, we must calculate the 'volume' in the ghost cells as 
    the total flow across the face in a given timestep.

    Args:
        mesh (xr.Dataset):   Clearwater Riverine model mesh

    Returns:
        ghost_vols_in:       Volume entering the domain from ghost cells
        ghost_vols_out:      Volume leaving the domain to ghost cells
    """
    f2_ghost = np.where(mesh['edges_face2'] > mesh.attrs['nreal'])[0]  
    ghost_vels = np.zeros((len(mesh['time']), len(mesh['nedge'])))

    # volume leaving 
    for t in range(len(mesh['time'])):
        # positive velocities 
        positive_velocity_indices = np.where(mesh['edge_velocity'][t] > 0 )[0]

        # get intersection - this is where water is flowing OUT to a ghost cell
        index_list = np.intersect1d(positive_velocity_indices, f2_ghost)

        if len(index_list) != 0:
            ghost_vels[t][index_list] = mesh['edge_velocity'][t][index_list]

    # calculate volume
    ghost_flux_vols = ghost_vels * mesh['edge_vertical_area'] * mesh['dt']

    # transfer values (acssociated with EDGES) to corresponding CELLS (FACES)
    ghost_vols_out = np.zeros((len(mesh['time']), len(mesh['nface'])))
    for t in range(len(mesh['time'])):
        indices = np.where(ghost_flux_vols[t] > 0)[0]
        cell_ind = mesh['edges_face2'][indices]
        vals = ghost_flux_vols[t][indices]
        if len(cell_ind) > 0:
            ghost_vols_out[t][np.array(cell_ind)] = vals 
    
    # # volume coming in
    ghost_vels_in = np.zeros((len(mesh['time']), len(mesh['nedge'])))

    for t in range(len(mesh['time'])):
        # negative velocities
        negative_velocity_indices = np.where(mesh['edge_velocity'][t] < 0 )[0]

        # get intersection - this is where water is flowing IN from a ghost cell
        index_list = np.intersect1d(negative_velocity_indices, f2_ghost)

        if len(index_list) != 0:
            ghost_vels_in[t][index_list] = mesh['edge_velocity'][t][index_list]

    # ghost_flux_in_vols = ghost_vels_in * mesh['edge_vertical_area'] * mesh['dt'] * -1 
    seconds = mesh['dt']
    ghost_flux_in_vols = np.sign(ghost_vels_in) * mesh['face_flow'] * seconds

    ghost_vols_in = np.zeros((len(mesh['time']), len(mesh['nface'])))
    for t in range(1, len(mesh['time'])):
        indices = np.where(ghost_flux_in_vols[t] > 0)[0]
        cell_ind = mesh['edges_face2'][indices]
        vals = ghost_flux_in_vols[t][indices]
        if len(cell_ind) > 0:
            ghost_vols_in[t-1][np.array(cell_ind)] = vals # shifts this back a timestep
        else:
            pass

        # all_ghost_vols = ghost_vols_out + ghost_vols_in
    return ghost_vols_in, ghost_vols_out

class WQVariableCalculator:
    """Calculates all parameters required for advection-diffusion equations"""
    def __init__(self, mesh: xr.Dataset):
        """Determine the units 
        Args:
            mesh (xr.Dataset):   Clearwater Riverine model mesh
        """
        mesh.attrs['units'] = _determine_units(mesh)
    
    def calculate(self, mesh: xr.Dataset):
        """Calculate required values for advection-diffusion transport equation
        Args:
            mesh (xr.Dataset):  Clearwater Riverine model mesh

        """
        if mesh.attrs['volume_calculation_required']:
            print( """
                Warning! Cell volumes are being manually calculated. 
                This functionality is not fully tested. 
                For best results, please re-run the RAS model with optional outputs Cell Volume, Face Flow, and Eddy Viscosity selected.
                """)
            cell_volumes = _compute_cell_volumes(
                mesh[variables.WATER_SURFACE_ELEVATION].values,
                mesh[variables.FACE_SURFACE_AREA].values,
                mesh.attrs['face_area_elevation_info']['Starting Index'].values,
                mesh.attrs['face_area_elevation_info']['Count'].values,
                mesh.attrs['face_area_elevation_values']['Elevation'].values,
                mesh.attrs['face_area_elevation_values']['Volume'].values,
            )
            mesh[variables.VOLUME] = xr.DataArray(
                cell_volumes,
                dims =  ('time', 'ncell'),
                attrs = {'Units': UNIT_DETAILS[mesh.attrs['units']]['Volume']}
            )
        
        if mesh.attrs['face_area_calculation_required']:
            print("""
                Warning! Flows across the face are being manually calculated.
                This functionality is not fully tested!
                For best results, please re-run the RAS model with optional outputs Cell Volume, Face Flow, and Eddy Viscosity selected.
                """)
            # should we be using 0 or 1 ?
            face_areas = _compute_face_areas(
                mesh[variables.WATER_SURFACE_ELEVATION].values,
                mesh.attrs['face_normalunitvector_and_length']['Face Length'].values,
                mesh.attrs['face_cell_indexes_df']['Cell 0'].values,
                mesh.attrs['face_area_elevation_values']['Starting Index'].values,
                mesh.attrs['face_area_elevation_info']['Count'].values,
                mesh.attrs['face_area_elevation_values']['Z'].values,
                mesh.attrs['face_area_elevation_values']['Area'].values,
            )
            mesh[variables.EDGE_VERTICAL_AREA] = xr.DataArray(
                face_areas,
                dims =  ('time', 'nedge'),
                attrs = {'Units': UNIT_DETAILS[mesh.attrs['units']]['Area']}
            )
            advection_coefficient = mesh[variables.EDGE_VERTICAL_AREA] * mesh[variables.EDGE_VELOCITY] 
            mesh[variables.ADVECTION_COEFFICIENT] = xr.DataArray(
                advection_coefficient,
                dims = ('time', 'nedge'),
                attrs = {'Units': UNIT_DETAILS[mesh.attrs['units']]['Load']})
            mesh[variables.FLOW_ACROSS_FACE] = xr.DataArray(
                abs(advection_coefficient),
                dims = ('time', 'nedge'),
                attrs={'Units': UNIT_DETAILS[mesh.attrs['units']]['Load']})
        else:
            mesh[variables.ADVECTION_COEFFICIENT] = xr.DataArray(
                mesh[variables.FLOW_ACROSS_FACE] * np.sign(abs(mesh[variables.EDGE_VELOCITY])),
                dims = ('time', 'nedge'),
                attrs={'Units': UNIT_DETAILS[mesh.attrs['units']]['Load']})
            
            vertical_area = mesh['advection_coeff'] / mesh['edge_velocity']
            mesh[variables.EDGE_VERTICAL_AREA] = xr.DataArray(
                vertical_area.fillna(0),
                dims = ('time', 'nedge'),
                attrs={'Units': UNIT_DETAILS[mesh.attrs['units']]['Area']})
        
        mesh[variables.FACE_TO_FACE_DISTANCE] = xr.DataArray(
            _calc_distances_cell_centroids(mesh),
            dims = ('nedge'),
            attrs={'Units': UNIT_DETAILS[mesh.attrs['units']]['Length']}
        )
        mesh[variables.COEFFICIENT_TO_DIFFUSION_TERM] = xr.DataArray(
            _calc_coeff_to_diffusion_term(mesh),
            dims = ("time", "nedge"),
            attrs={'Units': UNIT_DETAILS[mesh.attrs['units']]['Load']}
        )
        print(' Calculating sum coefficient to diffusion term...')
        mesh[variables.SUM_OF_COEFFICIENTS_TO_DIFFUSION_TERM] = xr.DataArray(
            _calc_sum_coeff_to_diffusion_term(mesh),
            dims=('time', 'nface'),
            attrs={'Units': UNIT_DETAILS[mesh.attrs['units']]['Load']}
        )
        # dt
        dt = np.ediff1d(mesh['time'])
        dt = dt / np.timedelta64(1, 's')
        dt = np.insert(dt, len(dt), np.nan)
        mesh[variables.CHANGE_IN_TIME] = xr.DataArray(dt, dims=('time'), attrs={'Units': 's'})