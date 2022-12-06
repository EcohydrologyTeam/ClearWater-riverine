import h5py
import numpy as np
import pandas as pd
import xarray as xr
import numba
import datetime
from scipy.sparse import *
from scipy.sparse.linalg import *
from typing import Dict
import geopandas as gpd
import holoviews as hv
import geoviews as gv
import datetime
from shapely.geometry import Polygon
hv.extension("bokeh")


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
                            }
                }

CONVERSIONS = {'Metric': {'Liters': 0.001},
               'Imperial': {'Liters': 0.0353147}
               }


### TODD FUNCTIONS

def parse_attributes(dataset) -> Dict:
    '''
    Parse the HDF5 attributes array, convert binary strings to Python strings, and return a dictionary of attributes
    '''
    attrs = {}
    for key, value in dataset.attrs.items():
        if type(value) == np.bytes_:
            attrs[key] = value.decode('ascii')
        elif type(value) == np.ndarray:
            values = []
            for v in value:
                if type(v) == np.bytes_:
                    values.append(v.decode('ascii'))
                else:
                    values.append(v)
            attrs[key] = values
        else:
            attrs[key] = value
    return attrs

def hdf_to_xarray(dataset, dims, attrs=None) -> xr.DataArray:
    '''Read n-dimensional HDF5 dataset and return it as an xarray.DataArray'''
    if attrs == None:
        attrs = parse_attributes(dataset)
    data_array = xr.DataArray(dataset[()], dims = dims, attrs = attrs)
    return data_array

def hdf_to_pandas(dataset) -> pd.DataFrame:
    '''Read n-dimensional HDF5 dataset and return it as an xarray.DataArray'''
    attrs = parse_attributes(dataset)
    df = pd.DataFrame(dataset[()], columns = attrs['Column'])
    return df

@numba.njit
def interp(x0: float, x1: float, y0: float, y1: float, xi: float):
    '''
    Linear interpolation:
    Inputs:
        x0: Lower x value
        x1: Upper x value
        y0: Lower y value
        y1: Upper y value
        xi: x value to interpolate
    Returns:
        y1: interpolated y value
    '''
    m = (y1 - y0)/(x1 - x0)
    yi = m * (xi - x0) + y0
    return yi

@numba.njit
def compute_cell_volumes(water_surface_elev_arr: np.ndarray, cells_surface_area_arr: np.ndarray, starting_index_arr: np.ndarray, count_arr: np.ndarray, elev_arr: np.ndarray, vol_arr: np.ndarray, VERBOSE=False) -> float:
    '''Compute the volumes of the RAS cells using lookup tables'''
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
                            cell_volumes[time, cell] = interp(elev[i], elev[i+1], vol[i], vol[i+1], water_surface_elev)

    return cell_volumes

@numba.njit
def compute_face_areas(water_surface_elev_arr: np.ndarray, faces_lengths_arr: np.ndarray, faces_cell_indexes_arr: np.ndarray, starting_index_arr: np.ndarray, count_arr: np.ndarray, elev_arr: np.ndarray, area_arr: np.ndarray):
    '''Compute the areas of the RAS cell faces using lookup tables'''
    ntimes, ncells = water_surface_elev_arr.shape
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
                    '''
                    Compute the net face surface area: the max face area in the lookup table plus the face area of 
                    the water above the max elevation in the lookup table.
                    
                    Note: this assumes a horizontal water surface, i.e., that the slope of the water surface across a cell face
                    is negligible. The error increases with cell size.
                    
                    The validity of this method was confirmed by Mark Jensen on Jul 29, 2022.
                    '''
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
                            face_areas[time, face] = interp(elev[i], elev[i+1], area[i], area[i+1], water_surface_elev)
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

@numba.njit
def compute_face_areas_from_faceWSE(water_surface_elev_arr: np.ndarray, faces_lengths_arr: np.ndarray, faces_cell_indexes_arr: np.ndarray, starting_index_arr: np.ndarray, count_arr: np.ndarray, elev_arr: np.ndarray, area_arr: np.ndarray, ntimes: int, nfaces: int):
    '''
    Compute the areas of the RAS cell faces using lookup table with the
    water surface elevation from the optional HEC-RAS hdf5 output "Face
    Water Surface"
    
        Parameters
    ----------
    water_surface_elev_arr: np.ndarray
        face water surface array from HEC-RAS HDF5 file-- optional output variable "Face Water Surface"
    faces_lengths_arr: np.ndarray
        face length from HEC-RAS HDF5 file-- "Faces NormalUnitVector and Length" table
    faces_cell_indexes_arr: np.ndarray
        face cell index array from HEC-RAS HDF5 file-- "Faces Cell Indexes"
    starting_index_arr: np.ndarray
        Starting index array for area-elevation relationship table from HEC-RAS HDF5 file-- "Faces Area Elevation Info"
    count_arr: np.ndarray
        Count of rows array for area-elevation relationship table from HEC-RAS HDF5 file-- "Faces Area Elevation Info"
    elev_arr: np.ndarray
        elevation array from the area-elevation relationship table from HEC-RAS HDF5 file-- "Faces Area Elevation Values"
    area_arr: np.ndarray
        area array from the area-elevation relationship table from HEC-RAS HDF5 file-- "Faces Area Elevation Values"
    ntimes: int
        Number of output time steps in model simulation
    ###ncells: int
        ###Number of cells in model domain
    nfaces: int
        Number of faces in model domain

    Returns
    ----------
    np.ndarray
        Array of face areas
    '''
    face_areas = np.zeros((ntimes, nfaces))
    for time in range(ntimes):
        for face in range(nfaces):
            ###cell = faces_cell_indexes_arr[face]
            water_surface_elev = water_surface_elev_arr[time, face]
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
                    '''
                    Compute the net face surface area: the max face area in the lookup table plus the face area of 
                    the water above the max elevation in the lookup table.
                    
                    The validity of this method was confirmed by Mark Jensen on Jul 29, 2022.
                    '''
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
                        #if elev[i] < water_surface_elev:
                        if elev[i] < water_surface_elev and elev[i+1] >= water_surface_elev:
                            face_areas[time, face] = interp(elev[i], elev[i+1], area[i], area[i+1], water_surface_elev)
                            # print('i, x, m, x1, y1, y: ', i, x, m, x1, y1, y)
                            if face_areas[time, face] < 0:
                                print('Computed face area = ', face_areas[time, face])
                                print('Time Step: ', time)
                                #print('Cell number: ', cell)
                                print('Face number: ', face)
                                print('water_surface_elev: ', water_surface_elev)
                                print('elev: ', elev)
                                print('area: ', area)
                                # msg = 'Negative face area: computed face area = ' + str(face_areas[time, face])
                                raise(ValueError('Negative face area'))

    return face_areas

### END OF TODD's CODE 


def parse_project_name(infile: h5py._hl.files.File) -> str:
    '''
    Parse the name of a project's 2D Flow Area

    Parameters:
        infile (h5py._hl.files.File):    HDF File containing RAS2D output 

    Returns:
        project_name (str):     name of the first 2D Flow area listed in the attributes 

    Notes: 
        RAS models can have multiple 2D flow areas within a single project. 
        The code is not currently configured to handle this kind of situation.
        This is a 'back burner' issue (#7) to address in the future. 
    '''
    project_name = infile['Geometry/2D Flow Areas/Attributes'][()][0][0].decode('UTF-8')
    return project_name

def calc_distances_cell_centroids(mesh: xr.Dataset) -> np.array:
    '''
    Calculate the distance between cell centroids

    Parameters:
        mesh (xr.Dataset):   Mesh created by the populate_ugrid function

    Returns:
        dist_data (np.array):   Array of distances between all cell centroids 
    '''
    # Get northings and eastings of relevant faces 
    x1_coords = mesh['face_x'][mesh['edges_face1']]
    y1_coords = mesh['face_y'][mesh['edges_face1']]
    x2_coords = mesh['face_x'][mesh['edges_face2']]
    y2_coords = mesh['face_y'][mesh['edges_face2']]

    # calculate distance 
    dist_data = np.sqrt((x1_coords - x2_coords)**2 + (y1_coords - y2_coords)**2)
    return dist_data

def calc_coeff_to_diffusion_term(mesh: xr.Dataset) -> np.array:
    '''
    Calculate the coefficient to the diffusion term. 
    For each edge, this is calculated as:
    (Edge vertical area * diffusion coefficient) / (distance between cells) 
    
    Parameters:
        mesh (xr.Dataset):   Mesh created by the populate_ugrid function

    Returns:
        diffusion_array (np.array):     Array of diffusion coefficients associated with each edge

    '''
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

def sum_vals(mesh: xr.Dataset, face: np.array, time_index: float, sum_array: np.array) -> np.array:
    '''
    Sums values associated with a given cell. 
    Developed this function with help from the following Stack Overflow thread:  
    https://stackoverflow.com/questions/67108215/how-to-get-sum-of-values-in-a-numpy-array-based-on-another-array-with-repetitive
    
    Parameters:
        mesh (xr.Dataset):      Mesh created by the populate_ugrid function
        face (np.array):        Array containing all cells on one side of an edge connection
        time_index (float):     Timestep for calculation 
        sum_array (np.array):   Empty array to populate with sum values

    Returns:
        sum_array (np.array):   Array populated with sum values     
    '''
    # _, idx, _ = np.unique(face, return_counts=True, return_inverse=True)
    nodal_values = np.bincount(face.values, mesh['coeff_to_diffusion'][time_index])
    sum_array[0:len(nodal_values)] = nodal_values
    return sum_array

def calc_sum_coeff_to_diffusion_term(mesh: xr.Dataset) -> np.array:
    '''
    Sums all coefficient to the diffusion term values associated with each individual cell
    (i.e., transfers values associated with EDGES to relevant CELLS)
    These values fall on the diagonal of the LHS matrix when solving the transport equation. 

    Parameters:
        mesh (xr.Dataset):   Mesh created by the populate_ugrid function

    Returns:
        sum_diffusion_array: Array containing the sum of all diffusion coefficients 
                                associated with each cell. 
    '''
    # initialize array
    sum_diffusion_array = np.zeros((len(mesh['time']), len(mesh['nface'])))
    for t in range(len(mesh['time'])):
        # initialize arrays
        f1_sums = np.zeros(len(mesh['nface'])) 
        f2_sums = np.zeros(len(mesh['nface']))

        # calculate sums for all values
        f1_sums = sum_vals(mesh, mesh['edges_face1'], t, f1_sums)
        f2_sums = sum_vals(mesh, mesh['edges_face2'], t, f2_sums)

        # add f1_sums and f2_sums together to get total 
        # need to do this because sometimes a cell is the first cell in a pair defining an edge
        # and sometimes a cell is the second cell in a pair defining an edge
        sum_diffusion_array[t] = f1_sums + f2_sums
    return sum_diffusion_array

def calc_ghost_cell_volumes(mesh: xr.Dataset) -> np.array:
    '''
    In RAS2D output, all ghost cells (boundary cells) have a volume of 0. 
    However, this results in an error in the sparse matrix solver
    because nothing can leave the mesh. 
    Therefore, we must calculate the 'volume' in the ghost cells as 
    the total flow across the face in a given timestep.

    Parameters:
        mesh (xr.Dataset):   Mesh created by the populate_ugrid function

    Returns:
        ghost_vols_in:       Volume entering the domain from ghost cells
        ghost_vols_out:      Volume leaving the domain to ghost cells
    '''
    f2_ghost = np.where(mesh['edges_face2'] > mesh.attrs['nreal'])[0]  
    ghost_vels = np.zeros((len(mesh['time']), len(mesh['nedge'])))

    # volume leaving 
    for t in range(len(mesh['time'])):
        # positive velocities 
        positive_velocity_indices = np.where(mesh['edge_velocity'][t] > 0 )[0]

        # get intersection - this is where water is flowing OUT to a ghost cell
        index_list = np.intersect1d(positive_velocity_indices, f2_ghost)

        if len(index_list) == 0:
            pass
        else:
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
        else:
            pass
    
    # # volume coming in
    ghost_vels_in = np.zeros((len(mesh['time']), len(mesh['nedge'])))

    for t in range(len(mesh['time'])):
        # negative velocities
        negative_velocity_indices = np.where(mesh['edge_velocity'][t] < 0 )[0]

        # get intersection - this is where water is flowing IN from a ghost cell
        index_list = np.intersect1d(negative_velocity_indices, f2_ghost)

        if len(index_list) == 0:
            pass
        else:
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


def define_ugrid(infile: h5py._hl.files.File, project_name: str) -> xr.Dataset:
    '''
    Define UGRID-compliant xarray

    Parameters:
        infile (h5py._hl.files.File):    HDF File containing RAS2D output 
        project_name (str):              Name of the 2D Flow Area being modeled

    Returns:
        UGRID-compliant xarray with all geometry / time coordinates populated

    '''

    # initialize mesh
    mesh = xr.Dataset()

    # initialize topology
    mesh["mesh2d"] = xr.DataArray(
        data=0,
        attrs={
            # required topology attributes
            'cf_role': 'mesh_topology',
            'long_name': 'Topology data of 2D mesh',
            'topology_dimension': 2,
            'node_coordinates': 'node_x node_y',
            'face_node_connectivity': 'face_nodes',
            # optionally required attributes
            'face_dimension': 'face',
            'edge_node_connectivity': 'edge_nodes',
            'edge_dimension': 'edge',
            # optional attributes 
            'face_edge_connectivity': 'face_edges',
            'face_face_connectivity': 'face_face_connectivity',
            'edge_face_connectivity': 'edge_face_connectivity',
            'boundary_node_connectivity': 'boundary_node_connectivity',
            'face_coordinates': 'face x face_y',
            'edge_coordinates': 'edge_x edge_y',
            },
        )
    # assign coordinates 
    # x-coordinates
    mesh = mesh.assign_coords(
        node_x=xr.DataArray(
            data = infile[f'Geometry/2D Flow Areas/{project_name}/FacePoints Coordinate'][()].T[0],
            # data=[f[0] for f in ras2d_data.geometry['nodes_array']],
            dims=("node",),
            )   
        )
    # y-coordinates
    mesh = mesh.assign_coords(
            node_y=xr.DataArray(
            data=infile[f'Geometry/2D Flow Areas/{project_name}/FacePoints Coordinate'][()].T[1],
            dims=("node",),
        )
    )
    # time
    time_stamps_binary = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp'][()]
    time_stamps = [x.decode("utf8") for x in time_stamps_binary]
    mesh = mesh.assign_coords(
            time=xr.DataArray(
            data=[datetime.datetime.strptime(x, '%d%b%Y %H:%M:%S') for x in time_stamps], # '02JAN1900 22:55:00'
            dims=("time",),
                )
        )
        
    # define topology
    # face nodes 
    mesh["face_nodes"] = xr.DataArray(
        data=infile[f'Geometry/2D Flow Areas/{project_name}/Cells FacePoint Indexes'][()],
        coords={
            "face_x": ("nface", infile[f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate'][()].T[0]),
            "face_y": ("nface", infile[f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate'][()].T[1]),
        },
        dims=("nface", "nmax_face"),
        attrs={
            'cf_role': 'face_node_connectivity',
            'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
            'start_index': 0, 
            '_FillValue': -1
    })
    # edge nodes 
    mesh["edge_nodes"] = xr.DataArray(
        data=infile[f'Geometry/2D Flow Areas/{project_name}/Faces FacePoint Indexes'][()],
        dims=("nedge", '2'),
        attrs={
            'cf_role': 'edge_node_connectivity',
            'long_name': 'Vertex nodes of mesh edges',
            'start_index': 0
        })
    # edge face connectivity
    mesh["edge_face_connectivity"] = xr.DataArray(
        data=infile[f'Geometry/2D Flow Areas/{project_name}/Faces Cell Indexes'][()],
        dims=("nedge", '2'),
        attrs={
            'cf_role': 'edge_face_connectivity',
            'long_name': 'neighbor faces for edges',
            'start_index': 0
        })
        
    return mesh

def determine_units(mesh: xr.Dataset) -> str:
    '''
    Determines units of RAS2D HDF output file. 

    Parameters:
        mesh (xr.Dataset):   Mesh created by the populate_ugrid function

    Returns:
        units (str):         Either 'Metric' or 'Imperial'

    ''' 
    u = mesh.edge_velocity.Units
    if u == 'm/s':
        units = 'Metric'
    elif u == 'ft/s':
        units = 'Imperial'
    else:
        print(f"Unable to handle {u} units")
    return units


def populate_ugrid(infile: h5py._hl.files.File, project_name: str, diffusion_coefficient_input: float) -> xr.Dataset:
    '''
    Populates data variables in UGRID-compliant xarray.

    Parameters:
        infile (h5py._hl.files.File):           HDF File containing RAS2D output 
        project_name (str):                     Name of the 2D Flow Area being modeled
        diffusion_coefficient_input (float):    User-defined diffusion coefficient for entire modeling domain. 

    Returns: 
        mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.

    Notes:
        This function requires cleanup. 
        Should remove some messy calculations and excessive code for vertical area calculations.         
    '''
    print("Populating Mesh...")
    print(" Initializing Geometry...")
    mesh = define_ugrid(infile, project_name)

    print(" Storing Results...")
    # store additional useful information for various coefficient calculations in the mesh
    mesh['edges_face1'] = hdf_to_xarray(mesh['edge_face_connectivity'].T[0], ('nedge'), attrs={'Units':''})  
    mesh['edges_face2'] = hdf_to_xarray(mesh['edge_face_connectivity'].T[1], ('nedge'), attrs={'Units':''})  
    mesh.attrs['nreal'] = mesh['edge_face_connectivity'].T[0].max()
    mesh.attrs['diffusion_coefficient'] = diffusion_coefficient_input

    # surface area 
    mesh['faces_surface_area'] = hdf_to_xarray(infile[f'Geometry/2D Flow Areas/{project_name}/Cells Surface Area'],
                                                ("nface"))
    # edge velocity
    mesh['edge_velocity'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Velocity'], 
                                           ('time', 'nedge'))
    # edge length
    mesh['edge_length'] = hdf_to_xarray(infile[f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length'][:,2],
                                        ("nedge"), attrs={'Units': 'ft'}) # to do : parse from HDF 
    # water surface elev 
    mesh['water_surface_elev'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Water Surface'], 
                                                (['time', 'nface']))
    # compute necessary values 
    # TO DO: clean all this up; review functions to see if they can be simplified 
    # calculate cell volume
    print(" Computing Necessary Values...")
    # Determine Units 
    units = determine_units(mesh)
    
    # Cell Volume
    try:
        mesh['volume'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Volume'], ('time', 'nface')) 
        mesh['volume'][:,mesh.attrs['nreal'].values+1:] = 0
    except KeyError: 
        print(" Warning! Cell volumes are being manually calculated. Please re-run the RAS model with optional outputs Cell Volume, Face Flow, and Eddy Viscosity selected.")
        cells_volume_elevation_info_df = hdf_to_pandas(infile[f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Info'])
        cells_volume_elevation_values_df = hdf_to_pandas(infile[f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Values'])
        # question: is it better to separate all of these things
        # or just input mesh, cells_volume_elevation_info_df / values df 
        cell_volumes = compute_cell_volumes(
                                            mesh['water_surface_elev'].values,
                                            mesh['faces_surface_area'].values,
                                            cells_volume_elevation_info_df['Starting Index'].values,
                                            cells_volume_elevation_info_df['Count'].values,
                                            cells_volume_elevation_values_df['Elevation'].values,
                                            cells_volume_elevation_values_df['Volume'].values,
                                                )
        mesh['volume_archive'] = hdf_to_xarray(cell_volumes, ('time', 'nface'), attrs={'Units': UNIT_DETAILS[units]['Volume']})
        mesh['volume'] = mesh['volume_archive']

    # Advection Coefficient and Associated Variables
    try:
        mesh['face_flow'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Flow'], ('time', 'nedge'))
        mesh['advection_coeff'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Flow'], ('time', 'nedge'))  * np.sign(abs(mesh['edge_velocity']))
        mesh['advection_coeff'].attrs['Units'] = UNIT_DETAILS[units]['Load']
        mesh['edge_vertical_area'] = mesh['advection_coeff'] / mesh['edge_velocity']
        # mesh['edge_vertical_area'][np.where(mesh['edge_vertical_area'].isnull())] = mesh['edge_vertical_area_archive'][np.where(mesh['edge_vertical_area'].isnull())]
        mesh['edge_vertical_area'] = mesh['edge_vertical_area'].fillna(0)
        mesh['edge_vertical_area'].attrs['Units'] = UNIT_DETAILS[units]['Area']
        # advection_coefficient = mesh['edge_vertical_area'] * mesh['edge_velocity'] 
        # mesh['advection_coeff'] = hdf_to_xarray(advection_coefficient, ('time', 'nedge'), attrs={'Units':'ft3/s'})  
    except KeyError:
        print(" Warning! Flows across the face are being manually calculated. This functionality is not fully tested! Please re-run the RAS model with optional outputs Cell Volume, Face Flow, and Eddy Viscosity selected.")
        faces_area_elevation_info_df = hdf_to_pandas(infile[f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Info'])
        faces_area_elevation_values_df = hdf_to_pandas(infile[f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Values'])
        faces_normalunitvector_and_length_df = hdf_to_pandas(infile[f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length'])
        faces_cell_indexes_df = hdf_to_pandas(infile[f'Geometry/2D Flow Areas/{project_name}/Faces Cell Indexes'])
        # should we be using 0 or 1 ?
        face_areas_0 = compute_face_areas(
                                            mesh['water_surface_elev'].values,
                                            faces_normalunitvector_and_length_df['Face Length'].values,
                                            faces_cell_indexes_df['Cell 0'].values,
                                            faces_area_elevation_info_df['Starting Index'].values,
                                            faces_area_elevation_info_df['Count'].values,
                                            faces_area_elevation_values_df['Z'].values,
                                            faces_area_elevation_values_df['Area'].values,
                                        )
        mesh['edge_vertical_area_archive'] = hdf_to_xarray(face_areas_0, ('time', 'nedge'), attrs={'Units': 'ft'})
        mesh['edge_vertical_area'] = mesh['edge_vertical_area_archive']
        mesh['edge_vertical_area'].attrs['Units'] = UNIT_DETAILS[units]['Area']
        advection_coefficient = mesh['edge_vertical_area'] * mesh['edge_velocity'] 
        mesh['advection_coeff'] = hdf_to_xarray(advection_coefficient, ('time', 'nedge'), attrs={'Units': UNIT_DETAILS[units]['Load']})
        mesh['face_flow'] = hdf_to_xarray(abs(advection_coefficient), ('time', 'nedge'), attrs={'Units': UNIT_DETAILS[units]['Load']})

    # computed values 
    # distance between centroids 
    distances = calc_distances_cell_centroids(mesh)
    mesh['face_to_face_dist'] = hdf_to_xarray(distances, ('nedge'), attrs={'Units': UNIT_DETAILS[units]['Length']})

    # coefficient to diffusion term
    coeff_to_diffusion = calc_coeff_to_diffusion_term(mesh)
    mesh['coeff_to_diffusion'] = hdf_to_xarray(coeff_to_diffusion, ("time", "nedge"), attrs={'Units': UNIT_DETAILS[units]['Load']})

    # sum of diffusion coeff
    sum_coeff_to_diffusion = calc_sum_coeff_to_diffusion_term(mesh)
    mesh['sum_coeff_to_diffusion'] = hdf_to_xarray(sum_coeff_to_diffusion, ('time', 'nface'), attrs={'Units': UNIT_DETAILS[units]['Load']})

    # advection
    # advection_coefficient = mesh['edge_vertical_area'] * mesh['edge_velocity'] 
    # mesh['advection_coeff'] = hdf_to_xarray(advection_coefficient, ('time', 'nedge'), attrs={'Units':'ft3/s'})
    # mesh['advection_coeff_archive'] = 

    # dt
    dt = np.ediff1d(mesh['time'])
    dt = dt / np.timedelta64(1, 's')
    dt = np.insert(dt, len(dt), np.nan)
    mesh['dt'] = hdf_to_xarray(dt, ('time'), attrs={'Units': 's'})

    # ghost cell volumes
    ghost_volumes_in, ghost_volumes_out = calc_ghost_cell_volumes(mesh)
    mesh['ghost_volumes_in'] = hdf_to_xarray(ghost_volumes_in, ('time', 'nface'), attrs={'Units': UNIT_DETAILS[units]['Volume']})
    mesh['ghost_volumes_out'] = hdf_to_xarray(ghost_volumes_out, ('time', 'nface'), attrs={'Units':UNIT_DETAILS[units]['Volume']})
    return mesh

def populate_boundary_information(infile: h5py._hl.files.File) -> pd.DataFrame:
    '''
    Parse attribute information from RAS2D HDF output required to set up boundary conditions.

    Parameters:
        infile (h5py._hl.files.File):    HDF File containing RAS2D output 

    Returns:
        boundary_data (pd.DataFrame):         Information about RAS boundaries 
    '''

    external_faces = pd.DataFrame(infile['Geometry/Boundary Condition Lines/External Faces'][()])
    attributes = pd.DataFrame(infile['Geometry/Boundary Condition Lines/Attributes/'][()])
    str_df = attributes.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        attributes[col] = str_df[col]
    boundary_attributes = attributes
    # merge attributes and boundary condition data 
    boundary_attributes['BC Line ID'] = boundary_attributes.index
    boundary_data = pd.merge(external_faces, boundary_attributes, on = 'BC Line ID', how = 'left')
    return boundary_data

# matrix solver 
class LHS:
    def __init__(self, mesh: xr.Dataset, t: float):
        '''
        Initialize Sparse Matrix used to solve transport equation. 
        Rather than looping through every single cell at every timestep, we can instead set up a sparse 
        matrix at each timestep that will allow us to solve the entire unstructured grid all at once. 

        We will solve an implicit Advection-Diffusion (transport) equation for the fractional total-load 
        concentrations. This discretization produces a linear system of equations that can be represented by 
        a sprase-matrix problem. 

        Parameters: 
            mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (float):           Timestep
        
        Notes:
            This is empty right now. Figure out what should go here versus updateValues function. 
        '''
        return
                
        
    def updateValues(self, mesh: xr.Dataset, t: float):
        '''
        Updates values in the LHS matrix based on the timestep. 

        Rather than looping through every single cell at every timestep, we can instead set up a sparse 
        matrix at each timestep that will allow us to solve the entire unstructured grid all at once. 
        We will solve an implicit Advection-Diffusion (transport) equation for the fractional total-load 
        concentrations. This discretization produces a linear system of equations that can be represented by 
        a sprase-matrix problem. 

        A sparse matrix is a matrix that is mostly zeroes. Here, we will set up an NCELL x NCELL sparse matrix. 
            - The diagonal values represent the reference cell ("P")
            - The non-zero off-diagonal values represent the other cells that share an edge with that cell:
                i.e., neighboring cell ("N") that shares a face ("f") with P. 

        This function populates the sparse matrix with:
            - Values on the Diagonal (associated with the cell with the same index as that row/column):
                - Load at the t+1 timestep (volume at the t + 1 timestep / change in time)
                - Sum of diffusion coefficients associated with a cell
                - FOR DRY CELLS ONLY (volume = 0), insert a dummy value (1) so that the matrix is not singular
            - Values Off-Diagonal:
                - Coefficient to the diffusion term at the t+1 timestep 
            - Advection: a special case (updwinds scheme)
                - When the advection coefficient is positive, the concentration across the face will be the reference cell ("P")
                    so the coefficient will go in the diagonal. This value will then be subtracted from the corresponding neighbor cell.
                - When the advection coefficient is negative, the concentration across the face will be the neighbor cell ("N")
                    so the coefficient will be off-diagonal. This value will the subtracted from the corresponding reference cell.

        Parameters:
            mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (float):           Timestep
        '''
        # define edges where flow is flowing in versus out and find all empty cells
        # at the t+1 timestep
        flow_out_indices = np.where(mesh['advection_coeff'][t+1] > 0)[0]
        flow_in_indices = np.where(mesh['advection_coeff'][t+1] < 0)[0]
        empty_cells = np.where(mesh['volume'][t+1] == 0)[0]

        # initialize arrays that will define the sparse matrix 
        len_val = len(mesh['nedge']) * 2 + len(mesh['nface']) * 2 + len(flow_out_indices)* 2  + len(flow_in_indices)*2 + len(empty_cells)
        self.rows = np.zeros(len_val)
        self.cols = np.zeros(len_val)
        self.coef = np.zeros(len_val)

        # put dummy values in dry cells
        start = 0
        end = len(empty_cells)
        self.rows[start:end] = empty_cells
        self.cols[start:end] = empty_cells
        self.coef[start:end] = 1

        ###### diagonal terms - load and sum of diffusion coefficients associated with each cell
        start = end
        end = end + len(mesh['nface'])
        self.rows[start:end] = mesh['nface']
        self.cols[start:end] = mesh['nface']
        seconds = mesh['dt'].values[t] # / np.timedelta64(1, 's'))
        self.coef[start:end] = mesh['volume'][t+1] / seconds + mesh['sum_coeff_to_diffusion'][t+1] 

        # add ghost cell volumes to diagonals: based on flow across face into ghost cell
        # note: these values are 0 for cell that is not a ghost cell
        # note: also 0 for any ghost cell that is not RECEIVING flow 

        start = end
        end = end + len(mesh['nface'])
        self.rows[start:end] = mesh['nface']
        self.cols[start:end] = mesh['nface']
        self.coef[start:end] = mesh['ghost_volumes_out'][t+1] / seconds 
             
        ###### advection
        # if statement to prevent errors if flow_out_indices or flow_in_indices have length of 0
        if len(flow_out_indices) > 0:
            start = end
            end = end + len(flow_out_indices)

            # where advection coefficient is positive, the concentration across the face will be the REFERENCE CELL 
            # so the the coefficient will go in the diagonal - both row and column will equal diag_cell
            self.rows[start:end] = mesh['edge_face_connectivity'].T[0][flow_out_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[0][flow_out_indices]
            self.coef[start:end] = mesh['advection_coeff'][t+1][flow_out_indices]  

            # subtract from corresponding neighbor cell (off-diagonal)
            start = end
            end = end + len(flow_out_indices)
            self.rows[start:end] = mesh['edge_face_connectivity'].T[1][flow_out_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[0][flow_out_indices]
            self.coef[start:end] = mesh['advection_coeff'][t+1][flow_out_indices] * -1  
        else:
            pass

        if len(flow_in_indices) > 0:
            # update indices
            start = end
            end = end + len(flow_in_indices)

            ## where it is negative, the concentration across the face will be the neighbor cell ("N")
            ## so the coefficient will be off-diagonal 
            self.rows[start:end] = mesh['edge_face_connectivity'].T[0][flow_in_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[1][flow_in_indices]
            self.coef[start:end] = mesh['advection_coeff'][t+1][flow_in_indices] 

            ## update indices 
            start = end
            end = end + len(flow_in_indices)
            ## do the opposite on the corresponding diagonal 
            self.rows[start:end] = mesh['edge_face_connectivity'].T[1][flow_in_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[1][flow_in_indices]
            self.coef[start:end] = mesh['advection_coeff'][t+1][flow_in_indices]  * -1 
        else:
            pass
        
        ###### off-diagonal terms - diffusion
        # update indices
        start = end
        end = end + len(mesh['nedge'])
        self.rows[start:end] = mesh['edges_face1']
        self.cols[start:end] = mesh['edges_face2']
        self.coef[start:end] = -1 * mesh['coeff_to_diffusion'][t+1]

        # update indices and repeat 
        start = end
        end = end + len(mesh['nedge'])
        self.rows[start:end] = mesh['edges_face2']
        self.cols[start:end] = mesh['edges_face1']
        self.coef[start:end] = -1 * mesh['coeff_to_diffusion'][t+1] 
        return

class RHS:
    def __init__(self, mesh: xr.Dataset, t: float, inp: np.array):
        '''
        Initialize the right-hand side matrix of concentrations based on user-defined boundary conditions. 

        Parameters:
            mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (float):           Timestep
            inp (np.array):      Array of shape (time x nface) with user-defined inputs of concentrations
                                    in each cell at each timestep. 

        Notes:
            Need to consider how ghost volumes / cells will be handled. 
            Need to consider how we will format the user-defined inputs 
                - An Excel file?
                - A modifiable table in a Jupyter notebook?
                - Alternatives?
        '''
        self.conc = np.zeros(len(mesh['nface']))
        self.conc = inp[t] 
        self.vals = np.zeros(len(mesh['nface']))
        seconds = mesh['dt'].values[t] 
        # SHOULD GHOST VOLUMES BE INCLUDED?
        vol = mesh['volume'][t] + mesh['ghost_volumes_in'][t]
        self.vals[:] = vol / seconds * self.conc 
        # self.vals[:] = mesh['volume'][t] / seconds * self.conc 
        return 

    def updateValues(self, solution: np.array, mesh: xr.Dataset, t: float, inp: np.array):
        ''' 
        Update right hand side data based on the solution from the previous timestep
            solution: solution from solving the sparse matrix 
            inp: array of shape (time x nface) with user defined inputs of concentrations
                in each cell at each timestep 

        Parameters:
            solution (np.array):    Solution of concentrations at timestep t from solving sparse matrix. 
            mesh (xr.Dataset):      UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (float):              Timestep
            inp (np.array):         Array of shape (time x nface) with user-defined inputs of concentrations
                                        in each cell at each timestep.

        Returns:
            Updates the solution to loads. 
            This is required to solve the trasnport equation at the following timestep.

        '''
        seconds = mesh['dt'].values[t] 
        # solution += inp[t][:] 
        # try replacing boundary values instead of adding them:
        solution[inp[t].nonzero()] = inp[t][inp[t].nonzero()] 
        vol = mesh['volume'][t] + mesh['ghost_volumes_in'][t]
        self.vals[:] = solution * vol / seconds
        # self.vals[:] = solution * mesh['volume'][t] / seconds
        return

class ClearwaterRiverine:
    def __init__(self, hdf_fpath: str, diffusion_coefficient_input: float):
        '''
        Initialize a Clearwater Riverine WQ model by reading HDF output from a RAS2D model. 

        Parameters:
            hdf_fpath (str):   Filepath to RAS2D HDF output
            diffusion_coefficient_input (float):    User-defined diffusion coefficient for entire modeling domain. 

        '''
        with h5py.File(hdf_fpath, 'r') as infile:
            self.project_name = parse_project_name(infile)
            self.mesh = populate_ugrid(infile, self.project_name, diffusion_coefficient_input)
            self.boundary_data = populate_boundary_information(infile)
    

    def initial_conditions(self, fpath: str):
        '''
        Define initial conditions for RAS2D model from CSV file. 

        Parameters:
            fpath (str):    Filepath to CSV containing initial conditions. The CSV should have two columns:
                                one called Cell_Index and one called Concentration, which denote the concentration
                                in each cell within the model domain at the first timestep. 

        Notes:
            Should we have the option to change what timestep gets initial conditions?
                Would that be a time index or datetime or either?
            Would there be use cases where a user would want to infuse conditions into the results that
                aren't initial conditions?
            Allow other file types?
            Where / when should we deal with units and conversions?
        '''
        init = pd.read_csv(fpath)
        init['Cell_Index'] = init.Cell_Index.astype(int)
        self.input_array = np.zeros((len(self.mesh.time), len(self.mesh.nface)))
        self.input_array[0, [init['Cell_Index']]] =  init['Concentration']
        return 

    def boundary_conditions(self, fpath: str):
        '''
        Define boundary conditions for RAS2D water quality model from CSV file. 

        Parameters:
            fpath (str):    Filepath to CSV containing boundary conditions. The CSV should have the following columns:
                                RAS2D_TS_Name (the timeseries name, as labeled in the RAS model)
                                Datetime
                                Concentration 
        '''
        bc_df = pd.read_csv(fpath, parse_dates=['Datetime'])
        bc_df = bc_df[(bc_df.Datetime >= self.mesh.time.min().values) & (bc_df.Datetime <= self.mesh.time.max().values)]

        boundary_df = pd.merge(bc_df, self.boundary_data, left_on = 'RAS2D_TS_Name', right_on = 'Name', how='left')
        # Define the ghost cell associated with the Face Index
        boundary_df['Ghost Cell'] = self.mesh.edges_face2[boundary_df['Face Index'].to_list()]
        boundary_df['Domain Cell'] = self.mesh.edges_face1[boundary_df['Face Index'].to_list()]
        # Find time index
        boundary_df.reset_index(inplace=True)
        boundary_df['Time Index'] = [np.where(self.mesh.time == i)[0][0] for i in boundary_df.Datetime]
        # Put values into input array
        self.input_array[[boundary_df['Time Index']], [boundary_df['Ghost Cell']]] = boundary_df['Concentration']
        return


    def wq_simulation(self, input_mass_units = 'mg', input_volume_units = 'L', input_liter_conversion = 1, save = False, 
                        fpath_out = '.', fname_out = 'clearwater-riverine-wq-model'):
        '''
        Steps through each timestep in the output of a RAS2D model (mesh) 
        and solves the total-load advection-diffusion transport equation 
        using boundary and initial conditions.

        Parameters:
            input_mass_units:       User-defined mass units for concentration timeseries. Assumes mg if no value
                                        is specified. 
            input_volume_units:     User-defined volume units for concentration timeseries. Assumes L if no value
                                        is specified.
            input_liter_conversion: If concentration inputs are not in mass/L, supply the conversion factor to 
                                        convert the volume unit to liters. For example, if the input timeseries has a
                                        volume unit of 100 mL, the input_liter_conversion value should be 0.1, because 
                                        100 mL * 0.1 = 1 L.
            save:                   Boolean indicating whether the file should be saved. Default is to not save the output.
            fpath_out:              Filepath where the output file should be stored. Default to save in current directory.
            fname_out:              Filename of saved output.
 
        '''
        print("Starting WQ Simulation...")

        # Convert Units
        units = determine_units(self.mesh)

        print(f" Assuming concentration input has units of {input_mass_units}/{input_volume_units}...")
        print("     If this is not true, please re-run the wq simulation with input_mass_units, input_volume_units, and liter_conversion parameters filled in appropriately.")

        conversion_factor = CONVERSIONS[units]['Liters'] 
        self.inp_converted = self.input_array / input_liter_conversion / conversion_factor # convert to mass/ft3 or mass/m3 

        output = np.zeros((len(self.mesh.time), len(self.mesh.nface)))
        t = 0
        b = RHS(self.mesh, t, self.inp_converted)
        output[0] = b.vals 

        for t in range(len(self.mesh['time']) - 1):
            if t == int(len(self.mesh['time']) / 4):
                print(' 25%')
            elif t == int(len(self.mesh['time']) / 2):
                print(' 50%')
            if t == int(3 * len(self.mesh['time']) / 4):
                print(' 75%')
            lhs = LHS(self.mesh, t)
            lhs.updateValues(self.mesh, t)
            A = csr_matrix( (lhs.coef,(lhs.rows, lhs.cols)), shape=(len(self.mesh['nface']),len(self.mesh['nface'])))
            x = spsolve(A, b.vals)
            b.updateValues(x, self.mesh, t+1, self.inp_converted)
            output[t+1] = b.vals

        print(' 100%')
        self.mesh['pollutant_load'] = hdf_to_xarray(output, dims=('time', 'nface'), attrs={'Units': f'{input_mass_units}/s'})  
        temp_vol = self.mesh['volume'] + self.mesh['ghost_volumes_in']
        concentration = self.mesh['pollutant_load'] / temp_vol * conversion_factor * input_liter_conversion * self.mesh['dt']
        self.mesh['concentration'] = hdf_to_xarray(concentration, dims = ('time', 'nface'), attrs={'Units': f'{input_mass_units}/L'})

        # self.mesh.nreal = self.mesh.nreal.values
        self.mesh.attrs['nreal'] = self.mesh.nreal.values
        # may need to move this if we want to plot things besides concentration
        self.max_value = int(self.mesh['concentration'].sel(nface=slice(0, self.mesh.attrs['nreal'])).max())

        if save == True:
            self.mesh.to_zarr(f'{fpath_out}/{fname_out}.zarr', 
                              mode='w', 
                              consolidated=True)
        return

    def prep_plot(self, crs: str):
        '''
        Creates a geodataframe of polygons to represent each RAS cell. 

        Parameters:
            crs:       coordinate system of RAS project.

        Notes:
            Could we parse the CRS from the PRJ file?
        '''

        nreal_index = self.mesh.attrs['nreal'] + 1
        real_face_node_connectivity = self.mesh.face_nodes[0:nreal_index]

        # Turn real mesh cells into polygons
        polygon_list = []
        for cell in real_face_node_connectivity:
            xs = self.mesh.node_x[cell[np.where(cell != -1)]]
            ys = self.mesh.node_y[cell[np.where(cell != -1)]]
            p1 = Polygon(list(zip(xs.values, ys.values)))
            polygon_list.append(p1)
        
        gdf_ls = []
        for t in range(len(self.mesh.time)):
            temp_gdf = gpd.GeoDataFrame({'cell': self.mesh.nface[0:nreal_index],
                                        'datetime': pd.to_datetime(self.mesh.time[t].values),
                                        'concentration': self.mesh.concentration[t][0:nreal_index],
                                        'volume': self.mesh.volume[t][0:nreal_index],
                                        'cell': self.mesh.nface[0:nreal_index],
                                        'geometry': polygon_list}, 
                                        crs = crs)
            gdf_ls.append(temp_gdf)
        full_df = pd.concat(gdf_ls)
        # full_df.to_crs('EPSG:4326')
        self.gdf = full_df
        return

    def plot(self):
        '''
        Creates a dynamic polygon plot of concentrations in the RAS2D model domain.

        Parameters:
            None

        Notes:
            Play button
            Move re-projection? This is really slow, but I think geoviews requires ESPG:4326 so necessary at some point. 
            Option to save
            Build in functionality to pass plotting arguments (clim, cmap, height, width, etc.)
            Input parameter of info to plot?
        '''
        def map_generator(datetime, mval=self.max_value):
            '''
            This function generates plots for the DynamicMap
            '''
            ras_sub_df = self.gdf[self.gdf.datetime == datetime]
            ras_map = gv.Polygons(ras_sub_df.to_crs('EPSG:4326'), vdims=['concentration']).opts(height=600,
                                                                          width = 800,
                                                                          color='concentration',
                                                                          colorbar = True,
                                                                          cmap = 'OrRd', 
                                                                          clim = (0, mval),
                                                                          line_width = 0.1,
                                                                          tools = ['hover'],
                                                                       )
            return (ras_map * gv.tile_sources.CartoLight())

        dmap = hv.DynamicMap(map_generator, kdims=['datetime'])
        return dmap.redim.values(datetime=self.gdf.datetime.unique())

    def quick_plot(self):
        '''
        Creates a dynamic scatterplot of cell centroids colored by cell concentration.

        Parameters:
            crs:       coordinate system of RAS project.

        Notes:
            Play button
            Move re-projection? This is really slow, but I think geoviews requires ESPG:4326 so necessary at some point. 
            Option to save
            Build in functionality to pass plotting arguments (clim, cmap, height, width, etc.)
            Input parameter of info to plot?
        '''
        def quick_map_generator(datetime, mval=self.max_value):
            '''
            This function generates plots for the DynamicMap
            '''
            ds = self.mesh.sel(time=datetime)
            ind = np.where(ds['concentration'][0:self.mesh.attrs['nreal']] > 0)
            nodes = np.column_stack([ds.face_x[ind], ds.face_y[ind], ds['concentration'][ind], ds['nface'][ind]])
            nodes = hv.Points(nodes, vdims=['concentration', 'nface'])
            nodes_all = np.column_stack([ds.face_x[0:self.mesh.attrs['nreal']], ds.face_y[0:self.mesh.attrs['nreal']], ds.volume[0:self.mesh.attrs['nreal']]])
            nodes_all = hv.Points(nodes_all, vdims='volume')
            p1 = hv.Scatter(nodes, vdims=['x', 'y', 'concentration', 'nface']).opts(width = 1000,
                                                                                    height = 500,
                                                                                    color = 'concentration',
                                                                                    cmap = 'plasma', 
                                                                                    clim = (0, mval),
                                                                                    tools = ['hover'], 
                                                                                    colorbar = True
                                                                                    )
            
            p2 = hv.Scatter(nodes_all, vdims=['x', 'y', 'volume']).opts(width = 1000,
                                                                    height = 500,
                                                                    color = 'grey',
                                                                     )
            title = pd.to_datetime(datetime).strftime('%m/%d/%Y %H:%M ')
            return p1 # hv.Overlay([p2, p1]).opts(title=title)

        return hv.DynamicMap(quick_map_generator, kdims=['Time']).redim.values(Time=self.mesh.time.values)
