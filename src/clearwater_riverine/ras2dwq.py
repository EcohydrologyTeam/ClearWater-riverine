from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix, linalg
import holoviews as hv
import geoviews as gv
import geopandas as gpd
from shapely.geometry import Polygon
hv.extension("bokeh")

from clearwater_riverine.mesh import model_mesh
from clearwater_riverine.utilities import _determine_units
from clearwater_riverine.io.hdf import _hdf_to_xarray


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


# matrix solver 
class LHS:
    """
    Initialize Sparse Matrix used to solve transport equation. 
    Rather than looping through every single cell at every timestep, we can instead set up a sparse 
    matrix at each timestep that will allow us to solve the entire unstructured grid all at once. 

    We will solve an implicit Advection-Diffusion (transport) equation for the fractional total-load 
    concentrations. This discretization produces a linear system of equations that can be represented by 
    a sprase-matrix problem. 
    """
                
    def update_values(self, mesh: xr.Dataset, t: float):
        """
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
        """
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
        """
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
        """
        self.conc = np.zeros(len(mesh['nface']))
        self.conc = inp[t] 
        self.vals = np.zeros(len(mesh['nface']))
        seconds = mesh['dt'].values[t] 
        # SHOULD GHOST VOLUMES BE INCLUDED?
        vol = mesh['volume'][t] + mesh['ghost_volumes_in'][t]
        self.vals[:] = vol / seconds * self.conc 
        # self.vals[:] = mesh['volume'][t] / seconds * self.conc 

    def update_values(self, solution: np.array, mesh: xr.Dataset, t: float, inp: np.array):
        """ 
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

        """
        seconds = mesh['dt'].values[t] 
        # solution += inp[t][:] 
        # try replacing boundary values instead of adding them:
        solution[inp[t].nonzero()] = inp[t][inp[t].nonzero()] 
        vol = mesh['volume'][t] + mesh['ghost_volumes_in'][t]
        self.vals[:] = solution * vol / seconds
        # self.vals[:] = solution * mesh['volume'][t] / seconds
        return

class ClearwaterRiverine:
    def __init__(self, ras_file_path: str, diffusion_coefficient_input: float, verbose: bool = False) -> None:
        """
        Initialize a Clearwater Riverine WQ model by reading HDF output from a RAS2D model. 

        Parameters:
            hdf_fpath (str):   Filepath to RAS2D HDF output
            diffusion_coefficient_input (float):    User-defined diffusion coefficient for entire modeling domain. 

        """
        # mesh_data = MeshManager(diffusion_coefficient_input)
        # ras_data = RASInput(hdf_fpath, mesh_data)
        # reader = input.ObjectSerializer()
        # reader.read_to_xarray(data, file_path)

        # with h5py.File(hdf_fpath, 'r') as infile:
        #     self.project_name = parse_project_name(infile)
        #     self.mesh = populate_ugrid(infile, self.project_name, diffusion_coefficient_input)
        #     self.boundary_data = populate_boundary_information(infile)

        # define model mesh
        self.mesh = model_mesh(diffusion_coefficient_input)
        if verbose: print("Populating Model Mesh...")
        self.mesh = self.mesh.cwr.read_ras(ras_file_path)
        self.boundary_data = self.mesh.attrs['boundary_data']
        if verbose: print("Calculating Required Parameters...")
        self.mesh = self.mesh.cwr.calculate_required_parameters()

    def initial_conditions(self, fpath: str):
        """
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
        """
        init = pd.read_csv(fpath)
        init['Cell_Index'] = init.Cell_Index.astype(int)
        self.input_array = np.zeros((len(self.mesh.time), len(self.mesh.nface)))
        self.input_array[0, [init['Cell_Index']]] =  init['Concentration']
        return 

    def boundary_conditions(self, fpath: str):
        """
        Define boundary conditions for RAS2D water quality model from CSV file. 

        Parameters:
            fpath (str):    Filepath to CSV containing boundary conditions. The CSV should have the following columns:
                                RAS2D_TS_Name (the timeseries name, as labeled in the RAS model)
                                Datetime
                                Concentration 
        """
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


    def simulate_wq(self, input_mass_units = 'mg', input_volume_units = 'L', input_liter_conversion = 1, save = False, 
                        output_file_path = './clearwater-riverine-wq-model.zarr'):
        """
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
 
        """
        print("Starting WQ Simulation...")

        # Convert Units
        units = _determine_units(self.mesh)

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
            # lhs = LHS(self.mesh, t)
            lhs = LHS()
            lhs.update_values(self.mesh, t)
            A = csr_matrix( (lhs.coef,(lhs.rows, lhs.cols)), shape=(len(self.mesh['nface']),len(self.mesh['nface'])))
            x = linalg.spsolve(A, b.vals)
            b.update_values(x, self.mesh, t+1, self.inp_converted)
            output[t+1] = b.vals

        print(' 100%')
        self.mesh['pollutant_load'] = _hdf_to_xarray(output, dims=('time', 'nface'), attrs={'Units': f'{input_mass_units}/s'})  
        temp_vol = self.mesh['volume'] + self.mesh['ghost_volumes_in']
        concentration = self.mesh['pollutant_load'] / temp_vol * conversion_factor * input_liter_conversion * self.mesh['dt']
        self.mesh['concentration'] = _hdf_to_xarray(concentration, dims = ('time', 'nface'), attrs={'Units': f'{input_mass_units}/L'})

        # may need to move this if we want to plot things besides concentration
        self.max_value = int(self.mesh['concentration'].sel(nface=slice(0, self.mesh.attrs['nreal'])).max())

        if save == True:
            self.mesh.cwr.save_clearwater_xarray(output_file_path)


    def prep_plot(self, crs: str):
        """
        Creates a geodataframe of polygons to represent each RAS cell. 

        Parameters:
            crs:       coordinate system of RAS project.

        Notes:
            Could we parse the CRS from the PRJ file?
        """

        nreal_index = self.mesh.attrs['nreal'] + 1
        real_face_node_connectivity = self.mesh.face_nodes[0:nreal_index]

        # Turn real mesh cells into polygons
        polygon_list = []
        for cell in real_face_node_connectivity:
            xs = self.mesh.node_x[cell[np.where(cell != -1)]]
            ys = self.mesh.node_y[cell[np.where(cell != -1)]]
            p1 = Polygon(list(zip(xs.values, ys.values)))
            polygon_list.append(p1)

        poly_gdf = gpd.GeoDataFrame({'geometry': polygon_list},
                                    crs = crs)
        poly_gdf = poly_gdf.to_crs('EPSG:4326')
        
        gdf_ls = []
        for t in range(len(self.mesh.time)):
            temp_gdf = gpd.GeoDataFrame({'cell': self.mesh.nface[0:nreal_index],
                                        'datetime': pd.to_datetime(self.mesh.time[t].values),
                                        'concentration': self.mesh.concentration[t][0:nreal_index],
                                        'volume': self.mesh.volume[t][0:nreal_index],
                                        'cell': self.mesh.nface[0:nreal_index],
                                        'geometry': poly_gdf['geometry']}, 
                                        crs = 'EPSG:4326')
            gdf_ls.append(temp_gdf)
        full_df = pd.concat(gdf_ls)
        # full_df.to_crs('EPSG:4326')
        self.gdf = full_df
        return

    def plot(self):
        """
        Creates a dynamic polygon plot of concentrations in the RAS2D model domain.

        Parameters:
            None

        Notes:
            Play button
            Move re-projection? This is really slow, but I think geoviews requires ESPG:4326 so necessary at some point. 
            Option to save
            Build in functionality to pass plotting arguments (clim, cmap, height, width, etc.)
            Input parameter of info to plot?
        """
        def map_generator(datetime, mval=self.max_value):
            """This function generates plots for the DynamicMap"""
            ras_sub_df = self.gdf[self.gdf.datetime == datetime]
            ras_map = gv.Polygons(ras_sub_df, vdims=['concentration']).opts(height=600,
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
        """
        Creates a dynamic scatterplot of cell centroids colored by cell concentration.

        Parameters:

        Notes:
            Play button
            Move re-projection? This is really slow, but I think geoviews requires ESPG:4326 so necessary at some point. 
            Option to save
            Build in functionality to pass plotting arguments (clim, cmap, height, width, etc.)
            Input parameter of info to plot?
        """

        def quick_map_generator(datetime, mval=self.max_value):
            """This function generates plots for the DynamicMap"""
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
