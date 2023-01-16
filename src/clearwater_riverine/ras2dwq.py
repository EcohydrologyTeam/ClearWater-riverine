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
from clearwater_riverine import variables
from clearwater_riverine.utilities import _determine_units
from clearwater_riverine.linalg import LHS, RHS
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



class ClearwaterRiverine:
    """ Creates Clearwater Riverine water quality model.

    Clearwater Riverine is a water quality model that calculates advection and diffusion of constituents
        by leveraging hydrodynamic output from HEC-RAS 2D. 

    Attributes:
        mesh (xr.Dataset): unstructured model mesh containing relevant HEC-RAS outputs, calculated parameters
            required for advection-diffusion calculations, and water quality ouptuts (e.g., concentration). 
            The unstructured mesh follows UGRID CF Conventions. 
        boundary_data (pd.DataFrame): information on RAS model boundaries, extracted directly from RAS HDF output. 



    """

    def __init__(self, ras_file_path: str, diffusion_coefficient_input: float, verbose: bool = False) -> None:
        """ Initialize a Clearwater Riverine WQ model mesh by reading HDF output from a RAS2D model to an xarray.

        Args:
            ras_file_path (str):   Filepath to RAS2D HDF output
            diffusion_coefficient_input (float):    User-defined diffusion coefficient for entire modeling domain. 
            verbose (bool): Boolean indicating whether or not to print model progress. 
        """
        self.gdf = None

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

        Args:
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
            Refactor IO to allow alternative use cases / file formats (OOP)
        """
        init = pd.read_csv(fpath)
        init['Cell_Index'] = init.Cell_Index.astype(int)
        self.input_array = np.zeros((len(self.mesh.time), len(self.mesh.nface)))
        self.input_array[0, [init['Cell_Index']]] =  init['Concentration']
        return 

    def boundary_conditions(self, fpath: str):
        """
        Define boundary conditions for RAS2D water quality model from CSV file. 

        Args:
            fpath (str): Filepath to CSV containing boundary conditions. The CSV should have the following columns:
                RAS2D_TS_Name (the timeseries name, as labeled in the RAS model), Datetime, Concentration 
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


    def simulate_wq(self, input_mass_units: str = 'mg', input_volume_units: str = 'L', input_liter_conversion: float = 1, save: bool = False, 
                        output_file_path: str = './clearwater-riverine-wq-model.zarr'):
        """
        Steps through each timestep in the output of a RAS2D model (mesh) 
        and solves the total-load advection-diffusion transport equation 
        using boundary and initial conditions.

        Args:
            input_mass_units (str): User-defined mass units for concentration timeseries. Assumes mg if no value
                is specified. 
            input_volume_units (str): User-defined volume units for concentration timeseries. Assumes L if no value
                is specified.
            input_liter_conversion (float): If concentration inputs are not in mass/L, supply the conversion factor to 
                convert the volume unit to liters.
            save (bool): Boolean indicating whether the file should be saved. Default is to not save the output.
            output_file_path (str): Filepath where the output file should be stored. Default to save in current directory.
 
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

        # loop over time to solve
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
        self.mesh[variables.POLLUTANT_LOAD] = _hdf_to_xarray(output, dims=('time', 'nface'), attrs={'Units': f'{input_mass_units}/s'})  
        temp_vol = self.mesh[variables.VOLUME] + self.mesh[variables.GHOST_CELL_VOLUMES_IN]
        concentration = self.mesh[variables.POLLUTANT_LOAD] / temp_vol * conversion_factor * input_liter_conversion * self.mesh[variables.CHANGE_IN_TIME]
        self.mesh[variables.CONCENTRATION] = _hdf_to_xarray(concentration, dims = ('time', 'nface'), attrs={'Units': f'{input_mass_units}/L'})

        # may need to move this if we want to plot things besides concentration
        self.max_value = int(self.mesh[variables.CONCENTRATION].sel(nface=slice(0, self.mesh.attrs[variables.NUMBER_OF_REAL_CELLS])).max())

        if save == True:
            self.mesh.cwr.save_clearwater_xarray(output_file_path)


    def _prep_plot(self, crs: str):
        """ Creates a geodataframe of polygons to represent each RAS cell. 

        Args:
            crs: coordinate system of RAS project.

        Notes:
            Could we parse the CRS from the PRJ file?
        """

        nreal_index = self.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] + 1
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
        self.gdf = full_df


    def _maximum_plotting_value(self, clim_max) -> float:
        """ Calculate the maximum value for color bar. 
        
        Uses the maximum concentration value in the model mesh if no user-defined  clim_max is specified,
        otherwise defines the maximum value as clim_max. 
        """
        if clim_max != None:
            mval = clim_max
        else:
            mval = self.max_value
        return mval

    def plot(self, crs: str = None, clim_max: float = None):
        """
        Creates a dynamic polygon plot of concentrations in the RAS2D model domain.

        Args:
            crs (str): coordinate system of the RAS HDF output. 
            clim_max (float): maximum value for color bar. 

        Notes:
            Play button
            Move re-projection? This is really slow, but I think geoviews requires ESPG:4326 so necessary at some point. 
            Option to save
            Build in functionality to pass plotting arguments (clim, cmap, height, width, etc.)
            Input parameter of info to plot?
        """
        if type(self.gdf) != gpd.geodataframe.GeoDataFrame:
            if crs == None:
                raise ValueError("This is your first time running the plot function. You must specify a crs!")
            else:
                self._prep_plot(crs)

        mval = self._maximum_plotting_value(clim_max)

        def map_generator(datetime, mval=mval):
            """This function generates plots for the DynamicMap"""
            ras_sub_df = self.gdf[self.gdf.datetime == datetime]
            units = self.mesh[variables.CONCENTRATION].Units
            ras_map = gv.Polygons(ras_sub_df, vdims=['concentration']).opts(height=600,
                                                                          width = 800,
                                                                          color='concentration',
                                                                          colorbar = True,
                                                                          cmap = 'OrRd', 
                                                                          clim = (0, mval),
                                                                          line_width = 0.1,
                                                                          tools = ['hover'],
                                                                          clabel = f"Concentration ({units})"
                                                                       )
            return (ras_map * gv.tile_sources.CartoLight())

        dmap = hv.DynamicMap(map_generator, kdims=['datetime'])
        return dmap.redim.values(datetime=self.gdf.datetime.unique())

    def quick_plot(self, clim_max: float = None):
        """
        Creates a dynamic scatterplot of cell centroids colored by cell concentration.

        Args:
            clim_max (float): maximum value for color bar. 

        Notes:
            Play button
            Move re-projection? This is really slow, but I think geoviews requires ESPG:4326 so necessary at some point. 
            Option to save
            Build in functionality to pass plotting arguments (clim, cmap, height, width, etc.)
            Input parameter of info to plot?
        """

        mval = self._maximum_plotting_value(clim_max)

        def quick_map_generator(datetime, mval=mval):
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