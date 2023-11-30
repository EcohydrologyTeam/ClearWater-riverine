import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix, linalg
import holoviews as hv
import geoviews as gv
import geopandas as gpd
from shapely.geometry import Polygon
hv.extension("bokeh")
from typing import Optional

from clearwater_riverine.mesh import model_mesh
from clearwater_riverine import variables
from clearwater_riverine.utilities import UnitConverter
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
    by leveraging hydrodynamic output from HEC-RAS 2D. The Clearwater Riverine model mesh is an xarray
    following UGRID conventions. 

    Args:
        ras_file_path (str):  Filepath to HEC-RAS output
        diffusion_coefficient_input (float): User-defined diffusion coefficient for entire modeling domain. 
        verbose (bool, optional): Boolean indicating whether or not to print model progress. 

    Attributes:
        mesh (xr.Dataset): Unstructured model mesh containing relevant HEC-RAS outputs, calculated parameters
            required for advection-diffusion calculations, and water quality ouptuts (e.g., concentration). 
            The unstructured mesh follows UGRID CF Conventions. 
        boundary_data (pd.DataFrame): Information on RAS model boundaries, extracted directly from HEC-RAS 2D output. 
    """

    def __init__(self, ras_file_path: str, diffusion_coefficient_input: float, verbose: bool = False) -> None:
        """ Initialize a Clearwater Riverine WQ model mesh by reading HDF output from a RAS2D model to an xarray."""
        self.gdf = None

        # define model mesh
        self.mesh = model_mesh(diffusion_coefficient_input)
        if verbose: print("Populating Model Mesh...")
        self.mesh = self.mesh.cwr.read_ras(ras_file_path)
        self.boundary_data = self.mesh.attrs['boundary_data']
        if verbose: print("Calculating Required Parameters...")
        self.mesh = self.mesh.cwr.calculate_required_parameters()
        self.input_array = np.zeros((len(self.mesh.time), len(self.mesh.nface)))


    def initial_conditions(self, filepath: str):
        """Define initial conditions for Clearwater Riverine model from a CSV file. 

        Args:
            filepath (str): Filepath to a CSV containing initial conditions. The CSV should have two columns:
                one called `Cell_Index` and one called `Concentration`. The file should the concentration
                in each cell within the model domain at the first timestep. 
        """
        init = pd.read_csv(filepath)
        init['Cell_Index'] = init.Cell_Index.astype(int)
        self.input_array[0, [init['Cell_Index']]] =  init['Concentration']
        return 

    def boundary_conditions(self, filepath: str):
        """Define boundary conditions for Clearwater Riverine model from a CSV file. 

        Args:
            filepath (str): Filepath to a CSV containing boundary conditions. The CSV should have the following columns:
                `RAS2D_TS_Name` (the timeseries name, as labeled in the HEC-RAS model), `Datetime`, `Concentration`. 
                This file should contain the concentration for all relevant boundary cells at every RAS timestep. 
                If a timestep / boundary cell is not included in this CSV file, the concentration will be set to 0
                in the Clearwater Riverine model.  
        """
        # Read in boundary condition data from user
        bc_df = pd.read_csv(
            filepath,
            parse_dates=['Datetime']
        )

        xarray_time_index = pd.DatetimeIndex(
            self.mesh.time.values
        )
        model_dataframe = pd.DataFrame({
            'Datetime': xarray_time_index,
            'Time Index': range(len(xarray_time_index))
        })

        result_df = pd.DataFrame()
        for boundary, group_df in bc_df.groupby('RAS2D_TS_Name'):
            # Merge with model timestep
            merged_group = pd.merge_asof(
                model_dataframe,
                group_df,
                on='Datetime'
            )
            # Interpolate
            merged_group['Concentration'] = merged_group['Concentration'].interpolate(method='linear')
            # Append to dataframe
            result_df = pd.concat(
                [result_df, merged_group], 
                ignore_index=True
            )
        
        # Merge with boundary data
        boundary_df = pd.merge(
            result_df,
            self.boundary_data,
            left_on = 'RAS2D_TS_Name',
            right_on = 'Name',
            how='left'
        )
        boundary_df['Ghost Cell'] = self.mesh.edges_face2[boundary_df['Face Index'].to_list()]
        boundary_df['Domain Cell'] = self.mesh.edges_face1[boundary_df['Face Index'].to_list()]

        # Assign to appropriate position in array
        self.input_array[[boundary_df['Time Index']], [boundary_df['Ghost Cell']]] = boundary_df['Concentration']

    def initialize(
        self,
        initial_condition_path: Optional[str] = None,
        boundary_condition_path: Optional[str] = None,
        units: Optional[str] = None,
    ):
        """Initializes model, developed to be BMI-adjacent.

        Args:
            initial_conditon_path: Filepath to a CSV containing initial conditions. 
                The CSV should have two columns: one called `Cell_Index` and one called
                `Concentration`. The file should the concentration in each cell within 
                the modeldomain at the first timestep. If not provided, no initial conditions
                will be set up as part of the initialize call.
            boundary_condition_path: Filepath to a CSV containing boundary conditions. 
                The CSV should have the following columns: `RAS2D_TS_Name` (the timeseries
                name, as labeled in the HEC-RAS model), `Datetime`, `Concentration`. 
                This file should contain the concentration for all relevant boundary cells
                at every RAS timestep. If a timestep / boundary cell is not included in this
                CSV file, the concentration will be set to 0 in the Clearwater Riverine model.
                If not provide,d no boundary condtiions will be set up as part of the initialize
                call. 
            units: the units of the concentration timeseries. If not provided, defaults to
                'Unknown.'
        """
        # Set initial and boundary conditions if provided
        if initial_condition_path:
            self.initial_conditions(initial_condition_path)
        if boundary_condition_path:
            self.boundary_conditions(boundary_condition_path)

        if not units:
            units = 'unknown'

        self.time_step = 0
        self.concentrations = np.zeros((len(self.mesh.time), len(self.mesh.nface)))
        self.concentrations[0] = self.input_array[0]
        self.mesh[variables.CONCENTRATION] = _hdf_to_xarray(
            self.concentrations,
            dims = ('time', 'nface'),
            attrs={'Units': f'{units}'})
        self.mesh[variables.CONCENTRATION][self.time_step][:] = self.concentrations[0]

        self.b = RHS(self.mesh, self.input_array)
        self.lhs = LHS(self.mesh)
    
    def update(
        self,
        update_concentration: Optional[dict[str, xr.DataArray]] = None,
    ):
        """Update a single timestep."""
        # Allow users to override concentration
        if update_concentration:
            for var_name, value in update_concentration.items():
                self.mesh['concentration'][self.time_step][0: self.mesh.nreal+1] = update_concentration[var_name].values[0:self.mesh.nreal + 1]
                x = update_concentration[var_name].values[0:self.mesh.nreal + 1]
        else:
            x = self.concentrations[self.time_step][0:self.mesh.nreal + 1]
        
        # Update the right hand side of the matrix 
        self.b.update_values(
            x,
            self.mesh,
            self.time_step
        )

        # Update the left hand side of the matrix 
        self.lhs.update_values(
            self.mesh,
            self.time_step
        )

        # Define compressed sparse row matrix
        A = csr_matrix(
            (self.lhs.coef, (self.lhs.rows, self.lhs.cols)),
            shape=(self.mesh.nreal + 1, self.mesh.nreal + 1)
        )

        # Solve
        x = linalg.spsolve(A, self.b.vals)

        # Update timestep and save data
        self.time_step += 1
        self.concentrations[self.time_step][0:self.mesh.nreal+1] = x
        self.concentrations[self.time_step][self.input_array[self.time_step].nonzero()] = self.input_array[self.time_step][self.input_array[self.time_step].nonzero()] 
        self.mesh['concentration'][self.time_step][:] = self.concentrations[self.time_step]

    def simulate_wq(
        self,
        input_mass_units: str = 'mg',
        input_volume_units: str = 'L',
        input_liter_conversion: float = 1,
        save: bool = False, 
        output_file_path: str = './clearwater-riverine-wq-model.zarr'
    ):
        """Runs water quality model. 

        Steps through each timestep of the HEC-RAS 2D output and solves the total-load advection-diffusion transport equation 
        using user-defined boundary and initial conditions. Users must use `initial_conditions()` and `boundary_conditions()` 
        methods before calling `simulate_wq()` or all concentrations will be 0. 

        Args:
            input_mass_units (str, optional): User-defined mass units for concentration timeseries used in model set-up. Assumes mg if no value
                is specified. 
            input_volume_units (str, optional): User-defined volume units for concentration timeseries. Assumes L if no value
                is specified.
            input_liter_conversion (float, optional): If concentration inputs are not in mass/L, supply the conversion factor to 
                convert the volume unit to liters.
            save (bool, optional): Boolean indicating whether the file should be saved. Default is to not save the output.
            output_file_path (str, optional): Filepath where the output file should be stored. Default to save in current directory as 
                `clearwater-riverine-wq.zarr`
 
        """
        print("Starting WQ Simulation...")

        # Convert Units
        unit_converter = UnitConverter(self.mesh, input_mass_units, input_volume_units, input_liter_conversion)
        self.inp_converted = unit_converter._convert_units(self.input_array, convert_to=True)
        # self.inp_converted = self.input_array / input_liter_conversion / conversion_factor # convert to mass/ft3 or mass/m3 

        output = np.zeros((len(self.mesh.time), self.mesh.nreal + 1))
        advection_mass_flux = np.zeros((len(self.mesh.time), len(self.mesh.nedge)))
        diffusion_mass_flux = np.zeros((len(self.mesh.time), len(self.mesh.nedge)))
        total_mass_flux = np.zeros((len(self.mesh.time), len(self.mesh.nedge)))
        concentrations = np.zeros((len(self.mesh.time), len(self.mesh.nface)))

        b = RHS(self.mesh, self.inp_converted)
        lhs = LHS(self.mesh)
        concentrations[0] = self.inp_converted[0]
        x = concentrations[0][0:self.mesh.nreal + 1]

        # loop over time to solve
        for t in range(len(self.mesh['time']) - 1):
            self.time_step = t
            self._timer(t)
            b.update_values(x, self.mesh, t)
            lhs.update_values(self.mesh, t)
            A = csr_matrix((lhs.coef,(lhs.rows, lhs.cols)), shape=(self.mesh.nreal + 1, self.mesh.nreal + 1))
            x = linalg.spsolve(A, b.vals)
            # reactions would go here
            concentrations[t+1][0:self.mesh.nreal+1] = x
            concentrations[t+1][self.inp_converted[t].nonzero()] = self.inp_converted[t][self.inp_converted[t].nonzero()] 
            self._mass_flux(concentrations, advection_mass_flux, diffusion_mass_flux, total_mass_flux, t)

        print(' 100%')
        concentrations_converted = unit_converter._convert_units(concentrations, convert_to=False)
        self.mesh[variables.CONCENTRATION] = _hdf_to_xarray(concentrations_converted, dims = ('time', 'nface'), attrs={'Units': f'{input_mass_units}/{input_volume_units}'})

        # add advection / diffusion mass flux
        self.mesh['mass_flux_advection'] = _hdf_to_xarray(advection_mass_flux, dims=('time', 'nedge'), attrs={'Units': f'{input_mass_units}'})
        self.mesh['mass_flux_diffusion'] = _hdf_to_xarray(diffusion_mass_flux, dims=('time', 'nedge'), attrs={'Units': f'{input_mass_units}'})
        self.mesh['mass_flux_total'] = _hdf_to_xarray(total_mass_flux, dims=('time', 'nedge'), attrs={'Units': f'{input_mass_units}'})

        # may need to move this if we want to plot things besides concentration
        self.max_value = int(self.mesh[variables.CONCENTRATION].sel(nface=slice(0, self.mesh.attrs[variables.NUMBER_OF_REAL_CELLS])).max())
        self.min_value = int(self.mesh[variables.CONCENTRATION].sel(nface=slice(0, self.mesh.attrs[variables.NUMBER_OF_REAL_CELLS])).min())

        if save == True:
            self.mesh.cwr.save_clearwater_xarray(output_file_path)
    
    def _timer(self, t):
        if t == int(len(self.mesh['time']) / 4):
            print(' 25%')
        elif t == int(len(self.mesh['time']) / 2):
            print(' 50%')
        if t == int(3 * len(self.mesh['time']) / 4):
            print(' 75%')

    def _mass_flux(self, output, advection_mass_flux, diffusion_mass_flux, total_mass_flux, t):
        negative_condition = self.mesh[variables.ADVECTION_COEFFICIENT][t] < 0
        parent = output[t][self.mesh[variables.EDGES_FACE1]]
        neighbor = output[t][self.mesh[variables.EDGES_FACE2]]

        advection_mass_flux[t] = xr.where(
            negative_condition,
            self.mesh[variables.ADVECTION_COEFFICIENT][t] * parent,
            self.mesh[variables.ADVECTION_COEFFICIENT][t] * neighbor
        )

        diffusion_mass_flux[t] = self.mesh[variables.COEFFICIENT_TO_DIFFUSION_TERM][t] * (neighbor - parent)
        total_mass_flux[t] = advection_mass_flux[t] + diffusion_mass_flux[t]

    def _prep_plot(self, crs: str):
        """ Creates a geodataframe of polygons to represent each RAS cell. 

        Args:
            crs: coordinate system of RAS project.

        Notes:
            Could we parse the CRS from the PRJ file?
        """

        self.nreal_index = self.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] + 1
        real_face_node_connectivity = self.mesh.face_nodes[0:self.nreal_index]

        # Turn real mesh cells into polygons
        polygon_list = []
        for cell in real_face_node_connectivity:
            xs = self.mesh.node_x[cell[np.where(cell != -1)]]
            ys = self.mesh.node_y[cell[np.where(cell != -1)]]
            p1 = Polygon(list(zip(xs.values, ys.values)))
            polygon_list.append(p1)

        poly_gdf = gpd.GeoDataFrame({
            'nface': self.mesh.nface[0:self.nreal_index],
            'geometry': polygon_list},
            crs = crs)
        self.poly_gdf = poly_gdf.to_crs('EPSG:4326')
        self._update_gdf()
        
        # gdf_ls = []

        # for t in range(len(self.mesh.time)):
        #     temp_gdf = gpd.GeoDataFrame({'cell': self.mesh.nface[0:nreal_index],
        #                                 'datetime': pd.to_datetime(self.mesh.time[t].values),
        #                                 'concentration': self.mesh.concentration.isel(time=t, nface=slice(0,nreal_index)),
        #                                 'volume': self.mesh.volume.isel(time=t, nface=slice(0,nreal_index)),
        #                                 'cell': self.mesh.nface[0:nreal_index],
        #                                 'geometry': poly_gdf['geometry']}, 
        #                                 crs = 'EPSG:4326')
        #     gdf_ls.append(temp_gdf)
        
    def _update_gdf(self):
        """Update gdf values."""
        print('set plotting timestep equal to timestep')
        self.plotting_time_step = self.time_step
        print(self.plotting_time_step, self.time_step)

        df_from_array = self.mesh['concentration'].isel(
            nface=slice(0,self.nreal_index)
            ).to_dataframe()
        df_from_array.reset_index(inplace=True)
        self.df_merged = gpd.GeoDataFrame(
            pd.merge(
                df_from_array,
                self.poly_gdf,
                on='nface',
                how='left'
            )
        )
        self.df_merged.rename(
            columns={
                'nface':'cell',
                'time': 'datetime'
            },
            inplace=True
        )
        self.gdf = self.df_merged


    def _maximum_plotting_value(self, clim_max) -> float:
        """ Calculate the maximum value for color bar. 
        
        Uses the maximum concentration value in the model mesh if no user-defined  clim_max is specified,
        otherwise defines the maximum value as clim_max. 

        Args:
            clim_max (float): user defined maximum colorbar value or default (None)
        
        Returns:
            mval (float): maximum plotting value, either based on user input or the maximum concentration value.
        """
        if clim_max != None:
            mval = clim_max
        else:
            mval = self.max_value
        return mval

    def _minimum_plotting_value(self, clim_min) -> float:
        """ Calculate the maximum value for color bar. 
        
        Uses the maximum concentration value in the model mesh if no user-defined  clim_max is specified,
        otherwise defines the maximum value as clim_max. 

        Args:
            clim_min (float): user defined minimum colorbar value or default (None)
        
        Returns:
            mval (float): minimum plotting value, either based on user input or the minimum concentration value.
        """
        if clim_min != None:
            mval = clim_min
        else:
            mval = self.min_value
        return mval

    def plot(self, crs: str = None, clim: tuple = (None, None), time_index_range: tuple = (0, -1)):
        """Creates a dynamic polygon plot of concentrations in the RAS2D model domain.

        The `plot()` method takes slightly  more time than the `quick_plot()` method in order to leverage the `geoviews` plotting library. 
        The `plot()` method creates more detailed and aesthetic plots than the `quick_plot()` method. 

        Args:
            crs (str): coordinate system of the HEC-RAS 2D model. Only required the first time you call this method.  
            clim_max (float, optional): maximum value for color bar. If not specifies, the default will be the 
                maximum concentration value in the model domain over the entire simulation horizon. 
            time_index_range (tuple, optional): minimum and maximum time index to plot.
        """

        if type(self.gdf) != gpd.geodataframe.GeoDataFrame:
            if crs == None:
                raise ValueError("This is your first time running the plot function. You must specify a crs!")
            else:
                self._prep_plot(crs)
        
        if self.plotting_time_step != self.time_step:
            print('updating GDF')
            self._update_gdf()

        mval = self._maximum_plotting_value(clim[1])
        mn_val = self._minimum_plotting_value(clim[0])

        def map_generator(datetime, mval=mval):
            """This function generates plots for the DynamicMap"""
            ras_sub_df = self.gdf[self.gdf.datetime == datetime]
            units = self.mesh[variables.CONCENTRATION].Units
            ras_map = gv.Polygons(
                ras_sub_df,
                vdims=['concentration', 'cell']).opts(
                    height = 400,
                    width = 800,
                    color='concentration',
                    colorbar = True,
                    cmap = 'OrRd',
                    clim = (mn_val, mval),
                    line_width = 0.1,
                    tools = ['hover'],
                    clabel = f"Concentration ({units})"
            )
            return (ras_map * gv.tile_sources.CartoLight())

        dmap = hv.DynamicMap(map_generator, kdims=['datetime'])
        return dmap.redim.values(datetime=self.gdf.datetime.unique()[time_index_range[0]: time_index_range[1]])

    def quick_plot(self, clim: tuple = (None,None)):
        """Creates a dynamic scatterplot of cell centroids colored by cell concentration.

        The `quick_plot()` method is meant to rapidly develop visualizations to explore results. 
        Use the `plot()` method for more aesthetic plots. 

        Args:
            clim_max (float, optional): maximum value for color bar. 
        """

        mval = self._maximum_plotting_value(clim[1])
        mn_val = self._minimum_plotting_value(clim[0])

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
                                                                                    cmap = 'OrRd', 
                                                                                    clim = (mn_val, mval),
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