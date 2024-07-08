import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix, linalg
import holoviews as hv
import geoviews as gv
import geopandas as gpd
import yaml
from shapely.geometry import Polygon
hv.extension("bokeh")
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
)
from pathlib import Path

from clearwater_riverine.mesh import model_mesh
from clearwater_riverine import variables
from clearwater_riverine.variables import (
    ADVECTION_COEFFICIENT,
    COEFFICIENT_TO_DIFFUSION_TERM,
    EDGES_FACE1,
    EDGES_FACE2,
    CHANGE_IN_TIME,
    NUMBER_OF_REAL_CELLS,
    CONCENTRATION
)
from clearwater_riverine.utilities import UnitConverter
from clearwater_riverine.linalg import LHS, RHS
from clearwater_riverine.io.hdf import _hdf_to_xarray
from clearwater_riverine.io.config import parse_config
from clearwater_riverine.constituents import Constituent

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

    def __init__(
        self,
        flow_field_file_path: Optional[str | Path] = None,
        diffusion_coefficient_input: Optional[float] = None,
        constituent_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        config_filepath: Optional[str] = None,
        verbose: Optional[bool] = False,
        datetime_range: Optional[Tuple[int, int] | Tuple[str, str]] = None,

    ) -> None:
        """
        Initialize a Clearwater Riverine WQ model mesh
        reading HDF output from a RAS2D model to an xarray.
        """
        self.gdf = None
        self.time_step = 0

        if config_filepath:
            model_config = parse_config(config_filepath=config_filepath)
            if diffusion_coefficient_input is None:
                diffusion_coefficient_input = model_config['diffusion_coefficient']
            if not flow_field_file_path:
                flow_field_file_path = model_config['flow_field_filepath']
            self.constituents = model_config['constituents'].keys()
        else:
            if flow_field_file_path:
                ## TODO: add some checking that input set up correctly
                if isinstance(constituent_dict, Dict):
                    self.constituents = constituent_dict
                    model_config = {'constituents': constituent_dict}
            else:
                raise TypeError(
                    'Missing a `config_filepath` or a `constituent_dict` and `flow_field_file_path` to run the model.'
                )
           
        self.contsituent_dict = {}

        # define model mesh
        self.mesh = model_mesh(diffusion_coefficient_input)
        if verbose: print("Populating Model Mesh...")
        self.mesh = self.mesh.cwr.read_ras(
            flow_field_file_path,
            datetime_range=datetime_range
        )
        self.boundary_data = self.mesh.attrs['boundary_data']

        if verbose: print("Calculating Required Parameters...")
        self.mesh = self.mesh.cwr.calculate_required_parameters()
    
        self.lhs = LHS(self.mesh)
        self.initialize_constituents(
            model_config=model_config
        )

    def initialize_constituents(
        self,
        model_config: Dict,
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
                If not provided no boundary condtiions will be set up as part of the initialize
                call. 
            units: the units of the concentration timeseries. If not provided, defaults to
                'Unknown.'
        """
        self.time_step = 0
        self.constituent_dict = {}

        for constituent in self.constituents:
            self.constituent_dict[constituent] = Constituent(
                name=constituent,
                constituent_config=model_config['constituents'][constituent],
                mesh=self.mesh,
                flow_field_boundaries=self.boundary_data,
            )
    
    def update(
        self,
        update_concentration: Optional[dict[str, xr.DataArray]] = None,
    ):
        """Update a single timestep."""

        # Update the left hand side of the matrix
        # This is the same for all constituents
        self.lhs.update_values(
            self.mesh,
            self.time_step
        )

        # Define compressed sparse row matrix for LHS
        A = csr_matrix(
            (self.lhs.coef, (self.lhs.rows, self.lhs.cols)),
            shape=(self.mesh.nreal + 1, self.mesh.nreal + 1)
        )

        # Check if constituent_name from update_concentration dict is one
        # of the constituents in the model
        
        if isinstance(update_concentration, dict):
            for update_constituent_name, _ in update_concentration.items():
                if update_constituent_name in self.constituent_dict:
                    pass
                else:
                    print(f"WARNING: {update_constituent_name} is not being used in the model.")
                    print("Please review the constituent names in the update dictionary")

        for constituent_name, constituent in self.constituent_dict.items():
            # Allow users to override concentration
            if isinstance(update_concentration, dict) and constituent_name in update_concentration.keys():
                self.mesh[constituent_name][self.time_step][0: self.mesh.nreal + 1] = \
                    update_concentration[constituent_name].values[0:self.mesh.nreal + 1]
                x = update_concentration[constituent_name].values[0:self.mesh.nreal + 1]
            else:
                x = self.mesh[constituent_name][self.time_step][0:self.mesh.nreal + 1]
        
            # Update the right hand side of the matrix 
            constituent.b.update_values(
                solution=x,
                mesh=self.mesh,
                t=self.time_step,
                name=constituent_name,
            )

            # Solve
            x = linalg.spsolve(A, constituent.b.vals)

            # Update timestep and save data
            self.mesh[constituent_name].loc[
                {
                    'time': self.mesh.time[self.time_step + 1],
                    'nface': self.mesh.nface.values[0:self.mesh.nreal+1]
                }
            ] = x
            nonzero_indices = np.nonzero(constituent.input_array[self.time_step + 1])
            self.mesh[constituent_name].loc[
                {
                    'time': self.mesh.time[self.time_step + 1],
                    'nface': nonzero_indices[0]
                }
            ] = constituent.input_array[self.time_step + 1][nonzero_indices]

            # Calculate mass flux
            self._mass_flux(
                self.mesh[constituent_name],
                constituent.advection_mass_flux,
                constituent.diffusion_mass_flux,
                constituent.total_mass_flux,
                self.time_step
                )

        # increment timestep
        self.time_step += 1


    def simulate_wq(
        self,
        input_mass_units: str = 'mg',
        input_volume_units: str = 'L',
        input_liter_conversion: float = 1,
        save: bool = False, 
        output_file_path: str = './clearwater-riverine-wq-model.zarr'
    ):
        """Deprecated
        
        Runs water quality model. 

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
        # unit_converter = UnitConverter(self.mesh, input_mass_units, input_volume_units, input_liter_conversion)
        # self.inp_converted = unit_converter._convert_units(self.input_array, convert_to=True)
        # self.inp_converted = self.input_array / input_liter_conversion / conversion_factor # convert to mass/ft3 or mass/m3 
        lhs = LHS(self.mesh)
        
        # Loop over time to solve
        for t in range(len(self.mesh['time']) - 1):
            self.time_step = t
            self._timer(t)
            lhs.update_values(self.mesh, t)
            A = csr_matrix(
                (lhs.coef,(lhs.rows, lhs.cols)),
                shape=(self.mesh.nreal + 1, self.mesh.nreal + 1)
            )

            # solve for each constituent
            for constituent_name, constituent in self.constituent_dict.items():
                # Solve sparse matrix
                constituent.b.update_values(
                    solution=x,
                    mesh=self.mesh,
                    t=self.time_step,
                    name=constituent_name,
                    input_array=constituent.input_array
                )
                x = linalg.spsolve(A, constituent.b.vals)

                # Save solution
                self.mesh[constituent_name].loc[
                    t+1, 0:self.mesh.nreal+1
                ] = x
                nonzero_indices = np.nonzero(self.input_array[self.time_step])
                self.mesh[constituent_name].loc[self.time_step, nonzero_indices] = self.input_array[self.time_step][nonzero_indices]

                self._mass_flux(
                    self.mesh[constituent_name],
                    constituent.advection_mass_flux,
                    constituent.diffusion_mass_flux,
                    constituent.total_mass_flux,
                    t+1
                )
        
        # self._mass_flux(concentrations, advection_mass_flux, diffusion_mass_flux, total_mass_flux, t+1)
        # concentrations_converted = unit_converter._convert_units(concentrations, convert_to=False)
        # self.mesh[CONCENTRATION] = _hdf_to_xarray(concentrations_converted, dims = ('time', 'nface'), attrs={'Units': f'{input_mass_units}/{input_volume_units}'})

        # # add advection / diffusion mass flux
        # self.mesh['mass_flux_advection'] = _hdf_to_xarray(advection_mass_flux, dims=('time', 'nedge'), attrs={'Units': f'{input_mass_units}'})
        # self.mesh['mass_flux_diffusion'] = _hdf_to_xarray(diffusion_mass_flux, dims=('time', 'nedge'), attrs={'Units': f'{input_mass_units}'})
        # self.mesh['mass_flux_total'] = _hdf_to_xarray(total_mass_flux, dims=('time', 'nedge'), attrs={'Units': f'{input_mass_units}'})

        # # TODO: move this to plot things besides concentration
        # self.max_value = int(self.mesh[CONCENTRATION].sel(nface=slice(0, self.mesh.attrs[NUMBER_OF_REAL_CELLS])).max())
        # self.min_value = int(self.mesh[CONCENTRATION].sel(nface=slice(0, self.mesh.attrs[NUMBER_OF_REAL_CELLS])).min())

        # if save == True:
        #     self.mesh.cwr.save_clearwater_xarray(output_file_path)
    
        print(' 100%')

    def finalize(
        self,
        save: Optional[bool] = False,
        output_filepath: Optional[str] = None
    ):
        for _, constituent in self.constituent_dict.items():
            constituent.set_value_range(self.mesh)            

        if save == True:
            self.mesh.cwr.save_clearwater_xarray(output_filepath)


    def _timer(self, t):
        if t == int(len(self.mesh['time']) / 4):
            print(' 25%')
        elif t == int(len(self.mesh['time']) / 2):
            print(' 50%')
        if t == int(3 * len(self.mesh['time']) / 4):
            print(' 75%')

    def _mass_flux(self,
        output: np.ndarray,
        advection_mass_flux: np.ndarray,
        diffusion_mass_flux: np.ndarray,
        total_mass_flux: np.ndarray,
        t: int,
    ):
        """Calculates mass flux across cell boundaries."""
        negative_condition = self.mesh[ADVECTION_COEFFICIENT].isel(time=t) < 0
        parent_concentration = output[t+1][self.mesh[EDGES_FACE1]]
        neighbor_concentration = output[t+1][self.mesh[EDGES_FACE2]]
        delta_time = self.mesh[CHANGE_IN_TIME].isel(time=t)

        advection_mass_flux[t] = xr.where(
            negative_condition,
            self.mesh[ADVECTION_COEFFICIENT].isel(time=t) * neighbor_concentration,
            self.mesh[ADVECTION_COEFFICIENT].isel(time=t) * parent_concentration,
        ) * delta_time

        diffusion_mass_flux[t] = self.mesh[COEFFICIENT_TO_DIFFUSION_TERM][t] * \
              (neighbor_concentration - parent_concentration) * \
              delta_time

        total_mass_flux[t] = advection_mass_flux[t] + diffusion_mass_flux[t]


    def _prep_plot(self, crs: str):
        """ Creates a geodataframe of polygons to represent each RAS cell. 

        Args:
            crs: coordinate system of RAS project.

        Notes:
            Could we parse the CRS from the PRJ file?
        """

        self.nreal_index = self.mesh.attrs[NUMBER_OF_REAL_CELLS] + 1
        real_face_node_connectivity = self.mesh.face_nodes[0:self.nreal_index]

        # Turn real mesh cells into polygons
        polygon_list = []
        for cell in real_face_node_connectivity:
            xs = self.mesh.node_x[cell[np.where(cell != -1)]]
            ys = self.mesh.node_y[cell[np.where(cell != -1)]]
            p1 = Polygon(list(zip(xs.values, ys.values)))
            polygon_list.append(p1)

        poly_gdf = gpd.GeoDataFrame(
            {
                'nface': self.mesh.nface[0:self.nreal_index],
                'geometry': polygon_list
            },
            crs = crs
        )
        self.poly_gdf = poly_gdf.to_crs('EPSG:4326')
        self._update_gdf()
    
        
    def _update_gdf(self):
        """Update gdf values."""
        self.plotting_time_step = self.time_step

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
            self._update_gdf()

        mval = self._maximum_plotting_value(clim[1])
        mn_val = self._minimum_plotting_value(clim[0])

        def map_generator(datetime, mval=mval):
            """This function generates plots for the DynamicMap"""
            ras_sub_df = self.gdf[self.gdf.datetime == datetime]
            units = self.mesh[CONCENTRATION].Units
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