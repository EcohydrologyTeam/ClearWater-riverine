import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix, linalg
import matplotlib.pyplot as plt
import holoviews as hv
# import geoviews as gv
import geopandas as gpd
from shapely.geometry import Polygon
# hv.extension("bokeh")
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
)
from pathlib import Path
import warnings
import inspect
from datetime import datetime, timedelta

from clearwater_data.io.base import DataSource, ChunkedDataSource
from clearwater_data.io.zarr import ZarrDataStore, ChunkedZarrDataStore
from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.float import FloatVariable
from clearwater_data.variables.xarray import DataArrayVariable

from clearwater_riverine.utilities import(
    CALCULATED_VARIABLE_MAP,
)
import clearwater_riverine.variables
from clearwater_riverine.variables import (
    ADVECTION_COEFFICIENT,
    COEFFICIENT_TO_DIFFUSION_TERM,
    EDGES_FACE1,
    EDGES_FACE2,
    FACES,
    CHANGE_IN_TIME,
    NFACE,
    NEDGE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    VOLUME_ELEVATION_INFO,
    VOLUME_ELEVATION_VALUES,
    VOLUME_ELEVATION_LOOKUP,
)
from clearwater_riverine.linalg import LHS
from clearwater_riverine.io.hdf import RASHDFDataSource
from clearwater_riverine.io.config import init_from_config
from clearwater_riverine.transport import TransportEngine
from clearwater_riverine.constituents import Constituent
from clearwater_riverine.postproc_util import calculate_global_mass_balance

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
        config_filepath: path to configuration file for Clearwater-Riverine.
    """
    ## TODO: Discuss- will anyone init the model except with a config?
    # Can we delete all other inputs to CW-R?
    def __init__(
        self,
        config_filepath: Optional[str | Path] = None,
        flow_field_file_path: Optional[str | Path] = None,
        # diffusion_coefficient_input: Optional[float] = None,
        constituent_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        variable_registry: Optional[VariableRegistry] = None,
        chunk_size: Optional[timedelta] = None,
        # datetime_range: Optional[Tuple[int, int] | Tuple[datetime, datetime]] = None,
        # mesh_file_path: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize a Clearwater Riverine water quality model from hydrodynamic model (e.g., HEC RAS) output.
        """
        self.registry = variable_registry if variable_registry is not None else VariableRegistry()
        self.__variable_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        self.__initial_condition_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        self.__boundary_condition_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        self.__category_attr_map = {
            "boundary_conditions": self.__boundary_condition_data_sources,
            "initial_conditions": self.__initial_condition_data_sources,
            "variable_data_sources": self.__variable_data_sources
        }
        self._constituents: dict[str: Constituent] = {}

        if config_filepath:
            model, data_sources, constituents = init_from_config(config_filepath)
            self.__simulation_directory = Path(model.get("simulation_directory", "./"))
            self.__flow_field_file_path = model["hydrodynamic_input"]
            raw_chunk = model.get("chunk_size", None)
            self.__chunk_size = pd.Timedelta(raw_chunk) if raw_chunk is not None else None
            self._start_datetime = pd.to_datetime(model.get("start_datetime", None))
            self._end_datetime = pd.to_datetime(model.get("end_datetime", None))
            self.__calculated_variables = model.get("calculated_variables", None)
            self.__output_variables = model.get("output_variables", constituents)
            self.__mass_flux_calculation = model.get("mass_flux_calculation", False)
            # Loud precondition (Phase-C C2, 6th canonical defect). In chunked
            # mode __finalize_chunk runs every chunk and re-registers
            # f"{constituent}_mass_flux" with overwrite=False, so chunk 2
            # crashes with a cryptic "already registered" ValueError. Even
            # past that, _calculate_mass_flux computes only the current
            # chunk's window with no cross-chunk accumulation, so a chunked
            # global mass balance would be incorrect. Cross-chunk flux/mass
            # continuity is a fork PORT item owned by Phase-C C3. Fail loudly
            # and early until C3 lands it.
            if self.__chunk_size is not None and self.__mass_flux_calculation:
                raise NotImplementedError(
                    "Chunked mode with mass_flux_calculation=True is not yet "
                    "supported: per-chunk mass flux re-registers without "
                    "cross-chunk accumulation, so the chunked global mass "
                    "balance would be incorrect. Cross-chunk flux/mass "
                    "continuity is deferred to Phase-C C3. Run with "
                    "chunk_size unset, or with mass_flux_calculation=False."
                )
            self.crs = model.get("crs", None)
            for category, data_sources_dict in data_sources.items():
                self.__category_attr_map[category].update(data_sources_dict)
        else:
            self.__flow_field_file_path = flow_field_file_path
            self._start_datetime = start_datetime
            self._end_datetime = end_datetime
            self.__variable_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        
        self.__chunked_mode: bool = self.__chunk_size is not None
        self.plotter = None

        self.__init_model(constituents)
        self.__init_output_store()
        self.__init_chunks()
        self.__transport_engine = TransportEngine(self.registry)

    
    def run(self) -> None:
        while self.__current_time < self._end_datetime:
            self.update()
        self.finalize()

    def update(self) -> None:
        # transport
        if self.__chunked_mode:
            self.__transport_chunked()
        else:
            self.__transport()

        # update timestep
        self.__increment_timestep()


    def finalize(self) -> None:
        if self.__chunked_mode:
            self.__finalize_chunk()
        else:
            if self.__mass_flux_calculation:
                self.__calculate_mass_flux()
            
            for variable_name in self.__output_variables:
                variable = self.registry.get(variable_name)
                self.__output_data_store.write(
                    data=variable,
                    parameter_name=variable_name,
                )
    
    def plot(self, constituent_name: str, **kwargs):
        if self.plotter is None:
            from clearwater_riverine.plotting import RiverinePlotter
            self.plotter = RiverinePlotter(registry=self.registry, crs=self.crs)
        return self.plotter.dynamic_plot(constituent_name=constituent_name, **kwargs)
    
    def static_plot(self, constituent_name: str, plotting_datetime: str | datetime, **kwargs):
        self.__set_up_plotter()
        return self.plotter.static_plot(constituent_name, plotting_datetime, **kwargs)
    
    def quick_plot(self, constituent_name: str, **kwargs):
        self.__set_up_plotter()
        return self.plotter.quick_plot(constituent_name=constituent_name, **kwargs)
 
    def calculate_mass_balance(
        self,
        constituent_name: str,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        calculate_answer: Optional[bool] = False,
        answer_value: Optional[float] = 100,
    ):
        if start_datetime is None:
            start_datetime = self._start_datetime
        if end_datetime is None:
            end_datetime = self._end_datetime
        return calculate_global_mass_balance(
            self.registry,
            constituent_name,
            start_datetime,
            end_datetime,
            calculate_answer,
            answer_value,
        )


    def __increment_timestep(self):
        """Increment the model timestep."""
        self.__current_time += timedelta(seconds=self.registry.get(CHANGE_IN_TIME)) 


    def __init_model(self, constituents: dict):
        """Initialize the Clearwater-Riverine Model"""      
        # Register configured information
        # For now this should just be the diffusion coefficient
        for variable_name, data_source in self.__variable_data_sources.items():
            if isinstance(data_source, ChunkedDataSource):
                data = data_source.read_chunk(
                    variable_name,
                    self._start_datetime, self._start_datetime + self.__chunk_size
                )
            else:
                data = data_source.read(variable_name)
            
            self.registry.register(
                variable_name,
                data,
            )

        # Register hydrodynamic data
        self.__variable_data_sources['hydrodynamic_model'] = RASHDFDataSource(
            ras_hdf_path=self.__flow_field_file_path,
            start_datetime=self._start_datetime,
            end_datetime=self._end_datetime,
            calculated_variables=self.__calculated_variables,
        )

        # Loud precondition (Finding #3, Phase-C C1a). io/hdf.py
        # __read_temporal_variables reads GATE_FLOW into the RAS mesh, but
        # this branch does not yet register GATE_FLOW into the
        # VariableRegistry. The linalg gate path keys off
        # `GATE_FLOW in registry` (TransportEngine), so an unregistered
        # GATE_FLOW would silently evaluate False and drop *all* gate flow
        # from the transport solution. Fail loudly until the registration
        # wire-up is implemented and validated against a gated fixture
        # (Phase-C scale validation).
        if self.__variable_data_sources['hydrodynamic_model'].gate_names is not None:
            raise NotImplementedError(
                "This HEC-RAS model contains gate structures, but gate flow "
                "is not yet wired into the variable registry on this branch. "
                "Running it would silently drop all gate flow from the "
                "transport solution. Gated models are unsupported here until "
                "GATE_FLOW registration is implemented and validated against "
                "a gated fixture."
            )

        for variable_name in self.__variable_data_sources['hydrodynamic_model'].temporal_variables:
            if self.__chunked_mode:
                data = self.__variable_data_sources['hydrodynamic_model'].read_chunk(
                    variable_name,
                    start_time = self._start_datetime,
                    end_time=self._start_datetime + self.__chunk_size
                )
            else:
                data = self.__variable_data_sources['hydrodynamic_model'].read(variable_name)
    
            self.registry.register(
                variable_name,
                data,
            )
        
        non_temporal_variables = list(self.__variable_data_sources['hydrodynamic_model'].static_variables.keys()) \
            + list(self.__variable_data_sources['hydrodynamic_model'].topology_variables)
        for variable_name in non_temporal_variables:
            self.registry.register(
                variable_name,
                DataArrayVariable(
                    self.__variable_data_sources['hydrodynamic_model'].mesh[variable_name],
                    space_dimension = self.__variable_data_sources['hydrodynamic_model'].static_variables.get(variable_name)
                )
            )
        
        for variable_name in self.__variable_data_sources['hydrodynamic_model'].lookup_variables:
            self.registry.register(
                variable_name,
                DataArrayVariable(self.__variable_data_sources['hydrodynamic_model'].volume_elevation_lookup[variable_name])
            )
        
        for variable_name in self.__variable_data_sources['hydrodynamic_model'].boundary_variables:
            self.registry.register(
                variable_name,
                DataArrayVariable(self.__variable_data_sources['hydrodynamic_model'].boundary_data[variable_name])
            )


        # register additional variables
        self.registry.register(
            NUMBER_OF_REAL_CELLS,
            FloatVariable(self.__variable_data_sources['hydrodynamic_model'].real_cell_count)
        )
        self.registry.register(
            NFACE,
            FloatVariable(self.__variable_data_sources['hydrodynamic_model'].nface)
        )
        self.registry.register(
            NEDGE,
            FloatVariable(self.__variable_data_sources['hydrodynamic_model'].nedge)
        )
        
        # calculate intermediate variables
        self.__update_calculated_variables()

        # initialize constituents
        for constituent_name in list(constituents.keys()):
            self.__init_constituents(
                constituent_name=constituent_name, 
                constituent_config=constituents[constituent_name]
            )
        
        # set current timestep
        self.__current_time = self._start_datetime


    def __init_constituents(
            self,
            constituent_name: str,
            constituent_config: dict,
    ):
        """Initalize model constituents."""
        initial_conditions = self.__initial_condition_data_sources[constituent_name].read(constituent_name)
        boundary_conditions = self.__boundary_condition_data_sources[constituent_name].read(constituent_name)
        if isinstance(boundary_conditions, DataArrayVariable):
            boundary_conditions = DataArrayVariable(boundary_conditions.get().interpolate_na(dim="time", method="linear"))

        self._constituents[constituent_name] = Constituent(
            constituent_name=constituent_name,
            registry=self.registry,
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            constituent_config=constituent_config,
            start_datetime=self._start_datetime
        )


    def __update_calculated_variables(self):
        """Initialize calculated variables."""
        # unregister
        for calculated_variable, calculate in self.__calculated_variables.items():
            if calculate and calculated_variable in self.registry:
                self.registry.unregister(calculated_variable)
        
        # recalculate and register
        for calculated_variable, calculate in self.__calculated_variables.items():
            if calculate:
                # if calculated_variable not in self.registry:
                # TODO: add logic to set calculated variables in order of dependencies
                if calculated_variable not in self.registry:
                    calculation_method = CALCULATED_VARIABLE_MAP[calculated_variable]
                    calculated_result = calculation_method(
                        registry=self.registry,
                    )
                    self.registry.register(
                        calculated_variable,
                        calculated_result,
                    )


    def __init_output_store(self):
        if self.__chunked_mode:
            self.__output_data_store = ChunkedZarrDataStore(
                store_path=self.__simulation_directory / "model_outputs.zarr",
                start_date=self._start_datetime,
                end_date=self._end_datetime,
                time_step=timedelta(seconds=self.registry.get(CHANGE_IN_TIME)),
                variables=self.__output_variables,
                chunk_size=self.__chunk_size,
                spatial_field=NFACE,
                spatial_field_values=self.registry.get(VOLUME).nface               
            )
        else:
            self.__output_data_store = ZarrDataStore(
                store_path=self.__simulation_directory / "model_outputs.zarr",
                start_date=self._start_datetime,
                end_date=self._end_datetime,
                time_step=timedelta(seconds=self.registry.get(CHANGE_IN_TIME)),
                variables=self.__output_variables,
                spatial_field=NFACE,
                spatial_field_values=self.registry.get(VOLUME).nface
            )


    def __init_chunks(self):
        """Define the end of each chunk."""
        self.__chunk_ends = pd.date_range(
            self._start_datetime,
            self._end_datetime,
            freq=self.__chunk_size
        )[1:-1]


    def __load_new_chunk(self):
        """Load new chunk."""
        for variable_name in self.__variable_data_sources['hydrodynamic_model'].temporal_variables:
            self.registry.unregister(variable_name)
            data = self.__variable_data_sources['hydrodynamic_model'].read_chunk(
                variable_name,
                start_time = self.__current_time,
                end_time=self.__current_time + self.__chunk_size
            )
            self.registry.register(
                variable_name,
                data,
            )

        self.__update_calculated_variables()
        for constituent_name, constituent in self._constituents.items():
            constituent.reset_initial_conditions(
                self.registry,
                self.registry.get_at_time(constituent_name, self.__current_time)
            )
            constituent.register_constituent(self.registry)
            constituent.set_initial_conditions(self.registry, self.__current_time)
            constituent.set_boundary_conditions(self.registry)        

    
    def __transport_chunked(self):
        if self.__current_time in self.__chunk_ends:
            self.__finalize_chunk()
            self.__load_new_chunk()
        self.__transport()


    def __finalize_chunk(self):
        if self.__mass_flux_calculation:
            self.__calculate_mass_flux()
            # TODO: write mass flux to output?
            # will neeed to handle multiple space dimensions

        for variable_name in self.__output_variables:
            # calculate mass flux, if necessary                
            # TODO: clean up chunk indexing
            variable = self.registry.get(variable_name).isel(time=slice(0, -1))
            self.__output_data_store.write_chunk(
                data=variable,
                parameter_name=variable_name,
                start_time=variable.time[0].values,
                end_time=variable.time[-1].values,
            )

    def __transport(self):
        """Call transport process"""
        self.__transport_engine.run(
            registry=self.registry,
            current_time=self.__current_time,
            time_step=timedelta(seconds=self.registry.get(CHANGE_IN_TIME)),
            constituents=self._constituents,
            mass_flux_calculation=self.__mass_flux_calculation
         )

    def __calculate_mass_flux(self):
        if self.__mass_flux_calculation:
            # TODO: toggle on and off for different constituents?
            for _, constituent in self._constituents.items():
                constituent._calculate_mass_flux(self.registry)

    def __set_up_plotter(self):
        if self.plotter is None:
            from clearwater_riverine.plotting import RiverinePlotter
            self.plotter = RiverinePlotter(registry=self.registry, crs=self.crs)