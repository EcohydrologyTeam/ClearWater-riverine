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
import hashlib
import json
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
    compute_wet_mask,
)
import clearwater_riverine.variables
from clearwater_riverine.variables import (
    ADVECTION_COEFFICIENT,
    COEFFICIENT_TO_DIFFUSION_TERM,
    EDGES_FACE1,
    EDGES_FACE2,
    FACES,
    FACE_HYD_DEPTH,
    CHANGE_IN_TIME,
    NFACE,
    NEDGE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    VOLUME_ELEVATION_INFO,
    VOLUME_ELEVATION_VALUES,
    VOLUME_ELEVATION_LOOKUP,
    WET_MASK,
)
from clearwater_riverine.linalg import LHS
from clearwater_riverine.io.hdf import RASHDFDataSource
from clearwater_riverine.io.config import init_from_config
from clearwater_riverine.transport import TransportEngine
from clearwater_riverine.constituents import Constituent
from clearwater_riverine.postproc_util import (
    accumulate_chunk_mass_balance,
    calculate_global_mass_balance,
)

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

_CHECKPOINT_SCHEMA_VERSION = 1
"""Bump when the C3b checkpoint payload structure changes (keys, dtypes,
sidecar files, …). ``from_checkpoint`` rejects mismatched versions."""


def _sha256_of_file(path: str | Path) -> str:
    """Sha-256 of a file's contents, used as the C3b config-identity check."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


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
        wet_dry_metric: Optional[str] = None,
        wet_dry_threshold: Optional[Dict[str, float]] = None,
        mass_loss_warn_threshold: Optional[float] = 0.01,
        zero_dry_initial_conditions: bool = False,
        _existing_output_store: bool = False,
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
            self.crs = model.get("crs", None)
            for category, data_sources_dict in data_sources.items():
                self.__category_attr_map[category].update(data_sources_dict)
        else:
            self.__flow_field_file_path = flow_field_file_path
            self._start_datetime = start_datetime
            self._end_datetime = end_datetime
            self.__variable_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        
        self.__chunked_mode: bool = self.__chunk_size is not None
        # Cross-chunk mass-balance accumulator (C3a). None until the first
        # chunk is finalized; only used in chunked mode.
        self.__mb_acc = None
        # Checkpoint/resume state (C3b). __existing_output_store routes
        # init_template=False through to the chunked store so an existing
        # pre-allocated output store survives the resumed construction.
        # __resuming_ics: per-constituent end-of-prev-chunk concentration to
        # use as the IC for the resumed chunk, consumed in __load_new_chunk.
        # __last_finalized_boundary: the most recent chunk-end whose
        # __finalize_chunk completed (so the accumulator is consistent up
        # through this timestamp); checkpoint() saves state as of this time.
        self.__existing_output_store: bool = _existing_output_store
        self.__resuming_ics: Optional[dict] = None
        self.__last_finalized_boundary = self._start_datetime
        # __just_resumed: first __transport_chunked after from_checkpoint
        # must skip the chunk-end-detection that would re-finalize/re-load
        # the chunk we just loaded in from_checkpoint.
        self.__just_resumed: bool = False
        self.__config_filepath = config_filepath
        # Phase-D Unit A wet/dry scaffolding. wet_dry_metric=None (default)
        # is opt-out: no WET_MASK is registered and behavior is identical
        # to the pre-Unit-A path. Setting metric to "volume", "depth", or
        # "both" enables auto-registration of WET_MASK in __init_model
        # (and per-chunk re-registration in __load_new_chunk). The other
        # three kwargs are stored for the Phase-D units that consume them
        # (mass_loss_warn_threshold / zero_dry_initial_conditions: Unit
        # C's IC zeroing + end-of-run warning; wet_dry_threshold: this
        # unit's compute_wet_mask thresholds).
        self.__wet_dry_metric: Optional[str] = wet_dry_metric
        _wdt = wet_dry_threshold or {}
        self.__wet_dry_h_min: float = float(_wdt.get("h_min", 0.01))
        self.__wet_dry_v_min: float = float(_wdt.get("V_min", 0.1))
        self.__mass_loss_warn_threshold: Optional[float] = mass_loss_warn_threshold
        self.__zero_dry_initial_conditions: bool = bool(zero_dry_initial_conditions)
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
            self.__finalize_chunk(is_last=True)
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
            chunk_accumulator=self.__mb_acc if self.__chunked_mode else None,
        )


    def checkpoint(self, checkpoint_dir: str | Path) -> Path:
        """Write a chunk-boundary checkpoint for later resume (Phase-C C3b).

        Captures the model's state as of the most recent fully-finalized
        chunk boundary (``__last_finalized_boundary``): the cross-chunk
        mass-balance accumulator (C3a), the resume timestamp, and the
        per-constituent concentration at that boundary (which the resumed
        run uses as the next chunk's IC). The output store at
        ``simulation_directory/model_outputs.zarr`` is left untouched and
        survives via ``init_template=False`` on resume.

        Calling ``run()`` on a model returned by ``from_checkpoint`` re-runs
        every chunk after the boundary; because the chunked transport and
        write path are deterministic (proven by the C2 oracle), the
        resumed run's final result equals an uninterrupted run.
        """
        if not self.__chunked_mode:
            raise RuntimeError(
                "checkpoint() is only meaningful in chunked mode."
            )
        if self.__last_finalized_boundary == self._start_datetime:
            raise RuntimeError(
                "No chunk boundaries have been finalized yet; nothing "
                "to checkpoint. Run at least one full chunk first."
            )
        if self.__mb_acc is None:
            raise RuntimeError(
                "checkpoint() requires mass_flux_calculation=True so the "
                "cross-chunk mass-balance accumulator (C3a) carries the "
                "state a continuity-preserving resume needs."
            )
        if self.__config_filepath is None:
            raise RuntimeError(
                "checkpoint() currently requires the model to have been "
                "built from a config_filepath (resume re-reads it)."
            )

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        boundary = self.__last_finalized_boundary

        # The boundary slot is set as chunk K+1's IC by __load_new_chunk;
        # registry.get_at_time at that timestamp returns the same value
        # regardless of how far into chunk K+1 transport has advanced
        # (transport writes to subsequent slots, not back to slot 0).
        resume_concentrations = {
            name: np.asarray(self.registry.get_at_time(name, boundary))
            for name in self._constituents
        }

        metadata = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "current_time": pd.Timestamp(boundary).isoformat(),
            "mb_acc": self.__mb_acc,
            "config_filepath": str(self.__config_filepath),
            "config_hash": _sha256_of_file(self.__config_filepath),
        }
        with open(checkpoint_dir / "checkpoint.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        np.savez(checkpoint_dir / "resume_state.npz", **resume_concentrations)
        return checkpoint_dir


    @classmethod
    def from_checkpoint(
        cls,
        config_filepath: str | Path,
        checkpoint_dir: str | Path,
    ) -> "ClearwaterRiverine":
        """Reconstruct a model from a chunk-boundary checkpoint (C3b).

        Builds the model from ``config_filepath`` with the existing output
        store preserved (``_existing_output_store=True`` → clearwater_data
        ``init_template=False``), restores the cross-chunk accumulator and
        the resume timestamp, stages per-constituent resume ICs, and loads
        the resume chunk's hydrodynamic window. The returned model is
        ready for ``run()`` to continue to ``end_datetime``.

        The current ``config_filepath`` must match the checkpointing run's
        config (sha256 identity check).
        """
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "checkpoint.json") as f:
            metadata = json.load(f)
        if metadata.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema_version "
                f"{metadata.get('schema_version')!r} does not match the "
                f"current {_CHECKPOINT_SCHEMA_VERSION}; resume not supported."
            )
        config_filepath = Path(config_filepath)
        if _sha256_of_file(config_filepath) != metadata["config_hash"]:
            raise ValueError(
                f"Config identity mismatch: {config_filepath} sha256 does "
                f"not match the checkpoint's config_hash. Resume requires "
                f"the same config file as the checkpointed run."
            )

        npz = np.load(checkpoint_dir / "resume_state.npz")

        model = cls(
            config_filepath=str(config_filepath),
            _existing_output_store=True,
        )

        # Wrap each per-constituent npz ndarray in an xr.DataArray with the
        # SAME nface coord the constituent's IC array uses, so the
        # downstream reset_initial_conditions / set_initial_conditions
        # chain (which does .rename(...).reindex(nface=...).data) finds a
        # matching index instead of NaN-padding.
        resume_ics = {}
        for name in npz.files:
            template = model.registry.get_at_time(name, model._start_datetime)
            if isinstance(template, xr.DataArray):
                resume_ics[name] = xr.DataArray(
                    npz[name],
                    dims=template.dims,
                    coords=template.coords,
                )
            else:
                resume_ics[name] = xr.DataArray(
                    npz[name], dims=(NFACE,)
                )

        cp_time = pd.to_datetime(metadata["current_time"])
        model.__current_time = cp_time
        model.__last_finalized_boundary = cp_time
        model.__mb_acc = metadata["mb_acc"]
        model.__resuming_ics = resume_ics
        model.__just_resumed = True
        # Load the resume chunk's hydrodynamic window + apply resume IC.
        model.__load_new_chunk()
        return model


    def __increment_timestep(self):
        """Increment the model timestep.

        ``calculate_change_in_time`` (utilities.py) returns a ``FloatVariable``
        (scalar) when RAS stamps are uniform and a ``DataArrayVariable``
        (per-step array) otherwise. The C4 uniform-cadence precondition
        (__init_model) normally rules out the array case, but this scalar/
        array fallback is cheap robustness if it ever slips through.
        """
        dt_raw = self.registry.get(CHANGE_IN_TIME)
        try:
            dt_seconds = float(dt_raw)
        except (TypeError, ValueError):
            # Per-step array: look up dt at the current time.
            dt_seconds = float(
                self.registry.get_at_time(CHANGE_IN_TIME, self.__current_time)
            )
        self.__current_time += timedelta(seconds=dt_seconds)


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

        # Loud precondition (Phase-C C4 / B4). The chunked design and the
        # C3a mass-balance accumulator's per-chunk drop-last-slot tiling
        # both assume a uniform RAS output cadence. Non-uniform stamps
        # would make calculate_change_in_time return a per-step array
        # (the latent dt-array path) and silently drift the chunk grid
        # away from the timestep grid. The resolved B3/B4 design is
        # uniform-cadence-only; verify it explicitly and fail loudly.
        all_dts = self.__variable_data_sources[
            'hydrodynamic_model'
        ].all_datetimes
        diffs = np.diff(np.asarray(all_dts.values))
        if len(diffs) > 1 and not np.all(diffs == diffs[0]):
            raise ValueError(
                "Non-uniform RAS output cadence detected (timestamps "
                "differ in spacing). The chunked transport design "
                "(Phase-C B3/B4) assumes a uniform timestep. Re-export "
                "the RAS run at a single uniform interval."
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

        # Phase-D Unit A: register WET_MASK after VOLUME (+ optionally
        # FACE_HYD_DEPTH) is available; opt-in via wet_dry_metric.
        self.__populate_wet_mask()

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


    def __populate_wet_mask(self):
        """Register ``WET_MASK`` on the current chunk window (Phase-D Unit A).

        Opt-in: no-op when ``wet_dry_metric is None`` (the default). When
        set, computes a per-cell wet/dry boolean from the resident
        ``VOLUME`` (and ``FACE_HYD_DEPTH`` if the requested metric needs
        it). Re-registers each time it's called, so a chunked run gets
        a fresh ``WET_MASK`` with the chunk's time coord (called from
        ``__init_model`` for chunk 1 and from ``__load_new_chunk`` for
        each subsequent chunk window).
        """
        if self.__wet_dry_metric is None:
            return
        volume = self.registry.get(VOLUME)
        depth = None
        if (
            self.__wet_dry_metric in ("depth", "both")
            and FACE_HYD_DEPTH in self.registry
        ):
            depth = self.registry.get(FACE_HYD_DEPTH)
        mask = compute_wet_mask(
            volume,
            depth,
            h_min=self.__wet_dry_h_min,
            V_min=self.__wet_dry_v_min,
            metric=self.__wet_dry_metric,
        )
        if WET_MASK in self.registry:
            self.registry.unregister(WET_MASK)
        self.registry.register(
            WET_MASK,
            DataArrayVariable(mask, space_dimension=NFACE),
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
        # init_template=False on resume (C3b) preserves the existing
        # pre-allocated store so subsequent write_chunk(region="auto")
        # appends rather than clobbers.
        init_template = not self.__existing_output_store
        if self.__chunked_mode:
            self.__output_data_store = ChunkedZarrDataStore(
                store_path=self.__simulation_directory / "model_outputs.zarr",
                start_date=self._start_datetime,
                end_date=self._end_datetime,
                time_step=timedelta(seconds=self.registry.get(CHANGE_IN_TIME)),
                variables=self.__output_variables,
                chunk_size=self.__chunk_size,
                spatial_field=NFACE,
                spatial_field_values=self.registry.get(VOLUME).nface,
                init_template=init_template,
            )
        else:
            self.__output_data_store = ZarrDataStore(
                store_path=self.__simulation_directory / "model_outputs.zarr",
                start_date=self._start_datetime,
                end_date=self._end_datetime,
                time_step=timedelta(seconds=self.registry.get(CHANGE_IN_TIME)),
                variables=self.__output_variables,
                spatial_field=NFACE,
                spatial_field_values=self.registry.get(VOLUME).nface,
                init_template=init_template,
            )


    def __init_chunks(self):
        """Define the interior chunk boundaries.

        ``pd.date_range(start, end, freq=chunk_size)`` produces stamps at
        ``start, start+chunk_size, ..., k*chunk_size <= end``. The previous
        ``[1:-1]`` slice dropped both endpoints, which is correct WHEN
        ``(end-start)`` is an exact multiple of ``chunk_size`` (the last
        element is ``end`` itself and we want only interior boundaries).
        When it is NOT a multiple, the last element is ``< end`` and the
        slice drops a *legitimate interior* boundary, so the final chunk
        spans up to ``~2 * chunk_size`` (Phase-C C4 / B4 fix).

        Filter explicitly to boundaries strictly between start and end:
        identical to ``[1:-1]`` on the even-split path the C2/C3a/C3b
        oracles use, plus correct on the uneven-split path the new C4
        oracle exercises.
        """
        all_stamps = pd.date_range(
            self._start_datetime,
            self._end_datetime,
            freq=self.__chunk_size,
        )
        self.__chunk_ends = all_stamps[
            (all_stamps > self._start_datetime)
            & (all_stamps < self._end_datetime)
        ]


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
        # Phase-D Unit A: refresh WET_MASK for the new chunk's time window.
        self.__populate_wet_mask()
        for constituent_name, constituent in self._constituents.items():
            # C3b: on the first __load_new_chunk after from_checkpoint, the
            # in-memory registry holds chunk 1's array, not the end-of-prev-
            # chunk state. Use the per-constituent concentration loaded
            # from the checkpoint sidecar instead of registry.get_at_time
            # (which would return NaN / wrong values). One-shot: cleared
            # after consumption so subsequent chunk transitions use the
            # normal path.
            if (self.__resuming_ics is not None
                    and constituent_name in self.__resuming_ics):
                ic_value = self.__resuming_ics[constituent_name]
            else:
                ic_value = self.registry.get_at_time(
                    constituent_name, self.__current_time
                )
            constituent.reset_initial_conditions(self.registry, ic_value)
            constituent.register_constituent(self.registry)
            constituent.set_initial_conditions(self.registry, self.__current_time)
            constituent.set_boundary_conditions(self.registry)
        self.__resuming_ics = None

    
    def __transport_chunked(self):
        if self.__just_resumed:
            # First call after from_checkpoint: __current_time equals a
            # chunk boundary (the resume point), but the resume chunk has
            # already been loaded in from_checkpoint and the accumulator
            # already contains chunks 1..K. Skipping the finalize+load on
            # this one call avoids re-finalizing chunk K (which would
            # double-count it in the accumulator) and re-loading chunk K+1
            # (which would clobber the staged resume IC). One-shot.
            self.__just_resumed = False
        elif len(self.__chunk_ends) > 0:
            # Tolerance-based chunk-boundary detection (Phase-C C4 / B4).
            # Identical to the previous exact-equality semantics when
            # __current_time lands on a boundary exactly (clearwater_data
            # A3 enforces chunk_size % time_step == 0, so this is the
            # common case). The ``>= next_unfired`` form is cheap
            # robustness for any misconfigured / array-dt edge that would
            # otherwise miss the boundary forever.
            unfired = self.__chunk_ends[
                self.__chunk_ends > self.__last_finalized_boundary
            ]
            if len(unfired) > 0 and self.__current_time >= unfired[0]:
                self.__finalize_chunk()
                self.__load_new_chunk()
        self.__transport()


    def __finalize_chunk(self, is_last: bool = False):
        if self.__mass_flux_calculation:
            self.__calculate_mass_flux()
            # Cross-chunk mass-balance continuity (C3a). Fold this chunk's
            # boundary contribution + (first) start / (last) end domain
            # snapshots into the accumulator. Interior chunks drop the
            # shared trailing slot so each timestep is counted once across
            # the run; the final chunk keeps it.
            for constituent_name in self._constituents:
                self.__mb_acc = accumulate_chunk_mass_balance(
                    self.__mb_acc,
                    self.registry,
                    constituent_name,
                    self._start_datetime,
                    self._end_datetime,
                    drop_last_slot=not is_last,
                    is_last=is_last,
                )

        # C3b: record the chunk boundary up through which the accumulator
        # is consistent. Interior finalize: the accumulator has chunks
        # 1..K and resume should start from this timestamp. Final finalize
        # (is_last=True) closes the run; not a resume point.
        if not is_last:
            self.__last_finalized_boundary = self.__current_time

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