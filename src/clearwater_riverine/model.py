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
from clearwater_riverine.transport import (
    TransportEngine,
    emit_mass_loss_warning,
    # Aliased to avoid name collision with the ``__init__`` kwarg of
    # the same spelling. Inside the class scope the bare name resolves
    # to the bool kwarg, shadowing the function and producing a
    # ``TypeError: 'bool' object is not callable`` at call time. The
    # ``_zero_dry_initial_conditions_fn`` alias is what model code
    # invokes; tests that want to spy on the call patch the alias on
    # this module.
    zero_dry_initial_conditions as _zero_dry_initial_conditions_fn,
)
from clearwater_riverine.fork_compat import (
    MeshView,
    apply_update_concentration,
)
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
        reconstruct_newly_wet: bool = True,
        continuity_correction: str = "bc_only",
        allow_cell_volume_fallback: bool = False,
        allow_face_flow_fallback: bool = False,
        _existing_output_store: bool = False,
    ) -> None:
        """
        Initialize a Clearwater Riverine water quality model from hydrodynamic model (e.g., HEC RAS) output.
        """
        self.registry = variable_registry if variable_registry is not None else VariableRegistry()
        self.__variable_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        self.__initial_condition_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        self.__boundary_condition_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
        # Phase F T2-A (2026-05-21): optional per-constituent point-
        # source CSV file paths. Keyed by constituent name; empty when
        # no constituent in the config declares a ``point_sources``
        # block. Stored as a path (str) rather than a DataSource
        # because the CSV schema (Cell_Index, Datetime, Flow_Rate,
        # Concentration) is fixed and does not need a provider
        # abstraction.
        self.__point_source_data_sources: dict[str, str] = {}
        self.__category_attr_map = {
            "boundary_conditions": self.__boundary_condition_data_sources,
            "initial_conditions": self.__initial_condition_data_sources,
            "point_sources": self.__point_source_data_sources,
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
            # Phase I-1 (2026-05-21): persist diffusion-dispatch
            # method + params for registration after the registry is
            # populated. ``_diffusion_method`` is one of
            # {"constant", "elder", "eddy_viscosity", "array"};
            # ``_diffusion_params`` is the method-specific kwargs
            # dict (alpha for elder, schmidt for eddy_viscosity,
            # file_path for array). For the legacy scalar config
            # form, _diffusion_method == "constant" and
            # _diffusion_params == {}.
            self.__diffusion_method = model.get("_diffusion_method", "constant")
            self.__diffusion_params = model.get("_diffusion_params", {})
            for category, data_sources_dict in data_sources.items():
                self.__category_attr_map[category].update(data_sources_dict)
        else:
            self.__flow_field_file_path = flow_field_file_path
            self._start_datetime = start_datetime
            self._end_datetime = end_datetime
            self.__variable_data_sources: dict[str, DataSource | ChunkedDataSource] = {}
            # Phase I-1 default for the no-config path: constant diffusion.
            self.__diffusion_method = "constant"
            self.__diffusion_params = {}
        
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
        # Phase F (2026-05-21): per-cell volume-continuity correction
        # mode. Default ``"bc_only"`` matches the streaming repo's
        # historical default (Option B-lite: distribute per-cell
        # residual across boundary edges only). ``"all_edges"`` invokes
        # the full Option B (graph-Laplacian solve across all incident
        # edges) and is required for RAS HDFs whose upstream BC was an
        # Internal type in the original solution (e.g. Santiam-Salem,
        # where Upstream and Santiam are Internal lines). ``"none"``
        # skips the correction entirely (ADVECTION_COEFFICIENT equals
        # the raw FLOW_ACROSS_FACE).
        if continuity_correction not in ("bc_only", "all_edges", "none"):
            raise ValueError(
                f"Unknown continuity_correction mode {continuity_correction!r}. "
                "Expected 'bc_only', 'all_edges', or 'none'."
            )
        self.__continuity_correction: str = continuity_correction
        # Phase F (2026-05-21): opt-out for newly-wet reconstruction.
        # Defaults to True (preserves the Phase-D Unit-B correctness
        # behaviour for new models). Set False to match the streaming
        # repo's reference-run configuration, which disables the pass
        # because it is O(newly_wet_cells x edges) per step and becomes
        # pathologically slow on RAS HDFs that start dry and wet up
        # across thousands of cells per hour (e.g. the Santiam-Salem
        # Sep 2008 deck, whose plan was configured with a fully-dry
        # initial condition and propagates inflow across ~30k cells
        # between simulation hours 10 and 20). When False, newly-wet
        # cells inherit whatever the solver writes -- typically 0.0 for
        # cells with no qualifying upstream neighbour -- matching the
        # accepted tradeoff that produced the streaming locked baseline
        # (Salem T bias -0.30 deg C, RMSE 0.62 deg C).
        self.__reconstruct_newly_wet: bool = bool(reconstruct_newly_wet)
        # Phase J+1 (Corvallis 2026-05-23): per-variable opt-ins for the
        # geometry-derived fallbacks. Default ``False`` is fail-loud:
        # ``RASHDFDataSource.__probe_temporal_fallbacks`` raises with a
        # remediation-rich error if the corresponding RAS optional
        # output is absent and the flag is False. Override per-variable
        # via these kwargs or via ``model.allow_*_fallback`` in the YAML
        # config (``init_from_config`` passes both through verbatim).
        # See design/missing_temporal_fallback.md for fidelity validation.
        if config_filepath:
            allow_cell_volume_fallback = bool(
                model.get("allow_cell_volume_fallback", allow_cell_volume_fallback)
            )
            allow_face_flow_fallback = bool(
                model.get("allow_face_flow_fallback", allow_face_flow_fallback)
            )
        self.__allow_cell_volume_fallback: bool = bool(allow_cell_volume_fallback)
        self.__allow_face_flow_fallback: bool = bool(allow_face_flow_fallback)
        self.plotter = None

        self.__init_model(constituents)
        self.__init_output_store()
        self.__init_chunks()
        self.__transport_engine = TransportEngine(
            self.registry,
            reconstruct_newly_wet=self.__reconstruct_newly_wet,
        )

        # Phase-D Unit D2: model-level IC-zeroing opt-in.
        # When ``zero_dry_initial_conditions=True`` AND ``WET_MASK`` is
        # in the registry (Unit-A opt-in), sweep any IC mass loaded
        # into sub-threshold cells at ``start_datetime``: the
        # concentration is zeroed for extensive constituents and the
        # discarded mass is logged to the engine's
        # ``mass_lost_to_dry`` accumulator so the end-of-run warning
        # can surface it. Intensive scalars (e.g. temperature) are
        # skipped by ``zero_dry_initial_conditions`` itself --
        # T = 0 in a sub-threshold cell is non-physical and would
        # cascade through coupled physics when the cell becomes wet.
        # Default ``False`` preserves the legacy IC behaviour for
        # real-mesh runs where the user's IC file is the source of
        # truth on sub-threshold cells.
        if self.__zero_dry_initial_conditions and WET_MASK in self.registry:
            ic_lost = _zero_dry_initial_conditions_fn(
                self.registry, self._constituents, self._start_datetime,
            )
            for name, mass in ic_lost.items():
                self.__transport_engine.mass_lost_to_dry.setdefault(
                    name, []
                ).append(float(mass))


    @property
    def transport_engine(self) -> TransportEngine:
        """Read-only accessor for the underlying transport engine.

        Exposes the ``TransportEngine`` instance so tests and callers
        can inspect post-run diagnostics (notably
        ``mass_lost_to_dry``) without reaching through the model's
        private name-mangled attribute.
        """
        return self.__transport_engine

    @property
    def mesh(self) -> MeshView:
        """Fork-compat view of the registry shaped like ``model.mesh``.

        Returned object is a ``MeshView`` (see ``fork_compat.py``) that
        proxies the subset of the xarray Dataset API the fork-side
        orchestrators use: keyed reads, membership tests, sizes for
        time / nface, the time and nface coord arrays, and ``nreal``.
        Writes via ``mesh[name].loc[...] = arr`` mutate the registry
        in place because the view returns the registry's own
        DataArrays, not copies.

        The mesh view is lazily constructed on first access and cached
        on the instance; subsequent accesses return the same view.
        """
        view = getattr(self, "_mesh_view", None)
        if view is None:
            view = MeshView(self.registry)
            self._mesh_view = view
        return view

    @property
    def current_time(self):
        """Read-only accessor for the model's current simulation time.

        Public alias for the name-mangled ``__current_time`` attribute
        so fork-compat callers (and tests) do not have to reach
        through the mangled name.
        """
        return self.__current_time

    def run(self) -> None:
        while self.__current_time < self._end_datetime:
            self.update()
        self.finalize()

    def update(self, update_concentration: Optional[dict] = None) -> None:
        """Advance transport one step.

        Fork-compat optional kwarg ``update_concentration`` accepts a
        ``dict[str, np.ndarray | xr.DataArray]`` of per-constituent
        overrides to apply at the current simulation time before the
        transport solver reads its initial condition for the next
        step. This mirrors the streaming fork's ``update(...)`` shape
        and lets the Phase-2 ESM streaming orchestrator
        (``08_run_coupled_v3_smoke.py``) inject TSM- and NSM1-evolved
        values back into transport.

        Default ``None`` preserves the prior no-arg behaviour and the
        existing test surface bit-identically. Overrides are applied
        to the first ``nreal + 1`` slots (real cells plus the
        boundary ghost slot) at the current-time index, matching the
        fork's slice semantics.
        """
        if update_concentration:
            # ``nreal + 1`` = real cells plus the boundary ghost slot the
            # fork orchestrator addresses with ``[0:nreal+1]``. The
            # canonical Phase-D code uses the same convention.
            nreal_plus_ghost = (
                int(self.registry.get_variable(NUMBER_OF_REAL_CELLS).get()) + 1
            )
            apply_update_concentration(
                self.registry,
                self.__current_time,
                nreal_plus_ghost,
                update_concentration,
            )

        # transport
        if self.__chunked_mode:
            self.__transport_chunked()
        else:
            self.__transport()

        # update timestep
        self.__increment_timestep()


    def finalize(
        self,
        save: bool = True,
        output_filepath: Optional[str] = None,
    ) -> None:
        """Finalize the simulation.

        Args:
            save: when ``False``, skip the default write to the
                configured output store.
            output_filepath: accepted for streaming-repo signature
                parity; canonical writes to the configured
                ``simulation_directory / model_outputs.zarr`` (set at
                construction time) and logs a warning rather than
                silently redirecting. Phase F (2026-05-21) T1-H found
                that the natural ``model_outputs.zarr`` plus the
                validation script's fallback to ``nsm1_history.nc``
                (orchestrator output) is sufficient for the streaming-
                vs-canonical comparison without dual-writing.

        Default ``save=True, output_filepath=None`` preserves the
        prior no-arg behaviour.
        """
        if output_filepath is not None:
            warnings.warn(
                "ClearwaterRiverine.finalize received output_filepath="
                f"{output_filepath!r}; canonical writes to the output "
                "store configured at __init__ "
                "(simulation_directory/model_outputs.zarr). Set the "
                "simulation_directory in the model config to control "
                "the destination.",
                UserWarning,
                stacklevel=2,
            )

        if self.__chunked_mode:
            self.__finalize_chunk(is_last=True)
        else:
            if self.__mass_flux_calculation:
                self.__calculate_mass_flux()

            if save:
                for variable_name in self.__output_variables:
                    variable = self.registry.get_variable(variable_name).get_data()
                    self.__output_data_store.write(
                        data=variable,
                        parameter_name=variable_name,
                    )

        # Phase-D Unit D2: end-of-run wet-dry mass-loss warning.
        # Compares per-constituent total ``mass_lost_to_dry`` against
        # ``mass_loss_warn_threshold * sum(bc_inflow_mass)`` and emits
        # a ``UserWarning`` for each extensive constituent that
        # breaches. No-op when ``mass_loss_warn_threshold`` is ``None``
        # or the engine's accumulator is empty. Intensive scalars are
        # skipped by ``emit_mass_loss_warning`` itself (the BC inflow
        # MASS denominator has the wrong units for a scalar like
        # temperature).
        emit_mass_loss_warning(
            self.__transport_engine.mass_lost_to_dry,
            self._constituents,
            self.__mass_loss_warn_threshold,
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


    def _current_dt_seconds(self) -> float:
        """Return the dt for the current step in seconds.

        Phase F (2026-05-21): when CHANGE_IN_TIME is a scalar (uniform
        cadence), float() it. When it is a per-step array (the relaxed
        cadence guard's non-uniform-stamp path), look up the actual dt
        at the current time. This ensures ``current_time + time_step``
        lands exactly on the next RAS stamp, which is the only thing
        that lets the downstream ``get_at_time(VOLUME, ...)`` ``.sel``
        succeed without a tolerance.
        """
        dt_raw = self.registry.get_variable(CHANGE_IN_TIME).get_data()
        try:
            return float(dt_raw)
        except (TypeError, ValueError):
            return float(
                self.registry.get_at_time(CHANGE_IN_TIME, self.__current_time)
            )

    def _representative_dt_seconds(self) -> float:
        """Return a scalar dt in seconds for code paths that need it.

        ``calculate_change_in_time`` (utilities.py) returns a
        ``FloatVariable`` (scalar) when RAS stamps are uniform and a
        ``DataArrayVariable`` (per-step array) otherwise. The C4
        uniform-cadence precondition normally rules out the array
        case, but the Phase-F relaxation (2026-05-21) allows up to 10%
        drift. When that drift triggers the array case, ``__init_output_store``
        and other call sites need a single representative dt. Use the
        ``nanmedian`` of the per-step array as the representative; it
        is the natural fit when the array is "near-uniform with drift"
        and the ``nan`` skip handles the final-slot NaN padding
        ``calculate_change_in_time`` inserts via
        ``np.insert(dt, len(dt), np.nan)``.
        """
        dt_raw = self.registry.get_variable(CHANGE_IN_TIME).get_data()
        try:
            val = float(dt_raw)
            if np.isnan(val):
                # Scalar but NaN is unusable; fall through to array path.
                raise ValueError("scalar dt is NaN")
            return val
        except (TypeError, ValueError):
            # Per-step array. Use the nanmedian across the run as the
            # representative scalar (skips the trailing NaN that
            # ``calculate_change_in_time`` appends so the array length
            # matches the time axis).
            return float(np.nanmedian(np.asarray(dt_raw)))

    def __increment_timestep(self):
        """Increment the model timestep.

        Uses ``_representative_dt_seconds`` for the scalar/array
        fallback; when the array path is active, looks up dt at the
        current time for per-step accuracy.
        """
        dt_raw = self.registry.get_variable(CHANGE_IN_TIME).get_data()
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
            allow_cell_volume_fallback=self.__allow_cell_volume_fallback,
            allow_face_flow_fallback=self.__allow_face_flow_fallback,
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
        if len(diffs) > 1:
            # Tolerance-based uniform-cadence check (Phase-F 2026-05-21
            # relaxation). Real-world HEC-RAS unsteady outputs occasionally
            # slip the wall-clock stamp by +/- 1 minute on an otherwise
            # hourly grid (observed on the USGS Santiam-Salem 2008 plan:
            # 59 / 60 / 61 minute diffs across 361 stamps). The original
            # exact-equality check (np.all(diffs == diffs[0])) over-rejected
            # those plans even though the C4 tolerance-based chunk-boundary
            # detection (>= next-unfired-boundary) is designed to absorb
            # exactly this kind of drift. Allow up to 10 percent deviation
            # from the median spacing; anything larger is a genuine
            # cadence change (e.g. hourly mixed with daily, or sparse months)
            # and still fails loudly.
            median_dt = float(np.median(diffs.astype("timedelta64[ns]").astype(np.int64)))
            dev = np.abs(
                diffs.astype("timedelta64[ns]").astype(np.int64) - median_dt
            )
            if median_dt > 0 and float(np.max(dev)) > 0.10 * median_dt:
                raise ValueError(
                    "Non-uniform RAS output cadence detected (timestamps "
                    "vary by more than 10% from the median spacing). The "
                    "chunked transport design (Phase-C B3/B4) assumes a "
                    "near-uniform timestep. Re-export the RAS run at a "
                    "single uniform interval."
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

        # Phase J+1 (2026-05-23): bridge the Internal-BC metadata that
        # ``RASHDFDataSource.__read_internal_bc_metadata`` cached on the
        # mesh's ``attrs`` over to the two surfaces downstream code
        # expects.
        #
        # The HDF reader sets:
        #   * ``mesh.attrs['internal_bc_line_types']`` -- dict {name: type}
        #   * ``mesh.attrs['internal_bc_cells_by_line']`` -- dict {name: idx[]}
        #   * ``mesh.attrs['internal_bc_cells_all']`` -- flat idx[] of all
        #     Internal-BC cells
        # but those dataset-level attrs do not flow through to either:
        #   (a) the registry's VOLUME DataArray (so
        #       Constituent.set_boundary_conditions cannot see them via
        #       ``registry.get_variable(VOLUME).get_data().to_dataset()
        #       .attrs.get('internal_bc_line_types', {})``), nor
        #   (b) a registry variable named ``internal_bc_cells`` that
        #       utilities._apply_continuity_correction expects.
        # Bridge both here, in a single place, immediately after the
        # registry has been populated with VOLUME but before any
        # downstream consumer (constituents, continuity correction,
        # synthetic point sources) runs.
        self.__bridge_internal_bc_metadata()

        # calculate intermediate variables
        self.__update_calculated_variables()

        # Phase-D Unit A: register WET_MASK after VOLUME (+ optionally
        # FACE_HYD_DEPTH) is available; opt-in via wet_dry_metric.
        self.__populate_wet_mask()

        # Phase F (2026-05-21): register the continuity-corrected
        # ADVECTION_COEFFICIENT. Must happen after FLOW_ACROSS_FACE and
        # VOLUME are in the registry; consumed by the LHS instead of
        # the raw FLOW_ACROSS_FACE.
        from clearwater_riverine.utilities import register_advection_coefficient
        register_advection_coefficient(
            self.registry,
            continuity_correction=self.__continuity_correction,
        )

        # Phase I-1 (2026-05-21): register the diffusion method code +
        # method-specific parameters so ``calculate_coeff_to_diffusion_term``
        # can dispatch correctly. The integer code map mirrors what
        # the dispatcher reads: 0=constant, 1=elder, 2=eddy_viscosity,
        # 3=array. ``constant`` is the default and registers nothing
        # (preserving the legacy code path that just reads
        # DIFFUSION_COEFFICIENT directly).
        self._register_diffusion_method()

        # initialize constituents
        for constituent_name in list(constituents.keys()):
            self.__init_constituents(
                constituent_name=constituent_name,
                constituent_config=constituents[constituent_name]
            )

        # Phase J+1 (2026-05-23): generate synthetic point sources for
        # Internal-type BC lines. Each Internal BC's per-cell, per-time
        # flow Q(t,c) is multiplied by the user's per-time concentration
        # C(t) (already interpolated to the model time axis by
        # Constituent.set_boundary_conditions and stashed in
        # ``{name}_boundary_interp_internal``). The product is registered
        # under the existing point-source keys
        # ``{name}_point_source_flows`` / ``_point_source_concentrations``;
        # the existing RHS._calculate_point_sources adds these as mass-
        # only injections (no perturbation of the RAS flow field) at the
        # right Internal-BC cells. See Step 2 plan in the design memo.
        self.__emit_internal_bc_point_sources()

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

        # Phase F T2-A (2026-05-21): optional point-source CSV path.
        # Present only when the constituent's YAML declared a
        # ``point_sources`` block; absent entries default to no point
        # loads (backwards-compatible with existing configs).
        point_sources_path = self.__point_source_data_sources.get(constituent_name)

        self._constituents[constituent_name] = Constituent(
            constituent_name=constituent_name,
            registry=self.registry,
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            constituent_config=constituent_config,
            start_datetime=self._start_datetime,
            point_sources_path=point_sources_path,
        )


    def _register_diffusion_method(self) -> None:
        """Register the diffusion-dispatch method + parameters on the registry.

        Phase I-1 (2026-05-21): translates the config-side method
        name and parameter dict into the FloatVariable entries the
        dispatcher in ``utilities.calculate_coeff_to_diffusion_term``
        reads. ``constant`` is the default and registers nothing.
        """
        from clearwater_data.variables.float import FloatVariable
        method_codes = {"constant": 0, "elder": 1, "eddy_viscosity": 2, "array": 3}
        method = self.__diffusion_method
        if method not in method_codes:
            raise ValueError(
                f"Unknown diffusion method {method!r}. Expected one of "
                f"{sorted(method_codes)}."
            )
        if method == "constant":
            return
        # Register method code so the dispatcher fires.
        code = method_codes[method]
        if "diffusion_method" in self.registry:
            self.registry.unregister("diffusion_method")
        self.registry.register("diffusion_method", FloatVariable(float(code)))
        # Register method-specific scalar params. The dispatcher looks
        # for these by name.
        params = self.__diffusion_params or {}
        if method == "elder" and "alpha" in params:
            if "diffusion_alpha" in self.registry:
                self.registry.unregister("diffusion_alpha")
            self.registry.register(
                "diffusion_alpha", FloatVariable(float(params["alpha"])),
            )
        if method == "eddy_viscosity" and "schmidt" in params:
            if "diffusion_schmidt" in self.registry:
                self.registry.unregister("diffusion_schmidt")
            self.registry.register(
                "diffusion_schmidt", FloatVariable(float(params["schmidt"])),
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
        volume = self.registry.get_variable(VOLUME).get_data()
        depth = None
        if self.__wet_dry_metric in ("depth", "both"):
            # Phase F (2026-05-21): auto-register FACE_HYD_DEPTH when
            # the wet-dry metric needs depth. Most RAS HDFs ship the
            # "Cell Hydraulic Depth" temporal output, but minimal-output
            # decks (e.g., the Santiam-Salem subset used in Phase F
            # validation) only write Water Surface + Depth-via-lookup.
            # In that case canonical computes it from WSE - cell-bed
            # elevation via the calculated-variable path so depth-based
            # wet-mask metrics work uniformly across all HDFs.
            #
            # Phase J+1 (2026-05-23): force-refresh on every call. The
            # previous "register only if absent" form left a stale
            # chunk-1 FACE_HYD_DEPTH (with chunk-1 time coord) in the
            # registry at chunk-2 boundary. The wet-mask then computed
            # ``(volume_chunk2 > V_min) & (depth_chunk1 > h_min)``,
            # which xarray broadcasts onto the *intersection* of the
            # two time axes (one shared stamp), producing a 1-timestep
            # WET_MASK that fails downstream LHS lookups. Always
            # recompute so the time coord matches VOLUME's current
            # chunk window.
            if FACE_HYD_DEPTH in self.registry:
                self.registry.unregister(FACE_HYD_DEPTH)
            from clearwater_riverine.utilities import calculate_face_hyd_depth
            self.registry.register(
                FACE_HYD_DEPTH,
                calculate_face_hyd_depth(self.registry),
            )
            depth = self.registry.get_variable(FACE_HYD_DEPTH).get_data()
        mask = compute_wet_mask(
            volume,
            depth,
            h_min=self.__wet_dry_h_min,
            V_min=self.__wet_dry_v_min,
            metric=self.__wet_dry_metric,
        )
        # Phase J+1 (2026-05-23) diagnostic
        try:
            t = mask.time.values if hasattr(mask, 'time') else None
            tdt = mask.time.dtype if hasattr(mask, 'time') else None
            vt = volume.time.values if hasattr(volume, 'time') else None
            vtdt = volume.time.dtype if hasattr(volume, 'time') else None
            print(f"      [WET_MASK refresh] volume.time dtype={vtdt}, "
                  f"range [{vt[0] if vt is not None and len(vt)>0 else '?'} "
                  f".. {vt[-1] if vt is not None and len(vt)>0 else '?'}] "
                  f"({len(vt) if vt is not None else '?'} stamps); "
                  f"mask.time dtype={tdt}, "
                  f"range [{t[0] if t is not None and len(t)>0 else '?'} "
                  f".. {t[-1] if t is not None and len(t)>0 else '?'}] "
                  f"({len(t) if t is not None else '?'} stamps)")
        except Exception as e:
            print(f"      [WET_MASK refresh] diag print failed: {e}")
        if WET_MASK in self.registry:
            self.registry.unregister(WET_MASK)
        self.registry.register(
            WET_MASK,
            DataArrayVariable(mask, space_dimension=NFACE),
        )


    def __bridge_internal_bc_metadata(self):
        """Phase J+1: bridge Internal-BC metadata from data-source mesh to
        the registry surfaces that downstream code expects.

        Called once at init (after VOLUME is registered). Idempotent.
        No-op when the data source has no Internal BCs (External-only
        HDFs, older HDFs that don't have the Internal Cells dataset).

        Two bridges:

        1. **VOLUME DataArray attrs** -- copy ``internal_bc_line_types``
           and ``internal_bc_cells_by_line`` from the data-source mesh's
           Dataset-level attrs onto the registered VOLUME DataArray's
           own attrs. ``Constituent.set_boundary_conditions`` (Step 3)
           reads via ``registry.get_variable(VOLUME).get_data().attrs
           .get('internal_bc_line_types', {})`` -- DIRECTLY off the
           DataArray's attrs, NOT via ``.to_dataset().attrs``. xarray's
           ``DataArray.to_dataset()`` does NOT propagate the DataArray's
           attrs to the resulting Dataset; they end up as variable-level
           attrs on the contained variable instead. The reader was
           updated for this on 2026-05-23 (bug discovered via a fresh-
           init probe); the writer here (``volume_da.attrs[...] = ...``)
           is unchanged.

        2. **Registry variable ``internal_bc_cells``** -- a flat array
           of all Internal-BC cell indices, registered as a
           DataArrayVariable. ``utilities._apply_continuity_correction``
           (Step 4) reads via ``'internal_bc_cells' in registry`` +
           ``registry.get_variable('internal_bc_cells').get_data()``,
           and uses the cell index set to exclude those cells from the
           residual-redistribution step.
        """
        hydro = self.__variable_data_sources['hydrodynamic_model']
        if not hasattr(hydro, 'mesh') or not hasattr(hydro.mesh, 'attrs'):
            return

        line_types = hydro.mesh.attrs.get('internal_bc_line_types', {}) or {}
        cells_by_line = hydro.mesh.attrs.get('internal_bc_cells_by_line', {}) or {}
        cells_all = hydro.mesh.attrs.get(
            'internal_bc_cells_all', np.array([], dtype=np.int64)
        )

        # Bridge 1: VOLUME DataArray attrs (for Step 3 / constituents.py).
        # The registry stores the DataArray by reference -- modifying
        # .attrs in place propagates to subsequent reads.
        try:
            volume_da = self.registry.get_variable(VOLUME).get_data()
            if line_types:
                volume_da.attrs['internal_bc_line_types'] = line_types
            if cells_by_line:
                volume_da.attrs['internal_bc_cells_by_line'] = cells_by_line
        except Exception as e:
            warnings.warn(
                f"Could not attach internal_bc_* attrs to VOLUME DataArray "
                f"({type(e).__name__}: {e}). Internal-BC set_boundary_conditions "
                f"branching may not activate.",
                UserWarning,
                stacklevel=2,
            )

        # Bridge 2: registry variable for the continuity correction
        # (Step 4 / utilities.py). Register even when empty so callers
        # can use a simple ``'internal_bc_cells' in registry`` check.
        cells_arr = np.asarray(cells_all, dtype=np.int64)
        if 'internal_bc_cells' in self.registry:
            self.registry.unregister('internal_bc_cells')
        self.registry.register(
            'internal_bc_cells',
            DataArrayVariable(
                xr.DataArray(cells_arr, dims=('internal_bc_cell',))
            ),
        )


    def __emit_internal_bc_point_sources(self):
        """Phase J+1: synthesize per-cell point sources for Internal BCs.

        For each Internal-type BC line, RAS reports per-cell flow Q(t,c)
        (m^3/s) in
        ``Results/Unsteady/.../Boundary Conditions/<line> - Flow``,
        already cached on the data source by
        ``__read_internal_bc_metadata``. Multiplying by the user's
        per-time concentration C(t) (interpolated to the model time
        axis by ``Constituent.set_boundary_conditions`` and stashed in
        the registry variable ``{name}_boundary_interp_internal``)
        gives a per-cell mass injection rate. Register that under the
        existing point-source keys; the existing
        ``RHS._calculate_point_sources`` (linalg.py) handles the
        addition to the transport solve as a *mass-only* injection
        (flow is RAS's already, not perturbed by CWR).

        No-op when the data source has no Internal BCs OR when no
        constituent has a BC entry whose ``RAS2D_TS_Name`` matches an
        Internal-type line.

        Called once at init (after constituents are constructed) and
        again at every chunk boundary (mirrors the Phase H-1
        point-source CSV reload pattern in ``__load_new_chunk``).
        """
        hydro = self.__variable_data_sources['hydrodynamic_model']
        cells_by_line = (hydro.mesh.attrs.get('internal_bc_cells_by_line', {})
                         if hasattr(hydro, 'mesh') else {})
        flows_by_line = getattr(hydro, '_internal_bc_flows', {})
        if not cells_by_line or not flows_by_line:
            return

        # Map current chunk's model time grid -> indices in the HDF's
        # full time axis (so per-cell Q can be sliced to chunk).
        all_dts = pd.DatetimeIndex(hydro.all_datetimes.values)
        volume_da = self.registry.get_variable(VOLUME).get_data()
        chunk_dts = pd.DatetimeIndex(volume_da.time.values)
        chunk_idx = all_dts.get_indexer(chunk_dts)
        if (chunk_idx < 0).any():
            warnings.warn(
                f"Could not align current chunk's time grid to the HDF "
                f"full time axis for Internal-BC point-source synthesis. "
                f"Missing indices: {int((chunk_idx < 0).sum())}. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            return
        n_chunk_time = len(chunk_dts)

        # Total nface for the output point-source arrays' shape.
        nface = int(self.registry.get_variable(NFACE).get_data())

        for constituent_name in self._constituents.keys():
            internal_key = f"{constituent_name}_boundary_interp_internal"
            if internal_key not in self.registry:
                # Constituent has no Internal-type BC entries.
                continue
            interp_internal = self.registry.get_variable(internal_key).get_data()
            # interp_internal shape: (time, RAS2D_TS_Name)

            # Initialize zero (time, nface) arrays; we'll only set
            # values at Internal-BC cell indices.
            flows_arr = np.zeros((n_chunk_time, nface), dtype=np.float64)
            concs_arr = np.zeros((n_chunk_time, nface), dtype=np.float64)

            for line_name, cells in cells_by_line.items():
                cells = np.asarray(cells, dtype=np.int64)
                if cells.size == 0:
                    continue
                if line_name not in flows_by_line:
                    continue
                full_flows = flows_by_line[line_name]  # (full_time, n_cells)
                # Slice to current chunk's time indices.
                chunk_flows = full_flows[chunk_idx, :]  # (n_chunk_time, n_cells)
                # Get the interpolated concentration for this BC line.
                # interp_internal is (time, RAS2D_TS_Name); .sel by name.
                try:
                    line_C = interp_internal.sel(RAS2D_TS_Name=line_name).values
                except (KeyError, ValueError):
                    # Constituent's BC CSV has no entry for this Internal
                    # BC line. Leave its injection at 0 for this line.
                    continue
                # line_C shape: (n_chunk_time,); broadcast to per-cell.
                # Per-cell flow x scalar-per-time C:
                flows_arr[:, cells] = chunk_flows
                concs_arr[:, cells] = line_C[:, np.newaxis]

            # Wrap as DataArrays with the chunk time coord + nface dim.
            time_coord = volume_da.time.values
            flows_da = xr.DataArray(
                flows_arr,
                dims=('time', NFACE),
                coords={'time': time_coord},
                attrs={
                    'Units': 'm3/s',
                    'long_name': 'Synthetic point-source flow (Internal-BC injection)',
                    'internal_bc_synthetic': 1,
                },
            )
            concs_da = xr.DataArray(
                concs_arr,
                dims=('time', NFACE),
                coords={'time': time_coord},
                attrs={
                    'Units': 'constituent reporting units',
                    'long_name': 'Synthetic point-source concentration (Internal-BC injection)',
                    'internal_bc_synthetic': 1,
                },
            )

            flows_key = f"{constituent_name}_point_source_flows"
            concs_key = f"{constituent_name}_point_source_concentrations"
            if flows_key in self.registry:
                self.registry.unregister(flows_key)
            if concs_key in self.registry:
                self.registry.unregister(concs_key)
            self.registry.register(
                flows_key, DataArrayVariable(flows_da, space_dimension=NFACE),
            )
            self.registry.register(
                concs_key, DataArrayVariable(concs_da, space_dimension=NFACE),
            )
            # Tell the constituent it now has point sources, so the
            # transport engine's RHS reads them (mirrors what
            # Constituent._load_point_sources sets).
            constituent = self._constituents[constituent_name]
            constituent.has_point_sources = True


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
                time_step=timedelta(seconds=self._representative_dt_seconds()),
                variables=self.__output_variables,
                chunk_size=self.__chunk_size,
                spatial_field=NFACE,
                spatial_field_values=self.registry.get_variable(VOLUME).get_data().nface,
                init_template=init_template,
            )
        else:
            # Phase G-4 (2026-05-21): pass the actual RAS time vector
            # via the new ``time_coord`` kwarg so the zarr template
            # carries the same non-uniform stamps the chunk write
            # later targets. Falling back to start+end+time_step
            # synthesized a uniform-hourly grid that, on a RAS HDF
            # with 59/60/61-minute jitter, left every constituent
            # variable all-NaN after finalize because the chunk write
            # could not align stamps with the template. Phase F
            # Santiam-Salem validation pre-G-4 reproduced this.
            ras_times = pd.DatetimeIndex(self.registry.get_variable(VOLUME).get_data().time.values)
            self.__output_data_store = ZarrDataStore(
                store_path=self.__simulation_directory / "model_outputs.zarr",
                start_date=self._start_datetime,
                end_date=self._end_datetime,
                time_step=timedelta(seconds=self._representative_dt_seconds()),
                time_coord=ras_times,
                variables=self.__output_variables,
                spatial_field=NFACE,
                spatial_field_values=self.registry.get_variable(VOLUME).get_data().nface,
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

        # Phase J+1 amendment (2026-05-24): re-bridge Internal-BC
        # metadata onto the freshly registered VOLUME DataArray. The
        # temporal-variables refresh above unregistered the chunk-1
        # VOLUME DataArray (which carried ``internal_bc_line_types``
        # and ``internal_bc_cells_by_line`` on its ``.attrs`` via
        # ``__bridge_internal_bc_metadata`` at __init__) and registered
        # a FRESH DataArray from ``hydro.mesh[VOLUME]``. The fresh
        # DataArray's ``.attrs`` does not carry the line-types /
        # cells-by-line dicts (those live on the Dataset-level
        # ``hydro.mesh.attrs``, which survive chunk advance, but NOT
        # on the per-variable DataArray's attrs). Without this
        # re-bridge, ``Constituent.set_boundary_conditions`` below
        # reads ``volume_da.attrs.get('internal_bc_line_types', {})``,
        # finds it empty, classifies every BC line as External, and
        # NEVER re-registers ``{name}_boundary_interp_internal``;
        # ``__emit_internal_bc_point_sources`` at the end of this
        # method then sees no constituent has the interim variable
        # and returns without refreshing the per-chunk synthetic
        # point sources, leaving ``{name}_point_source_flows`` bound
        # to chunk 1's time axis and causing a ``KeyError`` at the
        # first transport step of chunk 2. (The
        # ``internal_bc_cells`` registry variable is unaffected
        # because it persists across chunks under its own name.)
        self.__bridge_internal_bc_metadata()

        self.__update_calculated_variables()
        # Phase-D Unit A: refresh WET_MASK for the new chunk's time window.
        self.__populate_wet_mask()
        # Phase F (2026-05-21): refresh ADVECTION_COEFFICIENT (continuity
        # correction is timestep-dependent, so it must be recomputed for
        # each chunk's time window).
        from clearwater_riverine.utilities import register_advection_coefficient
        register_advection_coefficient(
            self.registry,
            continuity_correction=self.__continuity_correction,
        )
        # Phase H-1 (2026-05-21): refresh per-constituent point-source
        # arrays for the new chunk's time window. Without this, the
        # arrays remain bound to chunk 1's time axis (built during
        # ``Constituent.__init__``) and RHS._calculate_point_sources'
        # ``registry.get_at_time(flows_key, next_time)`` raises
        # ``KeyError`` at the first stamp of chunk 2 because that
        # stamp is past the registered axis. Re-read the CSV and
        # re-register on the current chunk's grid; mirrors the
        # WET_MASK and ADVECTION_COEFFICIENT refresh patterns above.
        for constituent_name, constituent in self._constituents.items():
            ps_path = self.__point_source_data_sources.get(constituent_name)
            if ps_path is not None:
                from pathlib import Path as _PathH1
                constituent._load_point_sources(
                    registry=self.registry,
                    filepath=_PathH1(ps_path),
                )

        # Phase J+1 (2026-05-23): refresh per-constituent boundary-
        # condition source on the new chunk's time window. The BC source
        # is loaded once at ``Constituent.__init__`` via the data source's
        # ``.read()``; for non-chunked providers that's the full file,
        # but the subsequent ``set_boundary_conditions`` interp pass in
        # ``__load_new_chunk`` operates against the CURRENT chunk's
        # constituent time axis, and the BC source the registry holds
        # can drift out of sync with the chunked target if the in-memory
        # source's time index was modified at chunk-1 set up. The
        # symptom: ``set_boundary_conditions``'s post-interp validator
        # raises with "N of N NaN" at chunk 2 even though the on-disk
        # BC CSV covers the full window.
        #
        # Re-read the BC source from disk at the chunk boundary and re-
        # register; mirrors the point-source refresh above. Cheap (BC
        # CSVs are small; a 10-day daily-cadence two-line CSV is ~700
        # bytes), explicit, and decoupled from whether the underlying
        # data source is chunked or not.
        for constituent_name, constituent in self._constituents.items():
            bc_source = self.__boundary_condition_data_sources.get(
                constituent_name
            )
            if bc_source is None:
                continue
            try:
                fresh = bc_source.read(constituent_name)
            except Exception:
                # Best-effort: if the data source can't re-read (e.g.
                # ChunkedDataSource requiring read_chunk), leave the
                # existing registry entry alone and let
                # set_boundary_conditions handle it.
                continue
            bkey = f"{constituent_name}_boundary"
            if bkey in self.registry:
                self.registry.unregister(bkey)
            self.registry.register(bkey, fresh)
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

            # Real-world-robustness sanitize (Phase J+1, 2026-05-23).
            # Real river meshes have large dry-cell fractions (the river
            # channel occupies a small part of the bounding-box mesh; the
            # rest is floodplain that stays dry through the simulation).
            # During the transport solve the LHS row for a cell with
            # V[t+1] ~ 0 is singular and the sparse solver returns NaN
            # for that cell's concentration. This is a structural
            # consequence of the wet/dry regime, not a bug in the IC
            # source or the solver. At a chunk boundary the
            # ``ic_value`` we just pulled from the registry is exactly
            # the end-of-prev-chunk transport state, which carries those
            # dry-cell NaNs. Without sanitization the per-constituent
            # NaN validator in ``set_initial_conditions`` (Phase F T2-D)
            # refuses to load chunk 2+ on any real-world mesh.
            #
            # Sanitize branch on ``constituent.is_intensive`` (Phase-D
            # Unit D1):
            #   * Extensive species (mass concentrations: Ap, NH4, NO3,
            #     TIP, DOX, dye, ...): NaN -> 0. 0 mg/L at a dry cell
            #     is the physically defensible "no mass present" value.
            #   * Intensive properties (temperature, ...): NaN -> median
            #     of the finite (wet-cell) values. 0 deg C at a dry
            #     cell is NOT physically defensible: it drags the
            #     spatial median toward 0 and produces unrealistic
            #     temperature comparisons downstream. The wet-cell
            #     median is the best estimate of "bulk temperature in
            #     the absence of any cell-specific information."
            #     Degenerate case (all NaN) falls back to 0 with warning.
            #
            # The input-NaN validator stays in place for genuine input
            # bugs (e.g. an IC CSV with missing rows).
            is_intensive = bool(getattr(constituent, "is_intensive", False))

            def _sanitize_arr(arr: np.ndarray) -> tuple[np.ndarray, int]:
                """Return (sanitized array, count of NaNs replaced)."""
                finite_mask = np.isfinite(arr)
                n_nan = int(arr.size - np.sum(finite_mask))
                if n_nan == 0:
                    return arr, 0
                if is_intensive:
                    if finite_mask.any():
                        fill_value = float(np.nanmedian(arr))
                    else:
                        fill_value = 0.0
                        print(
                            f"      WARN chunk boundary {self.__current_time}: "
                            f"intensive constituent {constituent_name} has "
                            f"NO finite values to draw a median from; "
                            f"falling back to 0.0. Downstream comparisons "
                            f"may be unphysical."
                        )
                else:
                    fill_value = 0.0
                return np.where(finite_mask, arr, fill_value), n_nan

            if isinstance(ic_value, np.ndarray) and not np.all(np.isfinite(ic_value)):
                sanitized, n_nan = _sanitize_arr(ic_value)
                kind = "intensive" if is_intensive else "extensive"
                fill_descr = (f"median of finite values"
                              if is_intensive else "0")
                print(
                    f"      chunk boundary {self.__current_time}: "
                    f"sanitizing {n_nan:,}/{ic_value.size:,} NaN values in "
                    f"{constituent_name} end-of-prev-chunk state "
                    f"(dry-cell artifacts from transport solve; "
                    f"{kind} constituent, replaced with {fill_descr})"
                )
                ic_value = sanitized
            elif isinstance(ic_value, xr.DataArray):
                # Same logic for the DataArray-valued path (Phase-D D2).
                vals = ic_value.values
                if not np.all(np.isfinite(vals)):
                    sanitized, n_nan = _sanitize_arr(vals)
                    kind = "intensive" if is_intensive else "extensive"
                    fill_descr = (f"median of finite values"
                                  if is_intensive else "0")
                    print(
                        f"      chunk boundary {self.__current_time}: "
                        f"sanitizing {n_nan:,}/{vals.size:,} NaN values in "
                        f"{constituent_name} end-of-prev-chunk state "
                        f"(dry-cell artifacts from transport solve; "
                        f"{kind} constituent, replaced with {fill_descr})"
                    )
                    # Re-wrap the sanitized values as a DataArray with
                    # the same dims/coords as the original.
                    ic_value = xr.DataArray(
                        sanitized,
                        dims=ic_value.dims,
                        coords=ic_value.coords,
                        attrs=ic_value.attrs,
                    )

            constituent.reset_initial_conditions(self.registry, ic_value)
            constituent.register_constituent(self.registry)
            constituent.set_initial_conditions(self.registry, self.__current_time)
            constituent.set_boundary_conditions(self.registry)
        self.__resuming_ics = None

        # Phase J+1 (2026-05-23): refresh Internal-BC synthetic point
        # sources for the new chunk's time window. set_boundary_conditions
        # has just re-registered the {name}_boundary_interp_internal
        # interim variable on the chunk-2 time grid; we now re-slice the
        # data source's per-cell Q to the chunk-2 indices and rebuild
        # {name}_point_source_flows / _concentrations so the existing
        # RHS._calculate_point_sources reads the right chunk's values.
        # Mirrors the Phase H-1 point-source CSV reload pattern above.
        self.__emit_internal_bc_point_sources()

    
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
            #
            # Phase J+1 amendment (2026-05-24): when ``is_last=True``,
            # ``accumulate_chunk_mass_balance`` looks up VOLUME and the
            # constituent array at the end timestamp to capture the
            # run's end snapshot. Passing ``self._end_datetime``
            # directly raises ``KeyError`` whenever the caller has
            # advanced fewer transport steps than the configured window
            # implies -- e.g., the runner does ``--days 10`` updates
            # against an HDF whose final stamp is 10.4 days past the
            # start, so ``self._end_datetime`` is past the last chunk's
            # last grid stamp. The mass balance then fails BEFORE the
            # ``write_chunk`` loop below, dropping the final chunk's
            # output. Use ``self.__current_time`` for ``is_last=True``
            # instead: it is the actual last-computed stamp (= the last
            # stamp with data in the registry after the user's final
            # ``update()`` call) and is always in the resident chunk's
            # grid by construction.
            end_for_mb = (
                self.__current_time if is_last else self._end_datetime
            )
            for constituent_name in self._constituents:
                self.__mb_acc = accumulate_chunk_mass_balance(
                    self.__mb_acc,
                    self.registry,
                    constituent_name,
                    self._start_datetime,
                    end_for_mb,
                    drop_last_slot=not is_last,
                    is_last=is_last,
                )

        # C3b: record the chunk boundary up through which the accumulator
        # is consistent. Interior finalize: the accumulator has chunks
        # 1..K and resume should start from this timestamp. Final finalize
        # (is_last=True) closes the run; not a resume point.
        if not is_last:
            self.__last_finalized_boundary = self.__current_time

        # Interior chunks drop the shared trailing slot; the next chunk
        # owns it as its first slot, so the cross-chunk write covers each
        # stamp exactly once. For the final chunk (is_last=True) there is
        # no next chunk, so keep the trailing slot -- otherwise the last
        # computed transport result is silently dropped from the zarr.
        end_slice = None if is_last else -1
        for variable_name in self.__output_variables:
            # calculate mass flux, if necessary
            # TODO: clean up chunk indexing
            variable = (
                self.registry.get_variable(variable_name)
                .get_data()
                .isel(time=slice(0, end_slice))
            )
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
            # Phase F: use the per-step actual dt (not the median)
            # so ``current_time + time_step`` lands on the next exact
            # RAS stamp. This matters for non-uniform stamps.
            time_step=timedelta(seconds=self._current_dt_seconds()),
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