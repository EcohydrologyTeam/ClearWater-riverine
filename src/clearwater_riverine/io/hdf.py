from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    Union,
)

import warnings
import h5py
import xarray as xr
# import variables
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import clearwater_riverine
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_riverine.io.mesh_cache import (
    CACHE_SCHEMA_VERSION,
    build_cache_key_inputs,
    cache_file_path,
    cache_key_hash,
    default_cache_dir,
    read_cache,
    write_cache,
)

from clearwater_riverine.mesh import (
    instantiate_model_mesh,
    load_model_mesh
)
from clearwater_riverine.utilities import (
    _compute_cell_volumes,
    _compute_face_areas,
)
from clearwater_riverine.variables import (
    BOUNDARY_CONDITION_LINE_ID,
    BOUNDARY_FACE_INDEX,
    BOUNDARY_NAME,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    DIFFUSION_COEFFICIENT,
    NEDGE,
    NFACE,
    NODE_X,
    NODE_Y,
    TIME,
    FACE_NODES,
    EDGE_NODES,
    EDGE_FACE_CONNECTIVITY,
    FACE_X,
    FACE_Y,
    FACE_SURFACE_AREA,
    FACE_TO_FACE_DISTANCE,
    GATE_CONNECTIVITY,
    GATE_FLOW,
    EDGE_LENGTH,
    EDGE_VELOCITY,
    EDGE_VERTICAL_AREA,
    LOOKUP_ELEVATION,
    LOOKUP_VOLUME,
    LOOKUP_WETTED_SURFACE_AREA,
    WATER_SURFACE_ELEVATION,
    FLOW_ACROSS_FACE,
    VOLUME,
    VOLUME_ELEVATION_INFO,
    VOLUME_ELEVATION_VALUES,
    VOLUME_ELEVATION_LOOKUP,
    # Phase I-1 (2026-05-21): variables required by the non-constant
    # diffusion methods (Elder, eddy viscosity). Each is OPTIONAL in
    # the source RAS HDF; the reader registers them only when the
    # dataset is present, so existing decks that ship only the
    # minimal output set continue to work with the constant-diffusion
    # default path.
    MANNINGS_N,
    EDDY_VISCOSITY,
    CELL_EDDY_VISCOSITY_X,
    CELL_EDDY_VISCOSITY_Y,
    FACE_HYD_DEPTH,
    FACE_VEL_X,
    FACE_VEL_Y,
)

def _parse_attributes(dataset) -> Dict[str, Any]:
    """
    Parse the HDF5 attributes array,
    convert binary strings to Python strings,
    and return a dictionary of attributes.
    """
    attrs = {}
    for key, value in dataset.attrs.items():
        if isinstance(value, np.bytes_):
            attrs[key] = value.decode('ascii')
        elif isinstance(value, np.ndarray):
            values = []
            for v in value:
                if isinstance(v, np.bytes_):
                    values.append(v.decode('ascii'))
                else:
                    values.append(v)
            attrs[key] = values
        else:
            attrs[key] = value
    return attrs


def _hdf_to_xarray(
    dataset,
    dims,
    attrs=None,
    time_constraint: Optional[Tuple] = (None, None),
) -> xr.DataArray:
    """Read n-dimensional HDF5 dataset and return it as an xarray.DataArray"""
    if attrs is None:
        try:
            attrs = _parse_attributes(dataset)
        except AttributeError as e:
            attrs = {}
    if time_constraint != (None, None):
        data_to_read = dataset[()][time_constraint[0]: time_constraint[1]]
    else:
        data_to_read = dataset[()]
    data_array = xr.DataArray(
        data_to_read,
        dims=dims,
        attrs=attrs
    )
    return data_array


def _hdf_to_dataframe(dataset) -> pd.DataFrame:
    """Read n-dimensional HDF5 dataset and return it as an pandas DataFrame"""
    attrs = _parse_attributes(dataset)
    df = pd.DataFrame(
        dataset[()],
        columns=attrs['Column']
    )
    return df


class RASHDFDataSource:
    """
    Reads RAS hydrodynamic data required for WQ calculations.
    """
    def __init__(self, **kwargs) -> None:
        self.ras_hdf_path: str = kwargs.pop("ras_hdf_path")
        self.start_datetime: datetime = kwargs.pop("start_datetime", None)
        self.end_datetime: datetime = kwargs.pop("end_datetime", None)
        self.datetime_range = (self.start_datetime, self.end_datetime)
        self.calculated_variables = kwargs.pop("calculated_variables", {})
        self.__rebuild_mesh: bool = kwargs.pop("rebuild_mesh", False)
        self.__mesh_cache_dir = kwargs.pop("mesh_cache_dir", None)
        # Phase J+1 (Corvallis 2026-05-23): per-variable opt-ins for the
        # geometry-derived fallbacks. Default ``False`` is fail-loud: if
        # the corresponding required temporal dataset is absent in the
        # HDF, ``__probe_temporal_fallbacks`` raises with the RAS option
        # to enable AND the YAML / CLI flag to opt in. Set to ``True``
        # to synthesize the variable from the static lookup table + WSE.
        # See ``utilities._compute_cell_volumes`` /
        # ``utilities._compute_face_areas`` for the reconstruction and
        # ``design/missing_temporal_fallback.md`` for the fidelity
        # validation against the Santiam-Salem fixture.
        self._allow_cell_volume_fallback: bool = bool(
            kwargs.pop("allow_cell_volume_fallback", False)
        )
        self._allow_face_flow_fallback: bool = bool(
            kwargs.pop("allow_face_flow_fallback", False)
        )
        # Set by ``__probe_temporal_fallbacks`` once the HDF is open.
        # When True, the corresponding ``__read_temporal_variables``
        # branch synthesizes the variable from the cached lookup tables
        # rather than reading from disk.
        self._volume_fallback_active: bool = False
        self._face_flow_fallback_active: bool = False
        # Lookup-table DataFrames cached by ``__preload_fallback_lookups``.
        # ``None`` until preload runs; absence is checked at synthesis
        # time so a stale cache load cannot silently skip the preload.
        self._volume_elev_info_df: Optional[pd.DataFrame] = None
        self._volume_elev_values_df: Optional[pd.DataFrame] = None
        self._face_area_elev_info_df: Optional[pd.DataFrame] = None
        self._face_area_elev_values_df: Optional[pd.DataFrame] = None
        self._face_length_df: Optional[pd.DataFrame] = None
        self._face_cell_indexes_df: Optional[pd.DataFrame] = None

        self.mesh = instantiate_model_mesh()
        self.temporal_variables = {
            VOLUME: NFACE,
            EDGE_VELOCITY: NEDGE,
            WATER_SURFACE_ELEVATION: NFACE,
            FLOW_ACROSS_FACE: NEDGE,
        }
        self.static_variables = {
            FACE_SURFACE_AREA: NFACE,
            EDGE_LENGTH: NEDGE,
        }
        self.topology_variables = [
            FACE_X,
            FACE_Y,
            EDGE_FACE_CONNECTIVITY,
            FACE_NODES,
            NODE_X,
            NODE_Y,
        ]
        self.lookup_variables = [
            LOOKUP_ELEVATION,
            LOOKUP_VOLUME,
            LOOKUP_WETTED_SURFACE_AREA,
        ]
        self.boundary_variables = [
            BOUNDARY_CONDITION_LINE_ID,
            BOUNDARY_FACE_INDEX,
            BOUNDARY_NAME,
        ]

        # Add internal ones or modify as needed
        self.calculated_variables.update({
            EDGE_VERTICAL_AREA: True,
            FACE_TO_FACE_DISTANCE: True,
            COEFFICIENT_TO_DIFFUSION_TERM: True,
            CHANGE_IN_TIME: True,
        })

        ## TODO: add datetime validation somewhere
        # self.__validate_datetime_range()

        # Phase J+1 (2026-05-23) -- pre-flight schema check.
        # Runs BEFORE any expensive static-build / cache work. Detects:
        #   (a) RAS 2025 (Mesh Version 2.0) layout, which the current
        #       reader cannot consume; raises with a pointer to the
        #       design spec so the user sees the actionable error in
        #       seconds, not after a 60-second mesh walk.
        #   (b) Missing required optional temporal outputs (Cell Volume,
        #       Face Flow). If absent AND the corresponding fallback flag
        #       is False, raises with the RAS option to enable AND the
        #       YAML / CLI flag to opt into the fallback. If absent AND
        #       the fallback flag is True, marks the fallback as active
        #       (lookup-table DataFrames are loaded later in
        #       __probe_temporal_fallbacks after self.paths is built).
        # See design/ras2025_format_analysis.md and
        # design/missing_temporal_fallback.md for the underlying
        # robustness rationale: real-world RAS plans frequently ship
        # without the optional outputs and on the newer schema, and
        # users deserve immediate, actionable feedback rather than a
        # deep init-time crash.
        self.__preflight_schema_check()

        # --- static-geometry cache (Phase-C C1b) ---------------------
        # __build_static_from_hdf walks thousands of small geometry
        # datasets plus the per-cell volume-elevation lookup loop; on a
        # real corridor (Albany 587k cells) that costs 30-60 min and is
        # fully deterministic from the HDF. Cache the static result on
        # disk. Temporal read()/read_chunk() always re-open the HDF for
        # the (cheap, sequential) time slabs, so they are unaffected and
        # the cache never depends on the requested datetime window.
        hdf_path = Path(self.ras_hdf_path)
        cache_disabled = False
        if self.__mesh_cache_dir is None:
            cache_dir = default_cache_dir(hdf_path)
        else:
            cache_dir = Path(self.__mesh_cache_dir)
            if str(cache_dir) in ('', '.'):
                cache_disabled = True

        cache_path = None
        if not cache_disabled:
            try:
                cache_path = cache_file_path(
                    cache_dir,
                    cache_key_hash(
                        build_cache_key_inputs(
                            hdf_path=hdf_path,
                            cwr_version=clearwater_riverine.__version__,
                            extra={},
                        )
                    ),
                )
            except OSError:
                # stat() failure -- treat as cache disabled.
                cache_disabled = True

        # Hit path: rehydrate the static state and skip the HDF walk.
        # Any unexpected payload shape falls through to a clean rebuild,
        # so a cache can never produce incorrect data.
        rehydrated_from_cache = False
        if (
            not cache_disabled
            and not self.__rebuild_mesh
            and cache_path is not None
        ):
            payload = read_cache(cache_path)
            if payload is not None:
                try:
                    self.__rehydrate_static(payload)
                    rehydrated_from_cache = True
                except Exception:
                    pass

        if not rehydrated_from_cache:
            # Miss / forced rebuild / stale: full static build from the HDF.
            self.__build_static_from_hdf()

            # Best-effort cache write (a write failure must not break a run).
            if not cache_disabled and cache_path is not None:
                try:
                    write_cache(cache_path, self.__static_payload())
                except Exception:
                    pass

        # Phase J+1 (Corvallis 2026-05-23): probe for the optional
        # temporal datasets ``Cell Volume`` and ``Face Flow`` and either
        # set the fallback flags + preload lookup tables, or fail loud
        # with an actionable error. Runs after both the cache-hit and
        # cache-miss paths, so the behaviour is the same whether the
        # static state came from disk or from a fresh HDF walk. The
        # lookup tables are intentionally NOT cached on disk -- they
        # are only needed when a fallback is active, they are cheap to
        # re-read (a few MB), and skipping the cache means changing
        # the ``allow_*_fallback`` flag between runs does not require
        # cache invalidation.
        self.__probe_temporal_fallbacks()

    def __preflight_schema_check(self) -> None:
        """Pre-flight schema detection. Runs BEFORE expensive static build.

        Fail-fast pass that catches the two classes of HDF that the current
        reader cannot consume, with clean actionable errors:

        1. **RAS 2025 layout** (Mesh ``Version 2.0``,
           ``Geometry/2D Flow Areas/Mesh/...``). The current reader
           targets the v5/6/7+ schema and is not v2025-compatible.
           Raises ``NotImplementedError`` pointing at the design spec
           that maps the v2025 layout for the future v2025 reader work.
        2. **Missing required optional temporal outputs** (``Cell Volume``,
           ``Face Flow``). Raises ``KeyError`` if absent AND the
           corresponding ``allow_*_fallback`` flag is False. Real-world
           RAS plans frequently ship without these in the optional
           output set; the user must either re-run RAS with them
           enabled or opt into the geometry-derived fallback. The
           fail-fast pre-flight surfaces this before a 60-second mesh
           walk, so the user can re-act in seconds.

        Does NOT preload the lookup-table DataFrames; that happens later
        in ``__probe_temporal_fallbacks`` after ``self.paths`` is built.

        Robustness motivation (see ``design/ras2025_format_analysis.md``
        and ``design/missing_temporal_fallback.md``): real-world models
        span multiple RAS schema versions and output configurations.
        Treating the case-study workflow as the only configuration
        produces a reader that crashes deep in init on representative
        real-world plans. Pre-flight detection shifts those crashes to
        clean, actionable errors at launch time.
        """
        with h5py.File(self.ras_hdf_path, 'r') as infile:
            # ---- 1. RAS 2025 layout detection ----
            # The v2025 fingerprint: ``Geometry/2D Flow Areas/Mesh/``
            # exists as a group AND has a ``Version`` attribute >= 2.0
            # (per Muncie 2025 inspection 2026-05-23, the value is
            # literally the string '2.0'). The v5/6/7+ layout has no
            # such ``Mesh`` group; project names live as siblings under
            # ``Geometry/2D Flow Areas/<name>``.
            v2025_path = 'Geometry/2D Flow Areas/Mesh'
            if v2025_path in infile:
                mesh_group = infile[v2025_path]
                version = mesh_group.attrs.get('Version', None)
                if version is not None:
                    if isinstance(version, (bytes, np.bytes_)):
                        version = version.decode('utf-8', errors='replace')
                    elif isinstance(version, np.ndarray):
                        version = str(version.flat[0])
                    raise NotImplementedError(
                        f"RAS 2025 HDF layout detected (Mesh Version "
                        f"{version!r}, ``{v2025_path}/`` present). The "
                        f"current reader targets the RAS v5/6/7+ schema. "
                        f"The fallback kernels (``utilities._compute_"
                        f"cell_volumes`` / ``_compute_face_areas``) are "
                        f"forward-compatible with v2025; the v2025 work "
                        f"is in the reader's translation layer (deriving "
                        f"Cells Surface Area + Faces NormalUnitVector "
                        f"and Length from the consolidated ``Mesh/...`` "
                        f"tables). Not yet implemented. See "
                        f"``design/ras2025_format_analysis.md`` and "
                        f"``design/ras2025_reader_design_spec.md``."
                    )

            # ---- 2. Required temporal outputs presence check ----
            # Read project_name directly (the same way
            # __build_static_from_hdf does later) so we can template
            # the Cell Volume / Face Flow paths without depending on
            # the full __set_internal_paths pass.
            #
            # If ``Geometry/2D Flow Areas/Attributes`` is absent on a
            # file that ALSO lacks the v2025 ``Geometry/2D Flow Areas/
            # Mesh`` marker, the file is most likely a RAS 2025 plan or
            # results sub-file (separate from the geometry sub-file).
            # Raise NotImplementedError with the v2025 pointer rather
            # than a generic schema-error.
            try:
                project_name = infile[
                    'Geometry/2D Flow Areas/Attributes'
                ][()][0][0].decode('UTF-8')
            except (KeyError, IndexError, AttributeError) as e:
                # If we couldn't find Attributes, this is either v2025
                # (which splits geometry into a separate sub-file -- the
                # plan/results sub-file we may have been handed lacks
                # the geometry tree entirely) or an unrecognized
                # schema. Either way, point at the v2025 design spec
                # because that's the most likely cause for any modern
                # RAS plan that fails this check.
                raise NotImplementedError(
                    f"This HDF lacks the legacy ``Geometry/2D Flow "
                    f"Areas/Attributes`` dataset that the v5/6/7+ "
                    f"reader requires ({e!r}). Likely causes:\n"
                    f"  (1) RAS 2025 splits the project across multiple "
                    f"      sub-files (Geometries/, Plans/, Results/, "
                    f"      Boundary Conditions/, etc.). The current "
                    f"      reader cannot consume the v2025 layout. "
                    f"      See ``design/ras2025_format_analysis.md`` "
                    f"      and ``design/ras2025_reader_design_spec.md``.\n"
                    f"  (2) Unrecognised HDF variant. Pass a RAS 5.0.7+ "
                    f"      ``.pXX.hdf`` plan file (combined geometry + "
                    f"      results)."
                ) from e

            ts_base = (
                'Results/Unsteady/Output/Output Blocks/Base Output/'
                f'Unsteady Time Series/2D Flow Areas/{project_name}'
            )
            volume_path = f'{ts_base}/Cell Volume'
            face_flow_path = f'{ts_base}/Face Flow'

            if volume_path not in infile:
                if not self._allow_cell_volume_fallback:
                    raise KeyError(
                        f"Required temporal variable 'Cell Volume' is "
                        f"absent from the HEC-RAS HDF (expected at "
                        f"'{volume_path}'). The RAS plan that produced "
                        f"this HDF did not write 'Cell Volume' to its "
                        f"output set.\n\n"
                        f"Two ways forward:\n"
                        f"  (1) Re-run the RAS plan with 'Cell Volume' "
                        f"enabled (RAS Mapper -> Plan -> Output Options).\n"
                        f"  (2) Opt into the geometry-derived fallback "
                        f"(V from WSE via the per-cell volume-elevation "
                        f"lookup): set "
                        f"``model.allow_cell_volume_fallback: true`` in "
                        f"the YAML config, or pass "
                        f"``--allow-cell-volume-fallback`` on the runner "
                        f"CLI. See ``design/missing_temporal_fallback.md`` "
                        f"for the fidelity validation."
                    )

            if face_flow_path not in infile:
                if not self._allow_face_flow_fallback:
                    raise KeyError(
                        f"Required temporal variable 'Face Flow' is "
                        f"absent from the HEC-RAS HDF (expected at "
                        f"'{face_flow_path}'). The RAS plan that produced "
                        f"this HDF did not write 'Face Flow' to its "
                        f"output set.\n\n"
                        f"Two ways forward:\n"
                        f"  (1) Re-run the RAS plan with 'Face Flow' "
                        f"enabled (RAS Mapper -> Plan -> Output Options).\n"
                        f"  (2) Opt into the geometry-derived fallback "
                        f"(face_flow = wetted_face_area * edge_velocity, "
                        f"signed, with wetted face area from the per-face "
                        f"area-elevation lookup): set "
                        f"``model.allow_face_flow_fallback: true`` in the "
                        f"YAML config, or pass "
                        f"``--allow-face-flow-fallback`` on the runner "
                        f"CLI. NOTE: this is a post-hoc reconstruction "
                        f"and does NOT embed the full SWE momentum "
                        f"balance RAS used. Agreement with RAS-native "
                        f"Face Flow is typically a few percent for "
                        f"well-resolved flow, larger at wet/dry edges. "
                        f"See ``design/missing_temporal_fallback.md``."
                    )

    def __probe_temporal_fallbacks(self) -> None:
        """Preload lookup-table DataFrames + set fallback-active flags.

        Pre-flight (``__preflight_schema_check``) has already raised
        cleanly if either Cell Volume or Face Flow is absent AND the
        corresponding opt-in flag is False. By the time this runs, the
        only remaining cases per variable are:

        1. Dataset present in the HDF -> no-op.
        2. Dataset absent, opt-in flag True -> set fallback flag,
           preload the lookup-table DataFrames as instance attrs, emit
           one-shot warning naming the reconstruction formula.

        Called once per ``__init__`` after the static state is loaded
        (so ``self.paths`` is populated and the lookup-table paths can
        be resolved).
        """
        with h5py.File(self.ras_hdf_path, 'r') as infile:
            volume_present = self.paths[VOLUME] in infile
            face_flow_present = self.paths[FLOW_ACROSS_FACE] in infile

            # --- Cell Volume ---------------------------------------------
            # Pre-flight has already raised if absent + flag False.
            # Reaching here with not-present means the flag is True;
            # load lookup tables and emit the activation warning.
            if not volume_present:
                self._volume_fallback_active = True
                self._volume_elev_info_df = _hdf_to_dataframe(
                    infile[self.paths['volume elevation info']]
                )
                self._volume_elev_values_df = _hdf_to_dataframe(
                    infile[self.paths['volume_elevation_values']]
                )
                warnings.warn(
                    "Cell Volume fallback ACTIVE. This run is "
                    "synthesizing per-cell volumes from the RAS "
                    "volume-elevation lookup table and Water Surface "
                    "at each timestep. Agreement with RAS-native Cell "
                    "Volume output is typically sub-0.1% for well-"
                    "wetted cells; see design/missing_temporal_fallback.md.",
                    UserWarning,
                    stacklevel=2,
                )

            # --- Face Flow ----------------------------------------------
            # Same: pre-flight has already handled the absent + flag-False
            # case. Here we just preload + emit activation warning.
            if not face_flow_present:
                self._face_flow_fallback_active = True
                self._face_area_elev_info_df = _hdf_to_dataframe(
                    infile[self.paths['area_elevation_info']]
                )
                self._face_area_elev_values_df = _hdf_to_dataframe(
                    infile[self.paths['area_elevation_values']]
                )
                self._face_length_df = _hdf_to_dataframe(
                    infile[self.paths['normalunitvector_length']]
                )
                self._face_cell_indexes_df = _hdf_to_dataframe(
                    infile[self.paths[EDGE_FACE_CONNECTIVITY]]
                )
                warnings.warn(
                    "Face Flow fallback ACTIVE. This run is synthesizing "
                    "per-edge face flow as signed (face_area * "
                    "edge_velocity), with face area interpolated from "
                    "the RAS area-elevation lookup. Agreement with RAS-"
                    "native Face Flow depends on flow regime; validate "
                    "results before drawing transport conclusions. See "
                    "design/missing_temporal_fallback.md.",
                    UserWarning,
                    stacklevel=2,
                )

    def __build_static_from_hdf(self) -> None:
        """Walk the HEC-RAS HDF and populate all static state.

        Sets project_name, paths, gate_names, all_datetimes, the static
        mesh (coords + topology + static vars), boundary_data, the
        volume-elevation lookup, nface/nedge, and real_cell_count. This
        is the expensive path the on-disk cache exists to skip.
        """
        with h5py.File(self.ras_hdf_path, 'r') as infile:
            # set-up steps
            self.project_name = infile[
                'Geometry/2D Flow Areas/Attributes'
            ][()][0][0].decode('UTF-8')
            self.paths = self.__set_internal_paths()
            self.gate_names = self.__identify_gates(infile)
            self.__parse_datetimes(infile)

            # populate mesh
            self.__define_spatial_coordinates(infile)
            self.__define_topology(infile)
            self.__define_boundary_hydrodynamics(infile)
            self.__read_static_variables(infile)

            # populate lookup table
            self.volume_elevation_lookup = self.__create_lookup_xarray(infile)

            # gather additional data
            self.real_cell_count = self.mesh[EDGE_FACE_CONNECTIVITY].T[0].values.max() + 1

    def __static_payload(self) -> Dict[str, Any]:
        """Assemble the cacheable static state.

        Everything __build_static_from_hdf produces, so a hit can fully
        reconstruct the data source without touching the HDF for
        geometry. Temporal arrays are deliberately excluded -- they are
        re-read fresh from the HDF on every run.
        """
        return {
            'schema_version': CACHE_SCHEMA_VERSION,
            'cwr_version': clearwater_riverine.__version__,
            'mesh': self.mesh,
            'boundary_data': self.boundary_data,
            'volume_elevation_lookup': self.volume_elevation_lookup,
            'nface': self.nface,
            'nedge': self.nedge,
            'real_cell_count': self.real_cell_count,
            'project_name': self.project_name,
            'paths': self.paths,
            'gate_names': self.gate_names,
            'all_datetimes': self.all_datetimes,
            # Phase I-2 (2026-05-21): round-trip Internal Cells data
            # so the cache hit path matches the cache miss path
            # exactly. ``None`` on HDFs that have no Internal BCs
            # (the common case).
            'internal_cells': self.internal_cells,
        }

    def __rehydrate_static(self, payload: Dict[str, Any]) -> None:
        """Restore static state from a cached payload (hit path).

        A missing key raises KeyError, which the __init__ hit path
        catches and treats as a stale cache -> rebuild.
        """
        self.mesh = payload['mesh']
        self.boundary_data = payload['boundary_data']
        self.volume_elevation_lookup = payload['volume_elevation_lookup']
        self.nface = payload['nface']
        self.nedge = payload['nedge']
        self.real_cell_count = payload['real_cell_count']
        self.project_name = payload['project_name']
        self.paths = payload['paths']
        self.gate_names = payload['gate_names']
        self.all_datetimes = payload['all_datetimes']
        # Phase I-2 (2026-05-21): cache-miss path defaults to None
        # when the key is absent so older cache payloads (pre-I-2)
        # rehydrate cleanly without forcing a full rebuild.
        self.internal_cells = payload.get('internal_cells', None)


    def _optional_temporal_variables(self) -> dict:
        """Return the diffusion-dispatch optional temporal variables.

        Phase I-1 (2026-05-21): each entry is
        ``(variable_name, space_dim)``. The reader consumes this
        lazily and skips entries whose HDF path is absent, so adding
        a new optional variable here does not require touching the
        read loop and existing HDFs without these datasets continue
        to load.
        """
        return {
            FACE_VEL_X: NFACE,
            FACE_VEL_Y: NFACE,
            EDDY_VISCOSITY: NEDGE,
            CELL_EDDY_VISCOSITY_X: NFACE,
            CELL_EDDY_VISCOSITY_Y: NFACE,
        }


    def __identify_gates(
        self,
        infile,
    ):
        """Assesses if gate structures exist in HEC RAS model."""
        try:
            gate_names = list(infile[self.paths['gate_path']].keys())
        except KeyError:
            gate_names = None
        return gate_names


    def __set_internal_paths(self):
        """ Define HDF paths to relevant data"""
        return {
            NODE_X: f'Geometry/2D Flow Areas/{self.project_name}/FacePoints Coordinate',
            NODE_Y: f'Geometry/2D Flow Areas/{self.project_name}/FacePoints Coordinate',
            TIME: 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
            FACE_NODES: f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes',
            EDGE_NODES: f'Geometry/2D Flow Areas/{self.project_name}/Faces FacePoint Indexes',
            EDGE_FACE_CONNECTIVITY: f'Geometry/2D Flow Areas/{self.project_name}/Faces Cell Indexes',
            FACE_X: f'Geometry/2D Flow Areas/{self.project_name}/Cells Center Coordinate',
            FACE_Y: f'Geometry/2D Flow Areas/{self.project_name}/Cells Center Coordinate',
            FACE_SURFACE_AREA: f'Geometry/2D Flow Areas/{self.project_name}/Cells Surface Area',
            EDGE_VELOCITY: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Face Velocity',
            EDGE_LENGTH: f'Geometry/2D Flow Areas/{self.project_name}/Faces NormalUnitVector and Length',
            WATER_SURFACE_ELEVATION: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Water Surface',
            FLOW_ACROSS_FACE: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Face Flow',
            VOLUME: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Volume',
            FACE_HYD_DEPTH: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Hydraulic Depth',
            FACE_VEL_X: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Velocity - Velocity X',
            FACE_VEL_Y: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Velocity - Velocity Y',
            # Phase I-1 (2026-05-21): paths for the diffusion-dispatch
            # consumers. MANNINGS_N is a static geometry attribute;
            # EDDY_VISCOSITY (per-edge, per-time) and the CELL_EDDY_*
            # X/Y components (per-cell, per-time) are optional
            # temporal outputs. All are conditional: the reader checks
            # ``hdf_path in infile`` before adding to the temporal /
            # static read loop, so HDFs that ship only the minimal
            # output set continue to work.
            MANNINGS_N: f"Geometry/2D Flow Areas/{self.project_name}/Cells Center Manning's n",
            EDDY_VISCOSITY: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Eddy Viscosity - Eddy Viscosity',
            CELL_EDDY_VISCOSITY_X: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Eddy Viscosity - X',
            CELL_EDDY_VISCOSITY_Y: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Eddy Viscosity - Y',
            'project_name': 'Geometry/2D Flow Areas/Attributes',
            'binary_time_stamps': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
            'volume elevation info': f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Info',
            'volume_elevation_values': f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Values',
            'area_elevation_info': f'Geometry/2D Flow Areas/{self.project_name}/Faces Area Elevation Info',
            'area_elevation_values': f'Geometry/2D Flow Areas/{self.project_name}/Faces Area Elevation Values',
            'normalunitvector_length': f'Geometry/2D Flow Areas/{self.project_name}/Faces NormalUnitVector and Length',
            'boundary_condition_external_faces': 'Geometry/Boundary Condition Lines/External Faces',
            # Phase I-2 (2026-05-21): Internal Cells path. Per-cell
            # mass-injection lookup for BC lines drawn through the
            # interior of the 2D mesh (rather than the perimeter).
            # Optional; absent on most subset HDFs (the Santiam-Salem
            # subset's BCs were flattened from Internal -> External-
            # face representations by the case-study subset extractor).
            'boundary_condition_internal_cells': 'Geometry/Boundary Condition Lines/Internal Cells',
            'boundary_condition_attributes': 'Geometry/Boundary Condition Lines/Attributes/',
            # Phase J+1 (2026-05-23): the BC results group has lived
            # at three different paths over the lifetime of the HEC-RAS
            # HDF schema; ClearWater encounters at least two of them in
            # the wild and must resolve which is present at read time.
            #
            # Trajectory (so the next person who hits a new HDF lands
            # at the right place):
            #
            #   Pre-5.0.7 / legacy  ->  top-level
            #     ``.../Unsteady Time Series/Boundary Conditions``
            #     (used by the Santiam-Salem synthetic subset and the
            #     original ClearWater test fixtures)
            #
            #   RAS 5.0.7+ / modern  ->  per-2D-area
            #     ``.../Unsteady Time Series/2D Flow Areas/<name>/Boundary Conditions``
            #     (used by the Corvallis_Santiam HDF and most real-world
            #     plans produced by recent HEC-RAS builds)
            #
            #   RAS 2025+            ->  reorganised again, see
            #     ``design/ras2025_format_analysis.md`` §2.7. The current
            #     reader is fail-loud on v2025 (see
            #     ``__preflight_schema_check``); a separate v2025 reader
            #     is the planned path forward.
            #
            # The two string entries below are the legacy and modern
            # paths. The actual path used at read time is resolved by
            # ``__resolve_boundary_condition_fixes_path`` (see below),
            # which detects the layout once per file and overwrites
            # ``boundary_condition_fixes`` so both
            # ``__define_boundary_hydrodynamics`` and
            # ``__fix_boundary_data`` see a consistent value.
            'boundary_condition_fixes': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Boundary Conditions',
            'boundary_condition_fixes_per_area': (
                'Results/Unsteady/Output/Output Blocks/Base Output/'
                'Unsteady Time Series/2D Flow Areas/'
                f'{self.project_name}/Boundary Conditions'
            ),
            VOLUME_ELEVATION_INFO: f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Info',
            VOLUME_ELEVATION_VALUES: f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Values',
            'gate_path': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/SA 2D Area Conn',
        }


    def __parse_datetimes(
        self, 
        infile: h5py.File,
    ):
        """Date handling."""
        # time
        time_stamps_binary = infile[self.paths['binary_time_stamps']][()]

        # pandas is working faster than numpy for binary conversion
        ## TODO: figure out if it's faster to store all timesteps in memory and subset on each read
        ## OR if it's faster to do this binary conversion for each new chunk read.
        time_stamps = pd.Series(time_stamps_binary).str.decode('utf8')
        self.all_datetimes = pd.to_datetime(time_stamps, format='%d%b%Y %H:%M:%S')


    def __define_spatial_coordinates(
        self,
        infile: h5py.File
    ):
        """Populate Coordinates and Dimensions"""
        # x-coordinates
        self.mesh = self.mesh.assign_coords(
            node_x=xr.DataArray(
                data=infile[self.paths[NODE_X]][()].T[0],
                dims=('node',),
            )
        )
        # y-coordinates
        self.mesh = self.mesh.assign_coords(
            node_y=xr.DataArray(
                data=infile[self.paths[NODE_X]][()].T[1],
                dims=('node',),
            )
        )    


    def __define_topology(
        self,
        infile: h5py.File,
    ):
        """Define mesh topology """
        self.mesh[FACE_NODES] = xr.DataArray(
            data=infile[
                f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes'
            ][()],
            coords={
                "face_x": ('nface', infile[self.paths[FACE_X]][()].T[0]),
                "face_y": ('nface', infile[self.paths[FACE_Y]][()].T[1]),
            },
            dims=('nface', 'nmax_face'),
            attrs={
                'cf_role': 'face_node_connectivity',
                'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                'start_index': 0,
                '_FillValue': -1
            }
        )
        self.mesh[EDGE_NODES] = xr.DataArray(
            data=infile[self.paths[EDGE_NODES]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_node_connectivity',
                'long_name': 'Vertex nodes of mesh edges',
                'start_index': 0
            })
        self.mesh[EDGE_FACE_CONNECTIVITY] = xr.DataArray(
            data=infile[self.paths[EDGE_FACE_CONNECTIVITY]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_face_connectivity',
                'long_name': 'neighbor faces for edges',
                'start_index': 0
            })
        
        if self.gate_names is not None:
            connectivity_list = []
            for g in self.gate_names:
                headwater_cells = infile[
                    f"{self.paths['gate_path']}/{g}/HW TW Segments/Headwater Cells"][()].astype(int)
                tailwater_cells = infile[
                    f"{self.paths['gate_path']}/{g}/HW TW Segments/Tailwater Cells"][()].astype(int)
                gate_connectivity = np.stack((tailwater_cells, headwater_cells), axis=1)
                connectivity_list.append(gate_connectivity)
            
            connectivity_array = np.concatenate(connectivity_list, axis=0)

            self.mesh[GATE_CONNECTIVITY] = xr.DataArray(
                data=connectivity_array,
                dims=[GATE_CONNECTIVITY, '2'],
                attrs={
                    'long_name': 'cells connected by gates'
                }
            )

    def read(self, parameter_name:str) -> DataArrayVariable:
        return DataArrayVariable(self.__read(parameter_name))

    def __read(self, parameter_name: str):
        if parameter_name in self.mesh.data_vars:
            return self.mesh[parameter_name]
        else:
            self.__subset_datetimes(
                self.start_datetime,
                self.end_datetime,
            )
            self.__update_time_coordinate()
            self.__update_mesh()
            return self.mesh[parameter_name]
        
    def read_chunk(
        self,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> DataArrayVariable:
        return DataArrayVariable(
            self.__read_chunk(parameter_name, start_time, end_time),
            space_dimension=self.temporal_variables[parameter_name]
        )        

    def __read_chunk(
        self,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime
    ):
        if (
            "time" in self.mesh.coords
            and (self.mesh.time[0] == start_time)
            and (self.mesh.time[-1] == end_time)
            and (parameter_name in self.mesh.data_vars)
        ):
            return self.mesh[parameter_name]
        else:
            self.__subset_datetimes(
                start_time,
                end_time
            )
            self.__update_time_coordinate()
            self.__update_mesh()
            return self.mesh[parameter_name]
        
    
    def __update_mesh(
            self,
    ):
        with h5py.File(self.ras_hdf_path, 'r') as infile:
            self.__read_temporal_variables(infile)

    # def read() --> as variable DataArray variable

    def __subset_datetimes(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ):
        
        subset_dates = self.all_datetimes[
            (self.all_datetimes >= start_datetime) & (self.all_datetimes <= end_datetime)
        ]
        subset_indices = subset_dates.index.intersection(
            self.all_datetimes.index
        )
        self.datetime_range_indices: Tuple[int, int] = (
            subset_indices[0],
            subset_indices[-1] + 1
        )

        if self.datetime_range_indices != (None, None):
            self.datetime_subset = self.all_datetimes[
                    self.datetime_range_indices[0]:
                    self.datetime_range_indices[1]
                    ]
        else:
            self.datetime_subset = self.all_datetimes


    def __update_time_coordinate(
        self,
    ):
        # Drop any time-dimensioned data vars from a previous chunk read
        # first; otherwise assign_coords(time=new) raises when the new
        # window length differs from the previous one (Phase-C C4 / B4:
        # the final chunk of an uneven (end-start) / chunk_size split is
        # shorter than the others). Static vars without a 'time' dim are
        # preserved.
        time_vars = [
            v for v in self.mesh.data_vars
            if 'time' in self.mesh[v].dims
        ]
        if time_vars:
            self.mesh = self.mesh.drop_vars(time_vars)
        self.mesh = self.mesh.assign_coords(
            time=xr.DataArray(
                data=self.datetime_subset,
                dims=('time',),
            )
        )

    
    def __read_static_variables(
        self,
        infile: h5py.File,
    ):
        ## TODO: is this needed?
        self.mesh[FACE_SURFACE_AREA] = _hdf_to_xarray(
            infile[self.paths[FACE_SURFACE_AREA]],
            (NFACE)
        )
        self.nface = len(self.mesh[NFACE])

        self.mesh[EDGE_LENGTH] = _hdf_to_xarray(
            infile[self.paths[EDGE_LENGTH]][:, 2],
            ('nedge'),
        )
        self.nedge = len(self.mesh[NEDGE])

        # Phase I-1 (2026-05-21): optional static read for Manning's n.
        # Required by the Elder diffusion method; absent from RAS HDFs
        # that ship only the minimal output set. Skipped silently
        # when absent; the diffusion dispatcher raises with a clear
        # error if the user requests ``method=elder`` without it.
        if self.paths[MANNINGS_N] in infile:
            self.mesh[MANNINGS_N] = _hdf_to_xarray(
                infile[self.paths[MANNINGS_N]],
                (NFACE,),
            )

    def __synthesize_volume_for_chunk(self) -> xr.DataArray:
        """Compute per-cell volume for the current chunk from WSE.

        The WSE slab is already on ``self.mesh`` (read earlier in
        ``__read_temporal_variables``). The lookup-table DataFrames were
        cached at init by ``__probe_temporal_fallbacks``.

        Returns an ``xr.DataArray`` with dims ``(time, nface)`` matching
        the shape of the read-from-disk variant.
        """
        wse_arr = self.mesh[WATER_SURFACE_ELEVATION].values
        cell_area_arr = self.mesh[FACE_SURFACE_AREA].values
        info = self._volume_elev_info_df
        values = self._volume_elev_values_df
        cell_volumes = _compute_cell_volumes(
            wse_arr.astype(np.float64, copy=False),
            cell_area_arr.astype(np.float64, copy=False),
            info['Starting Index'].values.astype(np.int64),
            info['Count'].values.astype(np.int64),
            values['Elevation'].values.astype(np.float64),
            values['Volume'].values.astype(np.float64),
        )
        return xr.DataArray(
            data=cell_volumes,
            dims=('time', NFACE),
            # Phase J+1 (2026-05-23): attach the chunk's time coord
            # explicitly. Without it, the returned DataArray has integer
            # indices on the time dim and downstream registry lookups
            # via ``get_at_time(VOLUME, Timestamp(...))`` fail with
            # ``KeyError: "not all values found in index 'time'"``. The
            # mesh's existing time coord is what the rest of the
            # pipeline (WET_MASK refresh, ADVECTION_COEFFICIENT compute,
            # LHS lookups at current_time+time_step) expects.
            coords={'time': self.mesh.time.values},
            attrs={
                'Units': 'ft3 or m3 (RAS-native; matches Water Surface units)',
                'long_name': 'Cell Volume (synthesized from WSE + lookup table)',
                'fallback_active': 1,
            },
        )

    def __synthesize_face_flow_for_chunk(self) -> xr.DataArray:
        """Compute per-edge face flow for the current chunk.

        Uses ``face_flow = signed(face_area * edge_velocity)`` where the
        face area comes from the per-face area-elevation lookup and the
        edge velocity is RAS's direct output. This matches the SIGN
        convention the canonical LHS expects (positive = leaving
        ``edges_face1``), unlike the streaming fork's ``abs(...)`` form.
        """
        wse_arr = self.mesh[WATER_SURFACE_ELEVATION].values
        edge_vel_arr = self.mesh[EDGE_VELOCITY].values
        info = self._face_area_elev_info_df
        values = self._face_area_elev_values_df
        lengths = self._face_length_df['Face Length'].values.astype(np.float64)
        # ``Cell 0`` is the cell on side 1 of each edge -- the WSE source
        # used by the per-face area-elevation lookup. Matches the
        # streaming fork's convention.
        cell0 = self._face_cell_indexes_df['Cell 0'].values.astype(np.int64)
        face_areas = _compute_face_areas(
            wse_arr.astype(np.float64, copy=False),
            lengths,
            cell0,
            info['Starting Index'].values.astype(np.int64),
            info['Count'].values.astype(np.int64),
            values['Z'].values.astype(np.float64),
            values['Area'].values.astype(np.float64),
        )
        # Signed reconstruction: positive = leaving edges_face1.
        face_flow = face_areas * edge_vel_arr
        return xr.DataArray(
            data=face_flow,
            dims=('time', NEDGE),
            # Phase J+1 (2026-05-23): same time-coord rationale as
            # __synthesize_volume_for_chunk above.
            coords={'time': self.mesh.time.values},
            attrs={
                'Units': 'ft3/s or m3/s (RAS-native units)',
                'long_name': 'Face Flow (synthesized from face_area * edge_velocity)',
                'fallback_active': 1,
            },
        )

    def __read_temporal_variables(
        self,
        infile: h5py.File,
    ):
        # Phase J+1 (Corvallis 2026-05-23): defer VOLUME / FLOW_ACROSS_FACE
        # to a second pass so WATER_SURFACE_ELEVATION is on the mesh
        # before any fallback synthesis runs. The fallback kernels need
        # this chunk's WSE slab, and the lookup tables they consume were
        # cached at init by ``__probe_temporal_fallbacks``.
        deferred = (VOLUME, FLOW_ACROSS_FACE)
        for variable in self.temporal_variables.keys():
            if variable in deferred:
                continue
            hdf_path = self.paths[variable]
            if hdf_path not in infile:
                raise KeyError(
                    f"Required temporal variable '{variable}' is absent from "
                    f"the HEC-RAS HDF (expected dataset: '{hdf_path}'). This "
                    f"RAS run did not output it; no geometry-derived fallback "
                    f"is defined for this variable."
                )
            self.mesh[variable] = _hdf_to_xarray(
                infile[hdf_path],
                ('time', self.temporal_variables[variable]),
                time_constraint=self.datetime_range_indices,
            )

        # ---- VOLUME ----------------------------------------------------
        # Direct read when the dataset is present, synthesis when the
        # opt-in fallback was activated by ``__probe_temporal_fallbacks``.
        # The probe already raised on the absent-without-opt-in case,
        # so an absent dataset here implies the fallback is active.
        vol_path = self.paths[VOLUME]
        if vol_path in infile:
            self.mesh[VOLUME] = _hdf_to_xarray(
                infile[vol_path],
                ('time', self.temporal_variables[VOLUME]),
                time_constraint=self.datetime_range_indices,
            )
        elif self._volume_fallback_active:
            self.mesh[VOLUME] = self.__synthesize_volume_for_chunk()
        else:
            # Defensive: probe should have caught this. Re-raise here so
            # a bypassed probe still fails loud rather than silently
            # producing a mesh missing VOLUME.
            raise KeyError(
                f"Required temporal variable 'volume' is absent and the "
                f"Cell Volume fallback is not active. Probe step bypassed."
            )

        # ---- FLOW_ACROSS_FACE -----------------------------------------
        ff_path = self.paths[FLOW_ACROSS_FACE]
        if ff_path in infile:
            self.mesh[FLOW_ACROSS_FACE] = _hdf_to_xarray(
                infile[ff_path],
                ('time', self.temporal_variables[FLOW_ACROSS_FACE]),
                time_constraint=self.datetime_range_indices,
            )
        elif self._face_flow_fallback_active:
            self.mesh[FLOW_ACROSS_FACE] = self.__synthesize_face_flow_for_chunk()
        else:
            raise KeyError(
                f"Required temporal variable 'face_flow' is absent and the "
                f"Face Flow fallback is not active. Probe step bypassed."
            )

        # Phase I-1 (2026-05-21): optional temporal reads for the
        # diffusion-dispatch consumers. Each is read only when the
        # HDF dataset exists; ``self.optional_temporal_variables`` is
        # a (name, space-dim) iterable shared with the chunked-read
        # path. Skipped silently when absent; the diffusion
        # dispatcher raises ``NotImplementedError`` with a clear
        # message at construction time if a method that needs the
        # variable is requested but the variable isn't in the
        # registry.
        for variable, space_dim in self._optional_temporal_variables().items():
            hdf_path = self.paths.get(variable)
            if hdf_path is None or hdf_path not in infile:
                continue
            self.mesh[variable] = _hdf_to_xarray(
                infile[hdf_path],
                ('time', space_dim),
                time_constraint=self.datetime_range_indices,
            )

        # add gate flows
        if self. gate_names is not None:
            flow_list = []
            for g in self.gate_names:
                gate_flow = infile[
                    f"{self.paths['gate_path']}/{g}/HW TW Segments/Flow"][()][:,0:-1] * -1
                flow_list.append(gate_flow)
            
            # flow_array = np.stack(flow_list, axis=1)
            flow_array = np.concatenate(flow_list, axis=1) 
            flow_array = flow_array[self.datetime_range_indices[0]: self.datetime_range_indices[1]]

            self.mesh[GATE_FLOW] = xr.DataArray(
                data=flow_array,
                dims=["time", GATE_FLOW],
                attrs={
                    'long_name': 'flow in cells'
                }
            )

    def __create_lookup_xarray(
        self,
        infile: h5py.File
    ):
        """Create volume elevation lookup xarray dataset."""
        volume_elevation_info_df = _hdf_to_dataframe(
            infile[self.paths[VOLUME_ELEVATION_INFO]]
            )
        volume_elevation_vals_df = _hdf_to_dataframe(
            infile[self.paths[VOLUME_ELEVATION_VALUES]]
        )
        # Define cells associated with each lookup value
        volume_elevation_vals_df['Cell'] = np.concatenate(
            [
                np.full(count, cell)
                for cell, count in zip(
                    volume_elevation_info_df.index,
                    volume_elevation_info_df['Count']
                )
            ]
        )

        # Create lookup dataset
        lookup_datasets = []

        for cell in volume_elevation_vals_df['Cell'].unique():
            cell_df = self.__create_cell_lookup_table(
                cell,
                volume_elevation_vals_df,
                infile,
            )
            cell_df = cell_df.rename(
                columns = {
                    "Elevation": LOOKUP_ELEVATION,
                    "Volume": LOOKUP_VOLUME,
                    "Wetted Surface Area": LOOKUP_WETTED_SURFACE_AREA,
                }
            )
            ds_cell = xr.Dataset.from_dataframe(cell_df)
            ds_cell = ds_cell.expand_dims({"nface": [cell]})
            lookup_datasets.append(ds_cell)

        return xr.concat(lookup_datasets, dim="nface", join="outer")

    def __create_cell_lookup_table(
        self,
        cell_no: int,
        df: pd.DataFrame,
        infile: h5py.File
    ) -> pd.DataFrame:
        """Create volume-elevation lookup table for each cell."""
        # Filter for single cell
        df_temp = df[df['Cell'] == cell_no]
        test_df = df_temp.copy().reset_index(drop=True)
        cell_surface_area = _hdf_to_dataframe(
            infile[self.paths[FACE_SURFACE_AREA]]
            )

        # Add row for flat cells (i.e., only one entry in lookup)
        # Create arbitrarily larger value
        increment_val = 0.01
        if len(test_df) == 1:
            # Preemptively add second row before any calculations
            new_row = test_df.iloc[0].copy()
            new_row['Elevation'] += increment_val
            new_row['Volume'] += increment_val
            test_df = pd.concat(
                [test_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

        # Compute differences in elevation and volume between adjacent rows
        # (i.e., vertical layers in the cell)
        test_df['Delta Elev'] = test_df['Elevation'].diff()
        test_df['Delta Volume'] = test_df['Volume'].diff()

        # Calculate the wetted surface area based on the volume and depth
        test_df['Surface Area'] = \
            test_df['Delta Volume'] / test_df['Delta Elev']

        # Average surface area between two elevation bands
        # Approximates wetted surface area between two elevatiosn
        test_df['Wetted Surface Area'] = \
            (test_df['Surface Area'] + test_df['Surface Area'].shift(-1)) / 2

        # Get maximum volume
        max_index = test_df['Volume'].idxmax()

        # Set edge cases (first and last slice)
        # Compare with total surface area for the cell as a whole
        cell_table = cell_surface_area[cell_surface_area.index == cell_no]
        input_value = cell_table['Surface Area'].values[0]
        # Set wetted surface area at the first row to 0 (i.e., first slice)
        test_df.at[max_index, 'Wetted Surface Area'] = input_value
        test_df.at[0, 'Wetted Surface Area'] = 0
        return test_df

    def __resolve_boundary_condition_fixes_path(
        self,
        infile: h5py.File,
    ) -> str:
        """Resolve which HDF group holds the per-BC-line results.

        Phase J+1 (Corvallis 2026-05-23): HEC-RAS HDFs store the
        ``Boundary Conditions`` results group in one of two locations
        depending on the RAS version that produced the file:

          legacy : Results/Unsteady/Output/Output Blocks/Base Output/
                   Unsteady Time Series/Boundary Conditions
          modern : Results/Unsteady/Output/Output Blocks/Base Output/
                   Unsteady Time Series/2D Flow Areas/<name>/Boundary Conditions

        The legacy layout is used by the Santiam-Salem synthetic
        subset HDF and the package's older test fixtures. The modern
        layout is what RAS 5.0.7+ writes natively (e.g., the
        Corvallis_Santiam.p01.hdf produced by the user's RAS 5.0.7
        run on the 933,827-cell Willamette mesh).

        We try the legacy path first to preserve byte-identical
        behaviour on existing case studies, then fall back to the
        per-2D-area path. The resolved value is cached on
        ``self.paths['boundary_condition_fixes']`` so the second
        consumer (``__fix_boundary_data``) reads the same group.
        """
        for key in ("boundary_condition_fixes",
                    "boundary_condition_fixes_per_area"):
            path = self.paths.get(key)
            if path is None:
                continue
            if path in infile:
                self.paths['boundary_condition_fixes'] = path
                return path
        raise KeyError(
            "Boundary Conditions results group not found in HDF at "
            f"either layout:\n"
            f"  legacy: {self.paths.get('boundary_condition_fixes')!r}\n"
            f"  modern: {self.paths.get('boundary_condition_fixes_per_area')!r}\n"
            "Inspect the HDF with h5dump or h5py to find the BC group "
            "and add the path to __parse_paths."
        )

    def __define_boundary_hydrodynamics(
        self,
        infile: h5py.File
    ):
        """Read necessary information on hydrodynamics."""
        # Phase J+1 (Corvallis 2026-05-23): resolve the BC results path
        # before any read of ``boundary_condition_fixes``. Updates
        # ``self.paths['boundary_condition_fixes']`` in place so the
        # downstream ``__fix_boundary_data`` consumer sees the same
        # value without needing its own lookup.
        self.__resolve_boundary_condition_fixes_path(infile)
        # Pull important boundary information from the HDF file.
        external_faces = pd.DataFrame(
            infile[self.paths['boundary_condition_external_faces']][()]
        )
        attributes = pd.DataFrame(
            infile[self.paths['boundary_condition_attributes']][()]
        )
        list_of_boundaries = list(
            infile[self.paths['boundary_condition_fixes']].keys()
        )

        # Decode data
        str_df = attributes.select_dtypes([object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            attributes[col] = str_df[col]
        boundary_attributes = attributes

        # Phase F T2-E + Phase I-2 (2026-05-21): detect Internal-type
        # BC lines and read the ``Internal Cells`` dataset when
        # present. The dataset is parsed onto ``self.internal_cells``
        # (a DataFrame with BC Line ID, Cell Index, Station Start,
        # Station End columns); a follow-up commit will join this with
        # the BC line's time-series and route the per-cell mass
        # injection through the T2-A point_sources infrastructure.
        # Until then, the warning is informative ("detected and
        # parsed, full routing pending") rather than alarming.
        self.internal_cells = None
        ic_path = self.paths.get('boundary_condition_internal_cells')
        if ic_path is not None and ic_path in infile:
            self.internal_cells = pd.DataFrame(infile[ic_path][()])

        if 'Type' in boundary_attributes.columns:
            internal_rows = boundary_attributes[
                boundary_attributes['Type'].astype(str).str.lower() == 'internal'
            ]
            if not internal_rows.empty:
                names = internal_rows.get('Name')
                names_str = (
                    ', '.join(str(n) for n in names.tolist())
                    if names is not None else '(unnamed)'
                )
                n_cells = (
                    len(self.internal_cells)
                    if self.internal_cells is not None else 0
                )
                warnings.warn(
                    f"Detected {len(internal_rows)} Internal-type boundary "
                    f"condition line(s) ({names_str}) with {n_cells} "
                    "Internal Cells entries. Phase I-2 (2026-05-21) reads "
                    "the Internal Cells dataset onto "
                    "``self.internal_cells``; the full per-cell mass "
                    "routing through the T2-A point_sources "
                    "infrastructure requires the BC line's per-cell flow "
                    "attribution time-series, which RAS does not write "
                    "to the standard output set. This is tracked as "
                    "Phase J work. For Phase I, if your case study "
                    "expects Internal-BC mass to enter the domain, "
                    "run the subset extractor that converts Internal "
                    "BCs into External-face representations (the "
                    "Santiam-Salem Sep 2008 deck used in Phase F "
                    "validation does this). See design/internal_bc_audit.md.",
                    UserWarning,
                    stacklevel=3,
                )

        # merge attributes and boundary condition data
        boundary_attributes['BC Line ID'] = boundary_attributes.index
        boundary_data = pd.merge(
            external_faces,
            boundary_attributes,
            on='BC Line ID',
            how='left'
        )

        # fix boundaries if needed
        boundary_data = self.__fix_boundary_hydrodynamics(
            boundary_data,
            infile
        )
        # store as boundary data
        self.boundary_data = (
            xr.Dataset.from_dataframe(boundary_data)
            # .set_coords(BOUNDARY_NAME)
            # .rename({BOUNDARY_NAME: 'RAS2D_TS_Name'})
        )

    def __fix_boundary_hydrodynamics(
        self,
        boundary_data: pd.DataFrame,
        infile: h5py.File
    ) -> pd.DataFrame:
        """
        Fixes a HEC-RAS bug in designating faces associated with
        boundary conditions.

        Phase J+1 (Corvallis 2026-05-23): the fix relies on a legacy
        per-BC-line ``<name> - Flow per Face`` dataset whose ``Faces``
        attribute the reader compares against
        ``Geometry/Boundary Condition Lines/External Faces``. Modern
        RAS HDFs (5.0.7+, e.g., the Corvallis_Santiam plan-13 output)
        do not write this dataset — they only emit per-BC-line ``Flow``
        and ``Stage`` arrays. When the dataset is absent for a given
        BC line, skip the bug fix for that line (the ``External Faces``
        data is taken as authoritative). When all BC lines lack the
        dataset, return the input unchanged so the caller proceeds
        with the original face mapping.
        """
        df_ls = []
        n_fix_applied = 0
        n_fix_skipped = 0

        # Identify correct boundary faces
        for boundary in boundary_data.Name.unique():
            fix_path = self.paths['boundary_condition_fixes']
            fpath = f"{fix_path}/{boundary} - Flow per Face"
            # Phase J+1 (Corvallis 2026-05-23): graceful skip when the
            # legacy per-face dataset is absent for this BC line.
            if fpath not in infile:
                n_fix_skipped += 1
                fixed_df = boundary_data[
                    boundary_data.Name == boundary
                ]
                df_ls.append(fixed_df)
                continue
            attrs = _parse_attributes(infile[fpath])
            boundary_faces_fix = attrs['Faces']
            boundary_faces_orig = boundary_data[
                (boundary_data.Name == boundary)]['Face Index']

            # compare with boundaries already identified
            # notify users if issues exist
            if set(boundary_faces_fix) != set(boundary_faces_orig):
                print(f'Extra boundary faces identified for {boundary}.')
                diff = set(boundary_faces_orig) - \
                    set(boundary_faces_fix)
                print(f'Removing erroneous boundaries {diff}.')

            # remove erroneous boundaries
            fixed_df = boundary_data[
                (boundary_data.Name == boundary) &
                (boundary_data['Face Index'].isin(boundary_faces_fix))
            ]
            df_ls.append(fixed_df)
            n_fix_applied += 1

        if n_fix_skipped > 0:
            warnings.warn(
                f"__fix_boundary_hydrodynamics: legacy '<name> - Flow per "
                f"Face' dataset absent for {n_fix_skipped} of "
                f"{n_fix_skipped + n_fix_applied} BC line(s); skipping the "
                "RAS-bug fix for those lines and treating "
                "Geometry/Boundary Condition Lines/External Faces as "
                "authoritative. Normal for RAS 5.0.7+ HDFs that do not "
                "emit the per-face sub-datasets.",
                stacklevel=3,
            )

        # remove any potential duplicates
        fixed_df_full = pd.concat(df_ls)
        fixed_df_full.drop(
            ['Station Start', 'Station End'],
            axis=1,
            inplace=True
        )
        fixed_df_full.drop_duplicates(inplace=True)

        return fixed_df_full


### SANDBOX:
### DERIVED VARIABLES
        # """Populates hydrodynamic data in UGRID-compliant xarray."""
        # mesh[EDGES_FACE1] = _hdf_to_xarray(
        #     mesh['edge_face_connectivity'].T[0],
        #     ('nedge'),
        #     attrs={'Units': ''}
        # )
        # mesh[EDGES_FACE2] = _hdf_to_xarray(
        #     mesh['edge_face_connectivity'].T[1],
        #     ('nedge'),
        #     attrs={'Units': ''}
        # )
        
        # nreal = mesh[EDGE_FACE_CONNECTIVITY].T[0].values.max()
        # mesh.attrs[NUMBER_OF_REAL_CELLS] = nreal
