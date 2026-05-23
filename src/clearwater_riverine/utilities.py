import warnings
from pathlib import Path

import numba
import pandas as pd
import numpy as np
import xarray as xr

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.variables import (
    ADVECTION_COEFFICIENT,
    AVERAGE_DEPTH,
    CELL_EDDY_VISCOSITY_X,
    CELL_EDDY_VISCOSITY_Y,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    DIFFUSION_COEFFICIENT,
    EDDY_VISCOSITY,
    EDGE_FACE_CONNECTIVITY,
    EDGE_VERTICAL_AREA,
    EDGE_VELOCITY,
    FACES,
    FACE_HYD_DEPTH,
    FACE_TO_FACE_DISTANCE,
    FACE_VEL_MAG,
    FACE_VEL_X,
    FACE_VEL_Y,
    FACE_X,
    FACE_Y,
    FLOW_ACROSS_FACE,
    LOOKUP_ELEVATION,
    LOOKUP_VOLUME,
    LOOKUP_WETTED_SURFACE_AREA,
    MANNINGS_N,
    MAXIMUM_DEPTH,
    NUMBER_OF_REAL_CELLS,
    TIME,
    FACE_SURFACE_AREA,
    VOLUME,
    VOLUME_ELEVATION_LOOKUP,
    WATER_SURFACE_ELEVATION,
    WETTED_SURFACE_AREA,
)


# -----------------------------------------------------------------------------
# Geometry-derived fallbacks for missing RAS optional temporal outputs.
#
# Some RAS plans are configured without "Cell Volume" or "Face Flow" in the
# unsteady output set (e.g. the Corvallis_Santiam plan, Phase J+1 2026-05-23).
# Both variables are deterministic from quantities the run DOES output
# (``Water Surface``, ``Face Velocity``) plus the static lookup tables RAS
# already writes into the geometry group:
#
#   * ``Cells Volume Elevation Info / Values``  -> V(WSE) per cell
#   * ``Faces Area Elevation Info  / Values``   -> A(WSE) per face
#
# These three helpers are direct ports of the kernels in the streaming fork's
# ``utilities.py`` (``_compute_cell_volumes``, ``_compute_face_areas``). They
# are private and gated by opt-in flags on the reader -- absence of the RAS
# output still raises a loud error by default, naming both the RAS option to
# enable AND the YAML / CLI flag to opt into the fallback. See
# ``RASHDFDataSource.__init__`` for the gating logic and
# ``design/missing_temporal_fallback.md`` for the fidelity notes.
#
# Numerical fidelity (validated against the Santiam-Salem fixture; results
# recorded in the design note):
#   * Cell Volume:  RAS uses the same lookup table internally; agreement is
#                   sub-0.1% for any cell well inside its tabulated range.
#   * Face Flow:    post-hoc reconstruction (face_area * edge_velocity); RAS's
#                   actual face flow embeds the full SWE momentum balance.
#                   Typically within a few percent for well-resolved steady
#                   flow; larger error at wet/dry edges. NOT a substitute for
#                   re-running RAS with the optional outputs enabled.
# -----------------------------------------------------------------------------
@numba.njit
def _linear_interpolate(x0, x1, y0, y1, xi):
    """Linear interpolation: return y(xi) given the two bracketing points."""
    m = (y1 - y0) / (x1 - x0)
    return m * (xi - x0) + y0


@numba.njit
def _compute_cell_volumes(
    water_surface_elev_arr: np.ndarray,
    cells_surface_area_arr: np.ndarray,
    starting_index_arr: np.ndarray,
    count_arr: np.ndarray,
    elev_arr: np.ndarray,
    vol_arr: np.ndarray,
) -> np.ndarray:
    """Compute per-cell volumes from WSE via the RAS volume-elevation lookup.

    The above-table extrapolation uses ``V_max + (WSE - Z_max) * surface_area``,
    which assumes a horizontal water surface across the cell. Mark Jensen
    (USACE/HEC) confirmed the validity of this method on Jul 29, 2022.

    Cells whose ``starting_index`` is past the end of the values array, or
    whose ``count`` is zero, are ghost cells: their volume is 0.
    """
    ntimes, ncells = water_surface_elev_arr.shape
    cell_volumes = np.zeros((ntimes, ncells))

    for time in range(ntimes):
        for cell in range(ncells):
            water_surface_elev = water_surface_elev_arr[time, cell]
            surface_area = cells_surface_area_arr[cell]
            index = starting_index_arr[cell]
            count = count_arr[cell]

            if index >= len(elev_arr) or count == 0:
                cell_volumes[time, cell] = 0.0
                continue

            elev = elev_arr[index:index + count]
            vol = vol_arr[index:index + count]

            if water_surface_elev > elev[-1]:
                cell_volumes[time, cell] = (
                    vol[-1] + (water_surface_elev - elev[-1]) * surface_area
                )
            elif water_surface_elev == elev[-1]:
                cell_volumes[time, cell] = vol[-1]
            elif water_surface_elev <= elev[0]:
                cell_volumes[time, cell] = vol[0]
            else:
                cell_volumes[time, cell] = 0.0  # default; replaced below
                npts = len(elev)
                for i in range(npts - 1, -1, -1):
                    if elev[i] < water_surface_elev:
                        cell_volumes[time, cell] = _linear_interpolate(
                            elev[i], elev[i + 1],
                            vol[i], vol[i + 1],
                            water_surface_elev,
                        )
                        break

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
    """Compute per-edge wetted cross-section areas from WSE via the RAS face
    area-elevation lookup.

    Per the RAS convention, each face is keyed by its ``Cell 0`` (the cell
    on side 1 of the edge); the face's WSE is taken from that cell. Above-
    table extrapolation uses ``A_max + (WSE - Z_max) * face_length``. Mark
    Jensen confirmation noted above.
    """
    ntimes, _ = water_surface_elev_arr.shape
    nfaces = len(faces_lengths_arr)
    face_areas = np.zeros((ntimes, nfaces))

    for time in range(ntimes):
        for face in range(nfaces):
            cell = faces_cell_indexes_arr[face]
            water_surface_elev = water_surface_elev_arr[time, cell]
            index = starting_index_arr[face]
            count = count_arr[face]

            if index >= len(elev_arr) or count == 0:
                face_areas[time, face] = 0.0
                continue

            elev = elev_arr[index:index + count]
            area = area_arr[index:index + count]

            if water_surface_elev > elev[-1]:
                face_areas[time, face] = (
                    area[-1]
                    + (water_surface_elev - elev[-1]) * faces_lengths_arr[face]
                )
            elif water_surface_elev == elev[-1]:
                face_areas[time, face] = area[-1]
            elif water_surface_elev <= elev[0]:
                face_areas[time, face] = area[0]
            else:
                face_areas[time, face] = 0.0  # default; replaced below
                npts = len(elev)
                for i in range(npts - 1, -1, -1):
                    if elev[i] < water_surface_elev:
                        face_areas[time, face] = _linear_interpolate(
                            elev[i], elev[i + 1],
                            area[i], area[i + 1],
                            water_surface_elev,
                        )
                        break

    return face_areas


def calculate_distances_cell_centroids(
        registry: VariableRegistry,
    ) -> np.array:
    """ Calculate the distance between cell centroids

    Args:
        registry: VariableRegistry 

    Returns:
        dist_data (np.array):   Array of distances between all cell centroids 
    """
    # Get northings and eastings of relevant faces 
    face_x = registry.get_variable(FACE_X).get_data()
    face_y = registry.get_variable(FACE_Y).get_data()
    edges_face1 = registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data().T[0]
    edges_face2 = registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data().T[1]

    x1_coords = face_x[edges_face1]  #mesh['face_x'][mesh['edges_face1']]
    y1_coords = face_y[edges_face1]
    x2_coords = face_x[edges_face2] 
    y2_coords = face_y[edges_face2]

    # calculate distance 
    dist_data = xr.DataArray(
        np.sqrt((x1_coords - x2_coords)**2 + (y1_coords - y2_coords)**2),
        dims = ('nedge'),  ## TODO: add unit management
    )
    return DataArrayVariable(dist_data)


def calculate_edge_vertical_area(
    registry: VariableRegistry
):
    vertical_area = registry.get_variable(FLOW_ACROSS_FACE).get_data() / registry.get_variable(EDGE_VELOCITY).get_data()
    # replace any NaN or Inf with 0
    vertical_area = vertical_area.where(np.isfinite(vertical_area), 0)
    return DataArrayVariable(vertical_area)


def _cell_diffusion_to_edge(D_cell, edges_face1, edges_face2, nreal):
    """Interpolate per-cell diffusion coefficients to edges via harmonic mean.

    Phase F T2-C (2026-05-21): forward-port of the streaming helper used
    by all non-constant diffusion methods (Elder, eddy viscosity, array).
    At boundary edges (``edges_face2 > nreal``, i.e. ghost-on-side-2),
    uses the real cell's value directly (no harmonic-mean blend with
    ghost data). At interior edges, returns
    ``2 * D1 * D2 / max(D1 + D2, eps)``.

    Args:
        D_cell: Per-cell diffusion values, shape (..., nface)
        edges_face1: Face indices on side 1 of each edge, shape (nedge,)
        edges_face2: Face indices on side 2 of each edge, shape (nedge,)
        nreal: Number of real (non-ghost) cells. ``edges_face2 > nreal``
            flags BC edges.

    Returns:
        D_edge: Per-edge diffusion values, shape matches ``D_cell[..., 0]``
            with the leading dims preserved and the last axis = nedge.
    """
    D1 = D_cell[..., edges_face1]
    D2 = D_cell[..., edges_face2]
    ghost_mask = edges_face2 > nreal
    return np.where(
        ghost_mask,
        D1,
        2.0 * D1 * D2 / np.maximum(D1 + D2, 1e-30),
    )


def _calc_diffusion_elder(
    registry: VariableRegistry,
    alpha: float = 0.6,
    g: float = 9.81,
) -> np.ndarray:
    """Elder (1959) depth-velocity scaling for diffusion (Phase F T2-C).

    ``D_cell = alpha * u* * h``, where ``u* = V * sqrt(g) * n / h^(1/6)``
    via Manning's. Requires ``MANNINGS_N``, ``FACE_HYD_DEPTH``, and
    velocity magnitude (``FACE_VEL_MAG`` or both
    ``FACE_VEL_X`` + ``FACE_VEL_Y``) in the registry. Raises with a
    clear message when any are missing.

    Returns per-edge ``D_edge`` of shape (time, nedge), suitable for
    multiplying by ``edge_vertical_area / face_to_face_distance`` to
    form the COEFFICIENT_TO_DIFFUSION_TERM.
    """
    if MANNINGS_N not in registry:
        raise NotImplementedError(
            "Elder diffusion method requires Manning's n in the registry. "
            "Phase F T2-C ported the helper; wiring the canonical HDF "
            "reader to register MANNINGS_N from 'Cells Center Manning's n' "
            "is a follow-up commit (the path exists in io/hdf.py path "
            "scaffolding but is not yet in __read_temporal_variables/"
            "__read_static_variables). Until then, use "
            "diffusion_coefficient as a scalar (constant method)."
        )
    if FACE_HYD_DEPTH not in registry:
        raise ValueError(
            "Elder diffusion method requires hydraulic depth. "
            "Set wet_dry_metric to 'depth' or 'both' (which auto-registers "
            "FACE_HYD_DEPTH from WSE - cell_min_elev), or re-run RAS with "
            "'Cell Hydraulic Depth' temporal output enabled."
        )
    mannings_n = np.asarray(registry.get_variable(MANNINGS_N).get_data())
    depth = np.asarray(registry.get_variable(FACE_HYD_DEPTH).get_data())

    if FACE_VEL_MAG in registry:
        vel_mag = np.asarray(registry.get_variable(FACE_VEL_MAG).get_data())
    elif FACE_VEL_X in registry and FACE_VEL_Y in registry:
        vx = np.asarray(registry.get_variable(FACE_VEL_X).get_data())
        vy = np.asarray(registry.get_variable(FACE_VEL_Y).get_data())
        vel_mag = np.sqrt(vx * vx + vy * vy)
    else:
        raise NotImplementedError(
            "Elder diffusion method requires cell velocity magnitude or "
            "components. Phase F T2-C ported the helper; wiring "
            "FACE_VEL_X / FACE_VEL_Y into the canonical HDF reader's "
            "temporal_variables is a follow-up commit. Until then, use "
            "diffusion_coefficient as a scalar (constant method)."
        )

    safe_depth = np.maximum(depth, 1e-10)
    shear_velocity = vel_mag * np.sqrt(g) * mannings_n / safe_depth ** (1.0 / 6.0)
    D_cell = alpha * shear_velocity * safe_depth

    ef = np.asarray(registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data())
    f1 = ef[:, 0].astype(np.int64)
    f2 = ef[:, 1].astype(np.int64)
    nreal = int(registry.get_variable(NUMBER_OF_REAL_CELLS).get_data()) - 1
    return _cell_diffusion_to_edge(D_cell, f1, f2, nreal)


def _calc_diffusion_eddy_viscosity(
    registry: VariableRegistry,
    Sc_t: float = 1.0,
) -> np.ndarray:
    """Eddy-viscosity diffusion: ``D = nu_t / Sc_t`` (Phase F T2-C).

    Uses ``EDDY_VISCOSITY`` (per-edge, time-varying) if available.
    Falls back to ``CELL_EDDY_VISCOSITY_X / Y`` magnitudes interpolated
    to edges via harmonic mean.
    """
    if EDDY_VISCOSITY in registry:
        return np.asarray(registry.get_variable(EDDY_VISCOSITY).get_data()) / Sc_t
    if CELL_EDDY_VISCOSITY_X in registry and CELL_EDDY_VISCOSITY_Y in registry:
        nx = np.asarray(registry.get_variable(CELL_EDDY_VISCOSITY_X).get_data())
        ny = np.asarray(registry.get_variable(CELL_EDDY_VISCOSITY_Y).get_data())
        nu_t_cell = np.sqrt(nx * nx + ny * ny)
        ef = np.asarray(registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data())
        f1 = ef[:, 0].astype(np.int64)
        f2 = ef[:, 1].astype(np.int64)
        nreal = int(registry.get_variable(NUMBER_OF_REAL_CELLS).get_data()) - 1
        return _cell_diffusion_to_edge(nu_t_cell / Sc_t, f1, f2, nreal)
    raise NotImplementedError(
        "Eddy-viscosity diffusion method requires EDDY_VISCOSITY or "
        "CELL_EDDY_VISCOSITY_X+Y in the registry. Phase F T2-C ported "
        "the helper; wiring these into the canonical HDF reader is a "
        "follow-up commit. Until then, use diffusion_coefficient as a "
        "scalar (constant method)."
    )


def _calc_diffusion_array(
    registry: VariableRegistry,
    filepath: str | Path,
    default_value: float = 0.0,
) -> np.ndarray:
    """Per-cell diffusion from a CSV (Phase F T2-C; Phase H-3 schema clarified).

    CSV schema (header required):

        cell_index,diffusion_coefficient
        0,0.01
        1,0.02
        ...

    Static values applied at all timesteps; cells absent from the CSV
    receive ``default_value``.

    Phase H-3 (2026-05-21): the prior docstring claimed "no header
    required" but ``pd.read_csv(filepath)`` uses ``header=0`` by
    default and would silently consume the first DATA row as the
    column-name row. Now requires a named header so the contract is
    explicit and validated below.
    """
    df = pd.read_csv(filepath)
    required = {"cell_index", "diffusion_coefficient"}
    missing = required - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(
            f"Array diffusion CSV {filepath!r} missing required columns: "
            f"{sorted(missing)}. Expected header row "
            "'cell_index,diffusion_coefficient'."
        )
    # Normalize column names case-insensitively for robust lookup.
    col_map = {c.lower(): c for c in df.columns}
    cell_col = col_map["cell_index"]
    diff_col = col_map["diffusion_coefficient"]
    volume = registry.get_variable(VOLUME).get_data()
    nface = int(volume.sizes['nface'])
    ntimes = int(volume.sizes['time'])
    D_cell = np.full(nface, default_value, dtype=np.float64)
    D_cell[df[cell_col].values.astype(int)] = df[diff_col].values
    D_cell_tv = np.broadcast_to(D_cell, (ntimes, nface)).copy()
    ef = np.asarray(registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data())
    f1 = ef[:, 0].astype(np.int64)
    f2 = ef[:, 1].astype(np.int64)
    nreal = int(registry.get_variable(NUMBER_OF_REAL_CELLS).get_data()) - 1
    return _cell_diffusion_to_edge(D_cell_tv, f1, f2, nreal)


def calculate_coeff_to_diffusion_term(
        registry: VariableRegistry,
    ) -> np.array:
    """ Calculate the coefficient to the diffusion term.

    For each edge, this is calculated as:
    (Edge vertical area * diffusion coefficient) / (distance between cells)

    Phase F T2-C (2026-05-21): when the registry carries a
    ``DIFFUSION_METHOD`` string variable (a Python ``str`` registered
    on the model at init time), this function dispatches to the
    appropriate per-method helper (``_calc_diffusion_elder``,
    ``_calc_diffusion_eddy_viscosity``, ``_calc_diffusion_array``).
    Default behaviour (no ``DIFFUSION_METHOD`` registered) is constant
    diffusion using the scalar ``DIFFUSION_COEFFICIENT``.

    Args:
        registry: VariableRegistry

    Returns:
        diffusion_array (np.array):     Array of diffusion coefficients associated with each edge
    """
    edge_vertical_area = registry.get_variable(EDGE_VERTICAL_AREA).get_data()
    face_to_face_distance = registry.get_variable(FACE_TO_FACE_DISTANCE).get_data()

    # Phase F T2-C dispatch. ``diffusion_method`` is stored as a
    # FloatVariable wrapping a flag (0=constant, 1=elder, 2=eddy,
    # 3=array) when set by the model; absent means constant.
    method_var = "diffusion_method"
    if method_var in registry:
        method_code = int(registry.get_variable(method_var).get_data())
    else:
        method_code = 0  # constant

    if method_code == 0:  # constant
        diffusion_coefficient = registry.get_variable(DIFFUSION_COEFFICIENT).get_data()
        diffusion_array = edge_vertical_area * diffusion_coefficient / face_to_face_distance
        return DataArrayVariable(diffusion_array)

    # Non-constant: build D_edge (shape (time, nedge) or (nedge,))
    if method_code == 1:  # elder
        alpha = float(registry.get_variable("diffusion_alpha").get_data()) if "diffusion_alpha" in registry else 0.6
        D_edge = _calc_diffusion_elder(registry, alpha=alpha)
    elif method_code == 2:  # eddy_viscosity
        Sc_t = float(registry.get_variable("diffusion_schmidt").get_data()) if "diffusion_schmidt" in registry else 1.0
        D_edge = _calc_diffusion_eddy_viscosity(registry, Sc_t=Sc_t)
    elif method_code == 3:  # array
        if "diffusion_array_path" not in registry:
            raise ValueError(
                "Array diffusion method requires ``diffusion_array_path`` "
                "registered on the registry (set by the model from "
                "diffusion_coefficient.data.file_path in the config)."
            )
        # Stored as a FloatVariable wrapping a hash-or-index would be
        # awkward; use the registry-attached attribute instead. The
        # model passes the path via a custom registry helper.
        from clearwater_data.variables.float import FloatVariable
        fv = registry.get_variable("diffusion_array_path")
        path = str(getattr(fv, "_path", "")) or ""
        if not path:
            raise ValueError("diffusion_array_path was registered but had no _path attr")
        D_edge = _calc_diffusion_array(registry, filepath=path)
    else:
        raise ValueError(f"Unknown diffusion method code {method_code}")

    # Multiply by edge_vertical_area / face_to_face_distance to form
    # the COEFFICIENT_TO_DIFFUSION_TERM that the LHS consumes.
    eva = np.asarray(edge_vertical_area)
    f2f = np.asarray(face_to_face_distance)
    diff_arr = eva * D_edge / np.maximum(f2f, 1e-30)
    # Wrap in a DataArray that matches the time + nedge dims of
    # edge_vertical_area for the registry.
    return DataArrayVariable(
        xr.DataArray(
            diff_arr,
            dims=edge_vertical_area.dims,
            coords=edge_vertical_area.coords,
        )
    )


def calculate_change_in_time(
    registry: VariableRegistry
):
    times = registry.get_variable(VOLUME).get_data().time
    dt = np.ediff1d(times)
    dt = dt / np.timedelta64(1, 's')
    dt = np.insert(dt, len(dt), np.nan)
    if np.all(dt[:-1] == dt[0]):
        return FloatVariable(dt[0])
    else:
        # Phase F (2026-05-21): also carry the ``time`` coordinate so
        # ``get_at_time(CHANGE_IN_TIME, t)`` can ``.sel`` by timestamp.
        # The previous DataArray had ``dims=('time')`` but no coord
        # values along that dim; ``.sel(time=Timestamp)`` then routed
        # through positional fancy-indexing and raised
        # ``invalid indexer array, does not have integer dtype``.
        dt = xr.DataArray(
            dt,
            dims=('time',),
            coords={'time': times.values},
            attrs={'Units': 's'}
        )
        return DataArrayVariable(dt)


def calculate_wetted_surface_area(
    registry: VariableRegistry
):
    """
    Calculate wetted surface area based on elevation-volume lookup table.
    """
    # Define required dimensions for lookup xarray
    nface = len(registry.get_variable(FACE_SURFACE_AREA).get_data()[FACES])
    ntime = len(registry.get_variable(VOLUME).get_data()["time"])
    lookup_volumes = registry.get_variable(LOOKUP_VOLUME).get_data()
    lookup_areas = registry.get_variable(LOOKUP_WETTED_SURFACE_AREA).get_data()

    # fill null lookup values with the maximum
    # this will help the interpolation function work correctly for large values
    lookup_volumes = lookup_volumes.fillna(lookup_volumes.max(dim='index', skipna=True))
    lookup_areas = lookup_areas.fillna(lookup_areas.max(dim='index', skipna=True))

    # preallocate output array
    result = xr.DataArray(
        np.full((ntime, nface), np.nan),
        dims=[TIME, FACES],
        coords={
            TIME: registry.get_variable(VOLUME).get_data()["time"],
            FACES:np.arange(nface),
        }
    )

    # loop through real faces, get wetted surface area for all timesteps
    for nf in lookup_areas.nface:
        volumes = registry.get_variable(VOLUME).get_data().sel(nface=nf).values
        result[:,  nf] = np.interp(
            volumes,
            lookup_volumes.sel(nface=nf).values,
            lookup_areas.sel(nface=nf).values,
            left=lookup_areas.sel(nface=nf).values[0],  # interp to lowermost value
            right=lookup_areas.sel(nface=nf).values[-1],  # interp to largest value
        )

    # Convert result back to xarray.DataArray
    return DataArrayVariable(result)

def calculate_average_depth(
    registry: VariableRegistry   
):
    """Calculate average depth based on volume and wetted surface area."""
    # If wetted surface area does not exist, calculate it.
    if WETTED_SURFACE_AREA not in registry:
        wetted_surface_area = calculate_wetted_surface_area(registry)
        registry.register(
            WETTED_SURFACE_AREA,
            wetted_surface_area,
        )
    
    # Calculate average depth
    average_depth = xr.where(
        registry.get_variable(WETTED_SURFACE_AREA).get_data() > 0,
        registry.get_variable(VOLUME).get_data() / registry.get_variable(WETTED_SURFACE_AREA).get_data(),
        0
    )

    return DataArrayVariable(average_depth)


def _cell_minimum_elevation(registry: VariableRegistry):
    """Return per-cell minimum bed elevation as a 1-D DataArray over ``nface``.

    ``LOOKUP_ELEVATION`` is the full elevation lookup curve per cell,
    shape ``(nface, index)``. The cell's true minimum (bed) elevation is
    the first knot of that curve (``index=0``), which is what HEC-RAS
    means by "Cells Minimum Elevation" in the geometry section. The
    lookup only covers real cells; ghost cells become NaN here, which
    propagates correctly through depth-based wet/dry checks (NaN > h_min
    is False).

    Phase F (2026-05-21): replaces the prior pattern that subtracted the
    whole 2-D lookup curve from WSE, which produced a spurious 3-D
    ``(time, nface, index)`` result instead of the scalar-per-cell depth
    the consumers expected.
    """
    lookup_elev = registry.get_variable(LOOKUP_ELEVATION).get_data()
    min_elev = lookup_elev.isel(index=0)
    return min_elev.reindex(nface=np.arange(len(registry.get_variable(FACE_X).get_data())))


def calculate_face_hyd_depth(
    registry: VariableRegistry,
):
    """Compute ``FACE_HYD_DEPTH`` = water-surface elevation - cell-bed elevation.

    Fallback for RAS HDFs that do not write the optional
    "Cell Hydraulic Depth" temporal variable (the Santiam-Salem subset
    used in Phase F validation is one such case). Result shape
    ``(time, nface)`` with NaN at ghost cells.

    Phase F (2026-05-21): unblocks ``wet_dry_metric="both"`` (and
    ``"depth"``) on canonical for RAS HDFs that ship only the minimal
    output set (Water Surface, Depth-derived-from-WSE-via-lookup, Face
    Velocity).
    """
    min_elev = _cell_minimum_elevation(registry)
    depth = registry.get_variable(WATER_SURFACE_ELEVATION).get_data() - min_elev
    return DataArrayVariable(depth)


def calculate_maximum_depth(
    registry: VariableRegistry
):
    """Calculate the maximum depth based on water surface elevation.

    Phase F (2026-05-21): use :func:`_cell_minimum_elevation` to extract
    the scalar bed elevation per cell rather than subtracting the whole
    elevation-volume lookup curve (which previously produced a spurious
    3-D ``(time, nface, index)`` result).
    """
    min_elev = _cell_minimum_elevation(registry)
    maximum_depth = registry.get_variable(WATER_SURFACE_ELEVATION).get_data() - min_elev
    return DataArrayVariable(maximum_depth)


def compute_wet_mask(
    volume,
    depth=None,
    *,
    h_min: float = 0.01,
    V_min: float = 0.1,
    metric: str = "both",
):
    """Compute a per-cell wet/dry boolean mask (Phase-D Unit A).

    Pure function; no registry coupling. ``volume`` and ``depth`` may be
    NumPy arrays or xarray DataArrays of compatible shape (typically
    ``(time, nface)``). The returned mask has the same shape and dtype
    bool: ``True`` where the cell is considered wet, ``False`` otherwise.

    Parameters
    ----------
    volume : array-like
        Cell volume (per HEC-RAS output), shape ``(time, nface)`` or
        ``(nface,)``.
    depth : array-like, optional
        Cell hydraulic depth, required when ``metric`` is ``"depth"`` or
        ``"both"`` (the default). Pass ``None`` if metric is ``"volume"``.
    h_min : float
        Minimum hydraulic depth (m) for a cell to count as wet under the
        depth-based metric. Default 0.01 m matches the fork's design spec.
    V_min : float
        Minimum cell volume (m^3) for a cell to count as wet under the
        volume-based metric. Default 0.1 m^3 matches the fork's spec.
    metric : {"depth", "volume", "both"}
        Which physical criterion to apply. ``"both"`` requires both
        ``depth > h_min`` AND ``volume > V_min`` (most conservative;
        matches the fork's default). ``"depth"`` uses only the depth
        threshold; ``"volume"`` uses only the volume threshold (and
        does not require ``depth``).

    Returns
    -------
    Boolean array or DataArray with the same shape as ``volume``.
    """
    if metric not in ("depth", "volume", "both"):
        raise ValueError(
            f"metric={metric!r} is not one of 'depth', 'volume', 'both'"
        )
    if metric in ("depth", "both") and depth is None:
        raise ValueError(
            f"metric={metric!r} requires the depth argument; "
            f"pass depth=... or use metric='volume'"
        )
    if metric == "volume":
        return volume > V_min
    if metric == "depth":
        return depth > h_min
    # metric == "both"
    return (volume > V_min) & (depth > h_min)


# ---------------------------------------------------------------------
# Phase F (2026-05-21): per-cell volume-continuity correction.
#
# Forward-ported from the streaming repo's
# ``utilities.py:_apply_continuity_correction`` (~445 lines) and adapted
# to the canonical VariableRegistry + EDGE_FACE_CONNECTIVITY convention.
#
# Why: HEC-RAS's serialized hourly output is not exactly continuity-
# closing in time. For each timestep ``t`` and real cell ``i`` the
# discrete residual
#
#     r_i(t) = (V_i[t+1] - V_i[t]) - dt[t] * net(face_flow[t]) at i
#
# is bounded but nonzero. On well-behaved decks it stays at round-off;
# on subset HDFs whose upstream BC was an Internal type in the original
# RAS solution (e.g. Santiam-Salem, where Upstream and Santiam are
# Internal BC lines), the per-cell residual reaches O(10%) of the
# cell's total inflow and the implicit advection-diffusion solve
# amplifies that into a compounding mass-balance error.
#
# Two modes are exposed. ``bc_only`` (default) closes the residual on
# boundary edges only; ``all_edges`` solves a graph-Laplacian system
# that redistributes the per-cell residual across all incident edges
# (the full Option B from streaming's
# ``design/mass_conservation_followups.md`` §1.3). The streaming
# reference run for Santiam-Salem (``v3_smoke_15day_wind10m_final_
# mumax_1_3``, locked T baseline Salem bias -0.30 deg C / RMSE 0.62
# deg C) uses ``all_edges``.
#
# Design intent: the corrected coefficient lives on
# ``ADVECTION_COEFFICIENT``; ``FLOW_ACROSS_FACE`` is left untouched so
# downstream diagnostics see the raw RAS data.
# ---------------------------------------------------------------------

def _apply_continuity_correction(
    registry: VariableRegistry,
    adv_coeff: np.ndarray,
    mode: str = "bc_only",
    eps: float = 1e-12,
    eps_warn: float = 1e-6,
    eps_converged: float = 1e-6,
    max_iter: int = 25,
    omega: float = 1.0,
) -> None:
    """Add a continuity-restoring correction to ``adv_coeff`` (Option B).

    Modifies ``adv_coeff`` in place. The corrected coefficient should be
    registered as ``ADVECTION_COEFFICIENT`` and consumed by the LHS in
    place of the raw ``FLOW_ACROSS_FACE``. See module docstring above
    for the math and rationale.

    No-op if the registry does not contain ``VOLUME`` or
    ``EDGE_FACE_CONNECTIVITY`` (e.g. minimal unit-test fixtures).

    Args:
        registry: VariableRegistry with ``VOLUME``, ``EDGE_FACE_CONNECTIVITY``,
            and ``NUMBER_OF_REAL_CELLS`` registered.
        adv_coeff: Working ``(ntime, nedge)`` advection-coefficient
            array. Modified in place.
        mode: ``"bc_only"`` (default) or ``"all_edges"``.
        eps: Per-cell residual magnitude below which no correction is
            applied (treated as round-off).
        eps_warn: Cumulative ``sum |r_i|`` over uncorrected cells beyond
            which a warning is logged (``"bc_only"`` mode only).
        eps_converged: ``"all_edges"`` convergence threshold on
            ``max(|r_i|)``.
        max_iter: Maximum refinement passes in ``"all_edges"`` mode.
        omega: Under-relaxation factor for the inner refinement passes
            (default 1.0 for a full Newton-style update).
    """
    if mode not in ("bc_only", "all_edges"):
        raise ValueError(
            f"Unknown continuity_correction mode {mode!r}. "
            "Expected 'bc_only' or 'all_edges'."
        )

    if (
        VOLUME not in registry
        or EDGE_FACE_CONNECTIVITY not in registry
        or NUMBER_OF_REAL_CELLS not in registry
    ):
        return

    # Phase H-5 (2026-05-21): defensive NaN guard at entry. If
    # ``adv_coeff`` (a copy of FLOW_ACROSS_FACE) carries NaN at any
    # edge / time step, the np.add.at / np.where downstream silently
    # propagate NaN through ``Q``, ``r``, and the per-edge correction
    # delta. The corrupted coefficient then flows into the LHS at
    # linalg.py and into the ghost-cell BC flux, silently degrading
    # the entire transport solve. RAS HDFs should not carry NaN at
    # populated edges; if they do, the right behavior is to fail
    # loudly here rather than mask the source defect.
    if not np.all(np.isfinite(adv_coeff)):
        n_nan = int(np.isnan(adv_coeff).sum())
        n_inf = int(np.isinf(adv_coeff).sum())
        raise ValueError(
            f"FLOW_ACROSS_FACE / ADVECTION_COEFFICIENT contains "
            f"{n_nan} NaN and {n_inf} Inf values at entry to "
            "continuity_correction. The correction would propagate "
            "these into the LHS coefficient matrix and silently "
            "corrupt the WQ transport solve. Investigate the source "
            "RAS HDF or any wet/dry amendments that may be writing "
            "NaN to flow_across_face."
        )

    nreal_count = int(registry.get_variable(NUMBER_OF_REAL_CELLS).get_data())  # number of real cells
    nreal_attr = nreal_count - 1  # maximum real-cell index (streaming convention)

    V = np.asarray(registry.get_variable(VOLUME).get_data())  # (ntime, nface)
    ef = np.asarray(registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data())
    f1 = ef[:, 0].astype(np.int64)
    f2 = ef[:, 1].astype(np.int64)
    n_face = V.shape[1] if V.ndim == 2 else int(max(f1.max(), f2.max()) + 1)
    n_edge = f1.shape[0]

    # Compute dt locally; the time coordinate is on ``VOLUME``.
    times = registry.get_variable(VOLUME).get_data().time.values
    if len(times) < 2:
        return
    dt = np.diff(times) / np.timedelta64(1, 's')  # (ntime - 1,)
    n_steps = len(dt)
    if adv_coeff.shape[0] < n_steps + 1 or n_steps == 0:
        return

    if mode == "bc_only":
        _apply_bc_only_correction(
            adv_coeff, V, f1, f2, dt,
            nreal_attr=nreal_attr,
            nreal_count=nreal_count,
            n_face=n_face,
            n_steps=n_steps,
            eps=eps,
            eps_warn=eps_warn,
        )
    else:  # mode == "all_edges"
        _apply_all_edges_correction(
            adv_coeff, V, f1, f2, dt,
            nreal_attr=nreal_attr,
            nreal_count=nreal_count,
            n_face=n_face,
            n_edge=n_edge,
            n_steps=n_steps,
            eps=eps,
            eps_converged=eps_converged,
            max_iter=max_iter,
            omega=omega,
        )


def _apply_bc_only_correction(
    adv_coeff: np.ndarray,
    V: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    dt: np.ndarray,
    *,
    nreal_attr: int,
    nreal_count: int,
    n_face: int,
    n_steps: int,
    eps: float,
    eps_warn: float,
) -> None:
    """BC-edge-only continuity correction (Option B-lite).

    Distributes each real cell's per-step residual ``r_i / dt`` across
    the cell's BC edges, weighted by ``|face_flow|``. Interior cells
    (no BC edge) are left alone; their cumulative ``sum |r_i|`` is
    reported via ``warnings.warn`` if it exceeds ``eps_warn``.
    """
    bc_edge_mask = (f2 > nreal_attr) | (f1 > nreal_attr)
    if not bc_edge_mask.any():
        return

    bc_edges = np.where(bc_edge_mask)[0]
    bc_f1 = f1[bc_edges]
    bc_f2 = f2[bc_edges]
    side_is_face1_real = bc_f1 <= nreal_attr
    bc_real_cell = np.where(side_is_face1_real, bc_f1, bc_f2)
    bc_inflow_sign = np.where(side_is_face1_real, -1.0, 1.0)

    sort_order = np.argsort(bc_real_cell, kind='stable')
    bc_edges_by_cell = bc_edges[sort_order]
    bc_real_cell_sorted = bc_real_cell[sort_order]
    bc_inflow_sign_sorted = bc_inflow_sign[sort_order]
    unique_cells, first_idx, counts = np.unique(
        bc_real_cell_sorted, return_index=True, return_counts=True,
    )

    has_bc_edge = np.zeros(n_face, dtype=bool)
    has_bc_edge[unique_cells] = True

    interior_residual_l1 = 0.0

    for t in range(n_steps):
        # Phase I-4 / F12 (2026-05-21): defensive dt=0 guard.
        # The Phase F cadence guard rejects > 10% deviation but a
        # perfectly-zero dt (degenerate / duplicate timestamp) is
        # not caught by that guard. Division by zero in
        # ``target_total = r_i / dt[t]`` would silently produce
        # inf/NaN and corrupt every adv_coeff entry it touches.
        # Skip this timestep with a warning instead.
        if dt[t] == 0:
            warnings.warn(
                f"continuity_correction (bc_only): dt[{t}] == 0 -- "
                "skipping this timestep. RAS output has a duplicate "
                "timestamp; investigate the source HDF.",
                stacklevel=4,
            )
            continue
        ff_t = adv_coeff[t]
        Q = np.zeros(n_face)
        np.add.at(Q, f1, -ff_t)
        np.add.at(Q, f2, +ff_t)

        dV = V[t + 1, :nreal_count] - V[t, :nreal_count]
        r = dV - dt[t] * Q[:nreal_count]
        abs_r = np.abs(r)

        interior_mask = ~has_bc_edge[:nreal_count]
        interior_residual_l1 += float(np.sum(abs_r[interior_mask]))

        for cell, start, count in zip(unique_cells, first_idx, counts):
            r_i = r[cell]
            if abs(r_i) <= eps:
                continue

            edges = bc_edges_by_cell[start:start + count]
            signs = bc_inflow_sign_sorted[start:start + count]
            target_total = r_i / dt[t]

            mags = np.abs(ff_t[edges])
            mag_sum = mags.sum()
            if mag_sum > 0.0:
                weights = mags / mag_sum
            else:
                weights = np.full(len(edges), 1.0 / len(edges))

            delta_q_per_edge = target_total * weights
            delta_adv = delta_q_per_edge / signs
            ff_t[edges] += delta_adv

    if interior_residual_l1 > eps_warn:
        warnings.warn(
            "BC-edge continuity correction left "
            f"{interior_residual_l1:.4g} of cumulative |residual| on interior "
            "cells uncovered (these cells touch no boundary edge so a "
            "BC-only correction cannot redistribute their residual). "
            "If this is large relative to domain volume, set "
            "continuity_correction='all_edges' to redistribute the "
            "interior residual across internal edges as well.",
            stacklevel=3,
        )


def _apply_all_edges_correction(
    adv_coeff: np.ndarray,
    V: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    dt: np.ndarray,
    *,
    nreal_attr: int,
    nreal_count: int,
    n_face: int,
    n_edge: int,
    n_steps: int,
    eps: float,
    eps_converged: float,
    max_iter: int,
    omega: float,
) -> None:
    """Full Option-B continuity correction across all edges.

    Solves a graph-Laplacian system per timestep to find the minimum-
    L2-norm per-edge correction satisfying the per-cell continuity
    constraint. Factorizes the cell-Laplacian once and reuses; per-step
    cost is one sparse triangular solve.

    For real cell ``i`` and edge ``e`` incident to ``i``:
        sum_{e in edges(i)} signs_i_e * c[e] = r_i / dt        (*)
    where signs are -1 if cell i is f1[e], +1 if f2[e]. (*) is
    ``D c = b`` with ``D`` the signed cell-edge incidence matrix
    (nreal_count x n_edge) and ``b = r / dt``. Minimum-norm solution:
    ``c = D^T phi``, ``L phi = b``, ``L = D D^T`` (the cell-graph
    Laplacian).
    """
    from scipy.sparse import csr_matrix, csc_matrix
    from scipy.sparse.linalg import splu

    rows = []
    cols = []
    vals = []
    for e in range(n_edge):
        if f1[e] <= nreal_attr:
            rows.append(int(f1[e]))
            cols.append(e)
            vals.append(-1.0)
        if f2[e] <= nreal_attr:
            rows.append(int(f2[e]))
            cols.append(e)
            vals.append(+1.0)
    if not rows:
        return
    D = csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64),
          np.asarray(cols, dtype=np.int64))),
        shape=(nreal_count, n_edge),
    )
    DT = D.T.tocsr()

    # Cell-graph Laplacian L = D D^T. Phase H-6 (2026-05-21): on
    # well-connected meshes (every cell reaches a BC via some edge
    # path), ``L`` is non-singular and a single iteration of the
    # ``c = D^T phi, L phi = b`` solve drives the per-cell residual
    # to round-off floor. The previous always-on Tikhonov ridge
    # ``1e-14 * max(L.diag(), 1.0)`` for closed-sub-domain robustness
    # was supposed to be numerically negligible, but on a
    # well-conditioned 2-cell mesh (plan02 fixture) it slowed
    # convergence from one-shot to geometric at ~0.55x per iteration,
    # leaving |r|~1e-6 after max_iter=5 and emitting a spurious
    # "did not converge" warning. Try the no-ridge solve first; fall
    # back to the ridge only if ``splu`` raises (which signals a
    # genuinely singular L).
    L_no_ridge = (D @ DT).tocsc()
    solver = None
    try:
        solver = splu(L_no_ridge)
    except Exception:
        # L is singular (closed sub-domain with no BC, or disconnected
        # components). Add the documented Tikhonov ridge and retry.
        ridge = 1e-14 * np.maximum(L_no_ridge.diagonal().max(), 1.0)
        L_ridged = L_no_ridge + csc_matrix(
            (np.full(nreal_count, ridge, dtype=np.float64),
             (np.arange(nreal_count, dtype=np.int64),
              np.arange(nreal_count, dtype=np.int64))),
            shape=(nreal_count, nreal_count),
        )
        try:
            solver = splu(L_ridged)
        except Exception as exc:  # pragma: no cover - degenerate-mesh defense
            warnings.warn(
                "all_edges continuity correction: cell-Laplacian factorization "
                f"failed even with Tikhonov ridge ({exc!r}). Skipping the "
                "correction; advection coefficient stays at the raw RAS "
                "face_flow.",
                stacklevel=3,
            )
            return

    worst_unconverged_cell = -1
    worst_unconverged_residual = 0.0
    worst_unconverged_t = -1

    for t in range(n_steps):
        # Phase I-4 / F12 (2026-05-21): defensive dt=0 guard
        # (mirrors the bc_only path; same rationale).
        if dt[t] == 0:
            warnings.warn(
                f"continuity_correction (all_edges): dt[{t}] == 0 -- "
                "skipping this timestep. RAS output has a duplicate "
                "timestamp; investigate the source HDF.",
                stacklevel=4,
            )
            continue
        ff_t = adv_coeff[t]
        dV = V[t + 1, :nreal_count] - V[t, :nreal_count]

        converged = False
        max_abs_r = 0.0
        r = np.zeros(nreal_count)
        for it in range(max_iter):
            Q = np.zeros(n_face)
            np.add.at(Q, f1, -ff_t)
            np.add.at(Q, f2, +ff_t)
            r = dV - dt[t] * Q[:nreal_count]
            max_abs_r = float(np.max(np.abs(r))) if r.size else 0.0
            if max_abs_r <= eps_converged:
                converged = True
                break

            b = r / dt[t]
            phi = solver.solve(b)
            c = DT @ phi  # per-edge correction
            ff_t += omega * c
        if not converged:
            if max_abs_r > worst_unconverged_residual:
                worst_unconverged_residual = max_abs_r
                worst_unconverged_cell = int(np.argmax(np.abs(r)))
                worst_unconverged_t = t

    if worst_unconverged_cell >= 0:
        warnings.warn(
            "all_edges continuity correction did not converge within "
            f"max_iter={max_iter} passes. Worst remaining residual: "
            f"|r_i|={worst_unconverged_residual:.4g} at cell "
            f"{worst_unconverged_cell}, timestep {worst_unconverged_t}. "
            "Input data is pathological enough that even Option B cannot "
            "close it; consider increasing max_iter or investigating the "
            "RAS continuity defect at this cell.",
            stacklevel=3,
        )


def register_advection_coefficient(
    registry: VariableRegistry,
    continuity_correction: str = "bc_only",
) -> None:
    """Register ``ADVECTION_COEFFICIENT`` on the registry (Phase F).

    Computes ``ADVECTION_COEFFICIENT`` as a copy of ``FLOW_ACROSS_FACE``
    with the continuity correction applied (per ``mode``), and registers
    it on the registry. Idempotent: unregisters any prior copy first.

    Args:
        registry: VariableRegistry already containing ``FLOW_ACROSS_FACE``,
            ``VOLUME``, ``EDGE_FACE_CONNECTIVITY``, ``NUMBER_OF_REAL_CELLS``.
        continuity_correction: ``"bc_only"`` (default), ``"all_edges"``,
            or ``"none"`` (skip the correction; ADVECTION_COEFFICIENT
            equals the raw FLOW_ACROSS_FACE).
    """
    if continuity_correction not in ("bc_only", "all_edges", "none"):
        raise ValueError(
            f"Unknown continuity_correction mode {continuity_correction!r}. "
            "Expected 'bc_only', 'all_edges', or 'none'."
        )
    flow = registry.get_variable(FLOW_ACROSS_FACE).get_data()
    adv_coeff = np.asarray(flow).copy()
    if continuity_correction != "none":
        _apply_continuity_correction(
            registry, adv_coeff, mode=continuity_correction,
        )
    adv_da = xr.DataArray(
        adv_coeff,
        dims=flow.dims,
        coords=flow.coords,
        attrs={**flow.attrs, 'continuity_correction': continuity_correction},
    )
    if ADVECTION_COEFFICIENT in registry:
        registry.unregister(ADVECTION_COEFFICIENT)
    registry.register(ADVECTION_COEFFICIENT, DataArrayVariable(adv_da))


CALCULATED_VARIABLE_MAP = {
    FACE_TO_FACE_DISTANCE: calculate_distances_cell_centroids,
    EDGE_VERTICAL_AREA: calculate_edge_vertical_area,
    COEFFICIENT_TO_DIFFUSION_TERM: calculate_coeff_to_diffusion_term,
    CHANGE_IN_TIME: calculate_change_in_time,
    WETTED_SURFACE_AREA: calculate_wetted_surface_area,
    AVERAGE_DEPTH: calculate_average_depth,
    MAXIMUM_DEPTH: calculate_maximum_depth,
    FACE_HYD_DEPTH: calculate_face_hyd_depth,
}

CALCULATED_VARIABLE_DEPENDENCIES = {
    AVERAGE_DEPTH: [WETTED_SURFACE_AREA],
}