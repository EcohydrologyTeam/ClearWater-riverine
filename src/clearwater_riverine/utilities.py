import warnings

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
    face_x = registry.get(FACE_X)
    face_y = registry.get(FACE_Y)
    edges_face1 = registry.get(EDGE_FACE_CONNECTIVITY).T[0]
    edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]

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
    vertical_area = registry.get(FLOW_ACROSS_FACE) / registry.get(EDGE_VELOCITY)
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
    mannings_n = np.asarray(registry.get(MANNINGS_N))
    depth = np.asarray(registry.get(FACE_HYD_DEPTH))

    if FACE_VEL_MAG in registry:
        vel_mag = np.asarray(registry.get(FACE_VEL_MAG))
    elif FACE_VEL_X in registry and FACE_VEL_Y in registry:
        vx = np.asarray(registry.get(FACE_VEL_X))
        vy = np.asarray(registry.get(FACE_VEL_Y))
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

    ef = np.asarray(registry.get(EDGE_FACE_CONNECTIVITY))
    f1 = ef[:, 0].astype(np.int64)
    f2 = ef[:, 1].astype(np.int64)
    nreal = int(registry.get(NUMBER_OF_REAL_CELLS)) - 1
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
        return np.asarray(registry.get(EDDY_VISCOSITY)) / Sc_t
    if CELL_EDDY_VISCOSITY_X in registry and CELL_EDDY_VISCOSITY_Y in registry:
        nx = np.asarray(registry.get(CELL_EDDY_VISCOSITY_X))
        ny = np.asarray(registry.get(CELL_EDDY_VISCOSITY_Y))
        nu_t_cell = np.sqrt(nx * nx + ny * ny)
        ef = np.asarray(registry.get(EDGE_FACE_CONNECTIVITY))
        f1 = ef[:, 0].astype(np.int64)
        f2 = ef[:, 1].astype(np.int64)
        nreal = int(registry.get(NUMBER_OF_REAL_CELLS)) - 1
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
    """Per-cell diffusion from a CSV (Phase F T2-C).

    CSV columns (no header required by position; first two used):
    ``cell_index, diffusion_coefficient``. Static values applied at
    all timesteps.
    """
    df = pd.read_csv(filepath)
    volume = registry.get(VOLUME)
    nface = int(volume.sizes['nface'])
    ntimes = int(volume.sizes['time'])
    D_cell = np.full(nface, default_value, dtype=np.float64)
    D_cell[df.iloc[:, 0].values.astype(int)] = df.iloc[:, 1].values
    D_cell_tv = np.broadcast_to(D_cell, (ntimes, nface)).copy()
    ef = np.asarray(registry.get(EDGE_FACE_CONNECTIVITY))
    f1 = ef[:, 0].astype(np.int64)
    f2 = ef[:, 1].astype(np.int64)
    nreal = int(registry.get(NUMBER_OF_REAL_CELLS)) - 1
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
    edge_vertical_area = registry.get(EDGE_VERTICAL_AREA)
    face_to_face_distance = registry.get(FACE_TO_FACE_DISTANCE)

    # Phase F T2-C dispatch. ``diffusion_method`` is stored as a
    # FloatVariable wrapping a flag (0=constant, 1=elder, 2=eddy,
    # 3=array) when set by the model; absent means constant.
    method_var = "diffusion_method"
    if method_var in registry:
        method_code = int(registry.get(method_var))
    else:
        method_code = 0  # constant

    if method_code == 0:  # constant
        diffusion_coefficient = registry.get(DIFFUSION_COEFFICIENT)
        diffusion_array = edge_vertical_area * diffusion_coefficient / face_to_face_distance
        return DataArrayVariable(diffusion_array)

    # Non-constant: build D_edge (shape (time, nedge) or (nedge,))
    if method_code == 1:  # elder
        alpha = float(registry.get("diffusion_alpha")) if "diffusion_alpha" in registry else 0.6
        D_edge = _calc_diffusion_elder(registry, alpha=alpha)
    elif method_code == 2:  # eddy_viscosity
        Sc_t = float(registry.get("diffusion_schmidt")) if "diffusion_schmidt" in registry else 1.0
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
    times = registry.get(VOLUME).time
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
    nface = len(registry.get(FACE_SURFACE_AREA)[FACES])
    ntime = len(registry.get(VOLUME)["time"])
    lookup_volumes = registry.get(LOOKUP_VOLUME)
    lookup_areas = registry.get(LOOKUP_WETTED_SURFACE_AREA)

    # fill null lookup values with the maximum
    # this will help the interpolation function work correctly for large values
    lookup_volumes = lookup_volumes.fillna(lookup_volumes.max(dim='index', skipna=True))
    lookup_areas = lookup_areas.fillna(lookup_areas.max(dim='index', skipna=True))

    # preallocate output array
    result = xr.DataArray(
        np.full((ntime, nface), np.nan),
        dims=[TIME, FACES],
        coords={
            TIME: registry.get(VOLUME)["time"],
            FACES:np.arange(nface),
        }
    )

    # loop through real faces, get wetted surface area for all timesteps
    for nf in lookup_areas.nface:
        volumes = registry.get(VOLUME).sel(nface=nf).values
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
        registry.get(WETTED_SURFACE_AREA) > 0,
        registry.get(VOLUME) / registry.get(WETTED_SURFACE_AREA),
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
    lookup_elev = registry.get(LOOKUP_ELEVATION)
    min_elev = lookup_elev.isel(index=0)
    return min_elev.reindex(nface=np.arange(len(registry.get(FACE_X))))


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
    depth = registry.get(WATER_SURFACE_ELEVATION) - min_elev
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
    maximum_depth = registry.get(WATER_SURFACE_ELEVATION) - min_elev
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
    eps_converged: float = 1e-12,
    max_iter: int = 5,
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

    nreal_count = int(registry.get(NUMBER_OF_REAL_CELLS))  # number of real cells
    nreal_attr = nreal_count - 1  # maximum real-cell index (streaming convention)

    V = np.asarray(registry.get(VOLUME))  # (ntime, nface)
    ef = np.asarray(registry.get(EDGE_FACE_CONNECTIVITY))
    f1 = ef[:, 0].astype(np.int64)
    f2 = ef[:, 1].astype(np.int64)
    n_face = V.shape[1] if V.ndim == 2 else int(max(f1.max(), f2.max()) + 1)
    n_edge = f1.shape[0]

    # Compute dt locally; the time coordinate is on ``VOLUME``.
    times = registry.get(VOLUME).time.values
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

    # Cell-graph Laplacian L = D D^T with a tiny Tikhonov ridge for
    # closed-sub-domain robustness. The ridge is several orders of
    # magnitude smaller than the smallest meaningful residual and does
    # not bias the solution.
    L = (D @ DT).tocsc()
    ridge = 1e-14 * np.maximum(L.diagonal().max(), 1.0)
    L = L + csc_matrix(
        (np.full(nreal_count, ridge, dtype=np.float64),
         (np.arange(nreal_count, dtype=np.int64),
          np.arange(nreal_count, dtype=np.int64))),
        shape=(nreal_count, nreal_count),
    )
    try:
        solver = splu(L)
    except Exception as exc:  # pragma: no cover - degenerate-mesh defense
        warnings.warn(
            "all_edges continuity correction: cell-Laplacian factorization "
            f"failed ({exc!r}). Skipping the correction; advection "
            "coefficient stays at the raw RAS face_flow.",
            stacklevel=3,
        )
        return

    worst_unconverged_cell = -1
    worst_unconverged_residual = 0.0
    worst_unconverged_t = -1

    for t in range(n_steps):
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
    flow = registry.get(FLOW_ACROSS_FACE)
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