import warnings

import numba
import pandas as pd
import numpy as np
import xarray as xr 

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.variables import (
    AVERAGE_DEPTH,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    DIFFUSION_COEFFICIENT,
    EDGE_FACE_CONNECTIVITY,
    EDGE_VERTICAL_AREA,
    EDGE_VELOCITY,
    FACES,
    FACE_TO_FACE_DISTANCE,
    FACE_X,
    FACE_Y,
    FLOW_ACROSS_FACE,
    LOOKUP_ELEVATION,
    LOOKUP_VOLUME,
    LOOKUP_WETTED_SURFACE_AREA,
    MAXIMUM_DEPTH,
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


def calculate_coeff_to_diffusion_term(
        registry: VariableRegistry,
    ) -> np.array:
    """ Calculate the coefficient to the diffusion term. 

    For each edge, this is calculated as:
    (Edge vertical area * diffusion coefficient) / (distance between cells) 
    
    Args:
        registry: VariableRegistry

    Returns:
        diffusion_array (np.array):     Array of diffusion coefficients associated with each edge

    """
    edge_vertical_area = registry.get(EDGE_VERTICAL_AREA)
    face_to_face_distance = registry.get(FACE_TO_FACE_DISTANCE)
    diffusion_coefficient = registry.get(DIFFUSION_COEFFICIENT)

    diffusion_array = edge_vertical_area * diffusion_coefficient / face_to_face_distance
    return DataArrayVariable(diffusion_array)


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


def calculate_maximum_depth(
    registry: VariableRegistry
):
    """Calculate the maximum depth based on water surface elevation."""
    minimum_elevation = registry.get(LOOKUP_ELEVATION)
    # volume elevation lookup only has real cells: expand dimensions to include all cells
    minimum_elevation = minimum_elevation.reindex(nface=np.arange(len(registry.get(FACE_X))))
    maxiumum_depth = registry.get(WATER_SURFACE_ELEVATION) - minimum_elevation

    return DataArrayVariable(maxiumum_depth)


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


CALCULATED_VARIABLE_MAP = {
    FACE_TO_FACE_DISTANCE: calculate_distances_cell_centroids,
    EDGE_VERTICAL_AREA: calculate_edge_vertical_area,
    COEFFICIENT_TO_DIFFUSION_TERM: calculate_coeff_to_diffusion_term,
    CHANGE_IN_TIME: calculate_change_in_time,
    WETTED_SURFACE_AREA: calculate_wetted_surface_area,
    AVERAGE_DEPTH: calculate_average_depth,
    MAXIMUM_DEPTH: calculate_maximum_depth,
}

CALCULATED_VARIABLE_DEPENDENCIES = {
    AVERAGE_DEPTH: [WETTED_SURFACE_AREA],
}