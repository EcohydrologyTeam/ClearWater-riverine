import warnings

import numba
import pandas as pd
import numpy as np
import xarray as xr 

from clearwater_data.variables import VariableRegistry
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
    MAXIMUM_DEPTH,
    TIME,
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
    dist_data = np.sqrt((x1_coords - x2_coords)**2 + (y1_coords - y2_coords)**2)
    return dist_data


def calculate_edge_vertical_area(
    registry: VariableRegistry
):
    vertical_area = registry.get(FLOW_ACROSS_FACE) / registry.get(EDGE_VELOCITY)
    return vertical_area


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
    return diffusion_array


def calculate_change_in_time(
    registry: VariableRegistry
):
    times = registry.get(VOLUME).time
    dt = np.ediff1d(times)
    dt = dt / np.timedelta64(1, 's')
    dt = np.insert(dt, len(dt), np.nan)
    # mesh[variables.CHANGE_IN_TIME] = xr.DataArray(dt, dims=('time'), attrs={'Units': 's'})
    return dt


def calculate_wetted_surface_area(
    mesh: xr.Dataset
):
    """
    Calculate wetted surface area based on elevation-volume lookup table.
    """
    # Define required dimensions for lookup xarray
    nface = len(mesh[FACES])
    surface_area_lookup = mesh.attrs[VOLUME_ELEVATION_LOOKUP]
    lookup_max = surface_area_lookup.groupby(
        'Cell').count()['Wetted Surface Area'].max()

    surface_area_lookup['lookup'] = \
        surface_area_lookup.groupby('Cell').cumcount()

    # Pivot to wide format
    volume_wide = surface_area_lookup.pivot(
        index='Cell', columns='lookup', values='Volume'
        ).reindex(range(nface))
    area_wide = surface_area_lookup.pivot(
        index='Cell', columns='lookup', values='Wetted Surface Area'
        ).reindex(range(nface))

    # Convert to xarray.DataArray, filling missing values with nan
    # Within an xarray dataset
    lookup_dataset = xr.Dataset(
        {
            VOLUME: xr.DataArray(volume_wide.values, dims=('nface', 'lookup')),
            WETTED_SURFACE_AREA: xr.DataArray(
                area_wide.values,
                dims=('nface', 'lookup')
            ),
        },
        coords={
            'nface':  mesh[FACES].values,
            'lookup': np.arange(lookup_max)
        }
    )

    # fill null lookup values with the maximum
    # this will help the interpolation function work correctly for large values
    for variable in [VOLUME, WETTED_SURFACE_AREA]:
        lookup_dataset[variable] = lookup_dataset[variable].fillna(
            lookup_dataset[variable].max(dim='lookup', skipna=True)
        )

    # Preallocate output array
    result = xr.DataArray(
        np.full((mesh.sizes[TIME], mesh.sizes[FACES]), np.nan),
        dims=[TIME, FACES],
        coords={
            TIME: mesh[TIME],
            FACES: mesh[FACES],
        }
    )

    # Loop through faces, get wetted surface area for all timesteps
    for nf in mesh[FACES].values:
        volumes = mesh[VOLUME].sel(nface=nf).values
        lookup_volumes = lookup_dataset[VOLUME].sel(nface=nf).values
        lookup_wetted_surface_area = \
            lookup_dataset[WETTED_SURFACE_AREA].sel(nface=nf).values

        result[:,  nf] = np.interp(
            volumes,
            lookup_volumes,
            lookup_wetted_surface_area,
            left=lookup_wetted_surface_area[0],  # interp to lowermost value
            right=lookup_wetted_surface_area[-1],  # interp to largest value
        )

    # Convert result back to xarray.DataArray
    mesh[WETTED_SURFACE_AREA] = result

def calculate_average_depth(
    mesh: xr.Dataset     
):
    """Calculate average depth based on volume and wetted surface area."""
    # If wetted surface area does not exist, calculate it.
    if WETTED_SURFACE_AREA not in mesh.data_vars:
        calculate_wetted_surface_area(mesh)
    
    # Calculate average depth
    mesh[AVERAGE_DEPTH] = xr.where(
        mesh[WETTED_SURFACE_AREA] > 0,
        mesh[VOLUME] / mesh[WETTED_SURFACE_AREA],
        0
    )


def calculate_maximum_depth(
    mesh: xr.Dataset,
):
    """Calculate the maximum depth based on water surface elevation."""

    minimum_elevation = (
        mesh.attrs[VOLUME_ELEVATION_LOOKUP]
        .groupby("Cell")["Elevation"]
        .min()
        .to_xarray()
        .rename({"Cell": "nface"})
        .reindex(nface=mesh.nface) # volume elevation lookup only has real cells
    )

    mesh[MAXIMUM_DEPTH] = mesh[WATER_SURFACE_ELEVATION] - minimum_elevation



CALCULATED_VARIABLE_MAP = {
    FACE_TO_FACE_DISTANCE: calculate_distances_cell_centroids,
    EDGE_VERTICAL_AREA: calculate_edge_vertical_area,
    COEFFICIENT_TO_DIFFUSION_TERM: calculate_coeff_to_diffusion_term,
    CHANGE_IN_TIME: calculate_change_in_time,
    # WETTED_SURFACE_AREA: _calc_wetted_surface_area,
}