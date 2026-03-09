from typing import (
    Dict,
    Literal,
    Optional
)
from pathlib import Path
import warnings

import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from clearwater_data.variables import VariableRegistry, DataArrayVariable
from clearwater_riverine.linalg import RHS
from clearwater_riverine.variables import (
    BOUNDARY_CONDITION_LINE_ID,
    BOUNDARY_FACE_INDEX,
    BOUNDARY_NAME,
    EDGE_FACE_CONNECTIVITY,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)


class Constituent:
    """Constituent class."""
    def __init__(
        self,
        constituent_name: str,
        registry: VariableRegistry,
        initial_conditions: xr.DataArray,
        boundary_conditions: xr.DataArray,
        constituent_config: dict,
        start_datetime: datetime,
    ):
        self._name = constituent_name
        self.__units = constituent_config.get("units", None)
        self.__initial_condition_spatial_field = constituent_config["initial_conditions"]["data"].get(
            "spatial_field", "Cell_Index"  # Default to old config requriement
        )
        self.__boundary_condition_spatial_field = constituent_config["boundary_conditions"]["data"].get(
            "spatial_field", "RAS2D_TS_Name"  # Default to old config requriement
        )
        registry.register(
            f"{self._name}_initial",
            initial_conditions,
        )
        registry.register(
            f"{self._name}_boundary",
            boundary_conditions,
        )

        ## Initialize 
        self.register_constituent(registry)

        self.set_initial_conditions(
            registry=registry,
            start_datetime=start_datetime,
        )
        self.set_boundary_conditions(
            registry=registry,
        )

        self.rhs = RHS(
            registry=registry,
        )


    def register_constituent(
        self,
        registry):
        """Register constituent to variable registry."""
        # unregister if it already exists
        if self._name in registry:
            registry.unregister(self._name)

        # initialize
        registry.register(
            self._name,
            DataArrayVariable(
                xr.full_like(
                    registry.get(VOLUME),
                    np.nan
                )
                .rename(self._name)
                .assign_attrs({
                    'units': self.__units
                })
            ),
        )

    def reset_initial_conditions(
        self,
        registry: VariableRegistry,
        initial_conditions: xr.DataArray,
    ):
        """Used for chunking mode to reset initial conditions from final calculation."""
        registry.unregister(f"{self._name}_initial")
        registry.register(
            f"{self._name}_initial",
            DataArrayVariable(initial_conditions),
        )
        self.__initial_condition_spatial_field =  'nface'

    def set_initial_conditions(
        self,
        registry: VariableRegistry,
        start_datetime: datetime,

    ):
        """Define constituent initial conditions."""
        constituent = registry.get_at_time(self._name, start_datetime)
        initial = registry.get_at_time(f"{self._name}_initial", start_datetime)

        if isinstance(initial, xr.DataArray):
            registry.set_at_time(
                self._name,
                start_datetime,
                initial
                .rename({self.__initial_condition_spatial_field: 'nface'})  # Align to mesh coords
                .reindex(nface=constituent.nface)
                .data
            )
        elif isinstance(initial, (float, int)):
            registry.set_at_time(
                self._name,
                start_datetime,
                initial
            )
        

    def set_boundary_conditions(
        self,
        registry,
    ):
        """Define boundary conditions for the Constituent."""
        # retrieve necessary variables
        boundary = registry.get(f"{self._name}_boundary")
        constituent = registry.get(self._name)
        target_time = registry.get(self._name).time
        boundary_index = registry.get(BOUNDARY_FACE_INDEX)
        boundary_names = registry.get(BOUNDARY_NAME)
        edges_face1 = registry.get(EDGE_FACE_CONNECTIVITY).T[0]
        edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]

        # find cells associated with each cell
        ghost_cells = edges_face2[boundary_index]

        # linear interpolation over time
        if isinstance(boundary, xr.DataArray):
            boundary = boundary.interp(
                time=target_time,
                method="linear"
            )
            # reshape from (time, boundary_name) to (time, boundary_index)
            # then map boundary indices to their associated ghost cells
            boundary = boundary.sel(
                RAS2D_TS_Name=boundary_names
            ). assign_coords(
                nface = ghost_cells
            ).groupby(
                "nface"
            ).first()

            # reshape to the shape of our constituent array
            boundary_reindexed = boundary.reindex(nface=constituent.nface)

            # place the boundary conditions into the constituent array
            registry.set(
                self._name,
                xr.where(
                    boundary_reindexed.notnull(),
                    boundary_reindexed,
                    constituent
                )
            )
        elif isinstance(boundary, (float, int)):
            constituent.loc[dict(nface=ghost_cells)] = boundary


    ## TODO: probably a more elegant way to do this
    # def set_value_range(
    #     self,
    #     mesh: xr.Dataset
    # ):
    #     self.max_value = int(mesh[self.name].sel(nface=slice(0, mesh.attrs[NUMBER_OF_REAL_CELLS])).max())
    #     self.min_value = int(mesh[self.name].sel(nface=slice(0, mesh.attrs[NUMBER_OF_REAL_CELLS])).min())

