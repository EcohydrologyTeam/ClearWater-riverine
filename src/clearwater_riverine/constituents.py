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
        # mesh: xr.Dataset,
        # flow_field_boundaries: Optional[pd.DataFrame] = None,
        # constituent_config: Optional[Dict] = None,
        # method: Optional[Literal['initialize', 'load']] = 'initialize',
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
            )
        )

        self.set_initial_conditions(
            registry=registry,
            start_datetime=start_datetime,
        )
        self.initialize_boundary_conditions(
            registry=registry,
        )

        # self.advection_mass_flux = np.zeros((len(mesh.time), len(mesh.nedge)))
        # self.diffusion_mass_flux = np.zeros((len(mesh.time), len(mesh.nedge)))
        # self.total_mass_flux = np.zeros((len(mesh.time), len(mesh.nedge)))
        # self.input_array = np.zeros((len(mesh.time), len(mesh.nface)))
        # # TODO: make units optional
        # if method == 'initialize':
        #     self.units = constituent_config['units']
        #     self.max_value = None
        #     self.min_value = None

            # # add to model mesh
            # mesh[self.name] = xr.DataArray(
            #     np.full(
            #         (len(mesh.time), len(mesh.nface)),
            #         np.nan
            #     ),
            #     dims = ('time', 'nface'),
            #     attrs = {
            #         'Units': f'{self.units}'
            #     }
            # )
        #     # define initial and boundary conditions
        #     self.set_initial_conditions(
        #         filepath=constituent_config['initial_conditions'],
        #         mesh=mesh,
        #     )
        #     self.set_boundary_conditions(
        #         filepath=constituent_config['boundary_conditions'],
        #         mesh=mesh,
        #         flow_field_boundaries=flow_field_boundaries,
        #     )

        #     # set up RHS matrix
        #     self.b = RHS(
        #         mesh=mesh,
        #         input_array=self.input_array,
        #     )
        # elif method == 'load':
        #     try:
        #         self.units = mesh[name].Units
        #     except AttributeError as err:
        #         warnings.warn(
        #             f'Constituent {self.name} does not have units defined',
        #             UserWarning       
        #         )

        #     self.set_value_range(mesh)


    def set_initial_conditions(
        self,
        registry: VariableRegistry,
        start_datetime: datetime,

    ):
        """Define cosntituetn initial conditions."""
        constituent = registry.get_at_time(self._name, start_datetime)
        initial = registry.get_at_time(f"{self._name}_initial", start_datetime)

        if isinstance(initial, xr.DataArray):
            constituent[:] =  (
                initial
                .rename({self.__initial_condition_spatial_field: 'nface'})  # Align to mesh coords
                .reindex(nface=constituent.nface)
                .data
            )
        elif isinstance(initial, (float, int)):
            constituent[:] = initial
        

    def initialize_boundary_conditions(
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

        # linear interpolation over time
        boundary = boundary.interp(
            time=target_time,
            method="linear"
        )
        # boundary = xr.merge([boundary, boundary_index])
        ghost_cells = edges_face2[boundary_index]
        domain_cells = edges_face1[boundary[BOUNDARY_FACE_INDEX]]


        # Assign to appropriate position in array
        constituent[[boundary_df['Time Index']], [boundary['Ghost Cell']]] = boundary[self._name]
    
    ## TODO: probably a more elegant way to do this
    def set_value_range(
        self,
        mesh: xr.Dataset
    ):
        self.max_value = int(mesh[self.name].sel(nface=slice(0, mesh.attrs[NUMBER_OF_REAL_CELLS])).max())
        self.min_value = int(mesh[self.name].sel(nface=slice(0, mesh.attrs[NUMBER_OF_REAL_CELLS])).min())

