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
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    FLOW_ACROSS_FACE,
    EDGE_FACE_CONNECTIVITY,
    NEDGE,
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)


def _validate_constituent_values(
    values,
    *,
    constituent_name: str,
    source_label: str,
    raise_on_nan: bool = True,
    warn_on_negative: bool = True,
) -> None:
    """Validate a constituent IC/BC source array before it enters the registry.

    Phase F (2026-05-21) T2-D: catches the silent-propagation bug in
    both repos where a malformed CSV (NaN rows, negative values) flowed
    through the transport solve and produced non-physical output with
    no diagnostic.

    Args:
        values: Scalar (int/float), numpy array, or xr.DataArray.
        constituent_name: For diagnostic messages.
        source_label: "initial_conditions" or "boundary_conditions".
        raise_on_nan: When True (default), raise ValueError on any
            NaN in the source. Set False to allow NaN through (e.g., a
            data source that uses NaN as a sentinel for "use default").
        warn_on_negative: When True (default), emit a UserWarning on
            any negative value. Most water-quality constituents
            (concentrations, biomass, temperature in deg C) are
            physically non-negative; an early warning catches CSV
            sign errors or interpolation artifacts.
    """
    if isinstance(values, (int, float)):
        arr = np.asarray([values], dtype=np.float64)
    elif isinstance(values, xr.DataArray):
        arr = np.asarray(values.values)
    else:
        arr = np.asarray(values)
    if arr.size == 0:
        return
    if raise_on_nan:
        nan_mask = np.isnan(arr)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            raise ValueError(
                f"Constituent {constituent_name!r}: {source_label} source "
                f"contains {n_nan} NaN value(s) (of {arr.size} total). "
                "NaN in IC/BC inputs propagates silently through the "
                "transport solve and produces non-physical output. Fix "
                "the source CSV/array (drop missing rows, fill via "
                "interpolation, or supply a scalar default), or pass "
                "``raise_on_nan=False`` if NaN is intentional."
            )
    if warn_on_negative:
        try:
            neg_mask = arr < 0
            n_neg = int(neg_mask.sum())
        except TypeError:
            n_neg = 0  # non-numeric (e.g. boolean wet mask)
        if n_neg > 0:
            min_val = float(np.nanmin(arr))
            warnings.warn(
                f"Constituent {constituent_name!r}: {source_label} source "
                f"contains {n_neg} negative value(s) (of {arr.size} total; "
                f"min = {min_val:.4g}). Most water-quality constituents "
                "are physically non-negative; check for sign errors in "
                "the source CSV or interpolation artifacts.",
                stacklevel=3,
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
        # Phase-D Unit D1: intensive vs extensive flag. Intensive
        # scalars (e.g. water temperature) skip the LHS rule-3
        # donor-diagonal amendment on wet-dry edges (which would
        # otherwise produce spurious cooling), skip the IC-zeroing
        # pass (zeroing a temperature in a sub-threshold cell is
        # non-physical), and skip the end-of-run mass_lost_to_dry
        # warning (the BC inflow MASS denominator has the wrong units
        # for an intensive scalar). Default ``False`` preserves the
        # existing concentration-species behaviour. Consumed by
        # ``LHS.update_values``, ``zero_dry_initial_conditions``, and
        # ``emit_mass_loss_warning`` via ``getattr(c, "is_intensive",
        # False)``.
        self.is_intensive: bool = bool(
            constituent_config.get("is_intensive", False)
        )
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
        registry: VariableRegistry
    ):
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
                }),
                space_dimension=NFACE,
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
            DataArrayVariable(
                initial_conditions,
                space_dimension=NFACE,
            ),
        )
        self.__initial_condition_spatial_field = NFACE

    def set_initial_conditions(
        self,
        registry: VariableRegistry,
        start_datetime: datetime,

    ):
        """Define constituent initial conditions."""
        constituent = registry.get_at_time(self._name, start_datetime)
        initial = registry.get_at_time(f"{self._name}_initial", start_datetime)

        # Phase F (2026-05-21) T2-D: validate IC source before it enters
        # the registry. Raises on NaN, warns on negative values.
        _validate_constituent_values(
            initial,
            constituent_name=self._name,
            source_label="initial_conditions",
        )

        if isinstance(initial, xr.DataArray):
            registry.set_at_time(
                self._name,
                start_datetime,
                initial
                .rename({self.__initial_condition_spatial_field: NFACE})  # Align to mesh coords
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
        registry: VariableRegistry,
    ):
        """Define boundary conditions for the Constituent."""
        # retrieve necessary variables
        boundary = registry.get(f"{self._name}_boundary")
        constituent = registry.get(self._name)

        # Phase F (2026-05-21) T2-D: validate BC source before it enters
        # the registry. Raises on NaN, warns on negative values. BC
        # interpolation can introduce NaN at the simulation window
        # boundaries if the source time series doesn't cover the
        # window; catching that early prevents silent diluton of the
        # ghost-cell BC injection.
        _validate_constituent_values(
            boundary,
            constituent_name=self._name,
            source_label="boundary_conditions",
        )
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
            ).assign_coords(
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

        # TODO: does there need to be a custom set method to set at custom locations?
        elif isinstance(boundary, (float, int)):
            constituent.loc[dict(nface=ghost_cells)] = boundary
    

    def get_minimum_value(self, registry):
        constituent = registry.get(self._name)
        return constituent.min()

    def get_maximum_value(self, registry):
        constituent = registry.get(self._name)
        return constituent.max()


    def _calculate_mass_flux(
        self,
        registry: VariableRegistry,
    ):
        # advection / diffusion coefficients from n timestep are used
        advection_coefficient = registry.get(FLOW_ACROSS_FACE)[:-1]
        diffusion_coefficient = registry.get(COEFFICIENT_TO_DIFFUSION_TERM)[:-1]
        edges_face1 = registry.get(EDGE_FACE_CONNECTIVITY).T[0]
        edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]

        # concentrations from n+1 timestep are used:
        # indexing here shifts the data accordingly 
        negative_condition = advection_coefficient < 0
        parent_concentration = registry.get(self._name)[1:].sel(nface=edges_face1)
        neighbor_concentration = registry.get(self._name)[1:].sel(nface=edges_face2)

        # coerce times so the xarray where function will work
        # think it makes most sense to assign the coordinates of advection coefficient
        # at n timestep, mass flux is calculated to then get the concentration at the next timestep
        parent_concentration = parent_concentration.assign_coords(time=advection_coefficient.time)
        neighbor_concentration = neighbor_concentration.assign_coords(time=advection_coefficient.time)

        advection_mass_flux = xr.where(
            negative_condition,
            advection_coefficient * neighbor_concentration,
            advection_coefficient * parent_concentration
        ) * registry.get(CHANGE_IN_TIME)

        diffusion_mass_flux = diffusion_coefficient * \
            (neighbor_concentration - parent_concentration) * \
            registry.get(CHANGE_IN_TIME)

        total_mass_flux = advection_mass_flux + diffusion_mass_flux

        # overwrite=True: in chunked mode this is recomputed every chunk on
        # the chunk-resident window. The whole-run reduction is folded into
        # the model's cross-chunk accumulator at each __finalize_chunk
        # (Phase-C C3a), so only the current chunk's flux needs to be
        # resident here. In non-chunked mode this is the single registration
        # (overwrite is a no-op on first register).
        registry.register(
            f"{self._name}_mass_flux",
            DataArrayVariable(total_mass_flux, space_dimension=NEDGE),
            overwrite=True,
        )
