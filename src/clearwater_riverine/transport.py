from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix, linalg
from datetime import datetime, timedelta
import xarray as xr

from clearwater_riverine.linalg import LHS
from clearwater_riverine.variables import (
    EDGE_FACE_CONNECTIVITY,
    FLOW_ACROSS_FACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    WET_MASK,
)
from clearwater_riverine.constituents import Constituent
from clearwater_data.variables import VariableRegistry


def reconstruct_newly_wet(
    registry: VariableRegistry,
    current_time: datetime,
    time_step: timedelta,
    constituent_name: str,
    x_full: xr.DataArray,
    next_constituent_value: xr.DataArray,
) -> xr.DataArray:
    """Lift the c~0 artifact in newly-wet cells (Phase-D Unit B).

    A cell ``i`` is "newly wet" when it is dry at ``current_time`` and
    wet at ``current_time + time_step`` (using the ``WET_MASK`` field
    that Unit A registers). The implicit transport solve produces
    ``c[t+1, i] ~ 0`` for such cells whenever no advective inflow has
    yet been recorded at any of the cell's edges at time ``t``: the
    HEC-RAS staggered-time pattern is that face flow into a newly-wet
    cell is reported at ``t+1``, not at ``t``. The artifact is bounded
    in mass impact but produces visible non-physical low concentrations
    in plotted output.

    This routine replaces ``x_full[i]`` for each newly-wet cell with an
    inflow-weighted average of upstream concentrations. The gather
    cascade tries three sources in order and accepts the first one that
    yields a strictly positive average: signed gather on ``adv[t]``,
    signed gather on ``adv[t+1]`` (the staggered-time fallback), then
    equal-weight gather over any wet-at-t neighbour or ghost cell. Two
    iterations let wetting fronts propagate (a newly-wet cell whose
    only wet neighbour is itself newly-wet and reconstructed in the
    same step). "Only lift, never lower": if the solver already
    produced a positive value, it is kept.

    Opt-in via Unit A's ``wet_dry_metric``. When ``WET_MASK`` is not in
    the registry, this routine is a no-op and returns ``x_full``
    unchanged, so callers that did not opt in see the existing
    behaviour.

    Args:
        registry: The variable registry holding VOLUME, WET_MASK,
            FLOW_ACROSS_FACE, EDGE_FACE_CONNECTIVITY,
            NUMBER_OF_REAL_CELLS, and the constituent's own array.
        current_time: The "t" timestamp -- volume, mask, and advection
            at this time gate which cells qualify and choose upstream
            neighbours.
        time_step: The simulation timestep. ``current_time + time_step``
            is the "t+1" timestamp.
        constituent_name: Name of the constituent being reconstructed.
            Its concentration at ``current_time`` is consulted for the
            wet-to-dry "gather concentration" swap; its boundary-cell
            values at ``current_time + time_step`` (read from
            ``next_constituent_value``) seed the ghost contributions.
        x_full: The post-solve concentration DataArray for ``t+1``,
            indexed over ``nface``. Modified in place and returned.
        next_constituent_value: The pre-existing constituent slice at
            ``current_time + time_step``. On canonical, boundary-cell
            (ghost) concentrations are placed here by
            ``set_boundary_conditions``; this routine reads them out
            for ghost-cell contributions.

    Returns:
        ``x_full`` with newly-wet entries lifted where appropriate.
    """
    # Opt-in: no WET_MASK in registry means Unit A was not enabled,
    # which means the rest of the wet-dry plumbing is also absent.
    # No-op is the safe default.
    if WET_MASK not in registry:
        return x_full

    next_time = current_time + time_step
    nreal = int(registry.get(NUMBER_OF_REAL_CELLS))

    # Identify newly-wet cells via the mask (spec §5 criterion).
    wet_t = np.asarray(
        registry.get_at_time(WET_MASK, current_time), dtype=bool
    )[:nreal]
    wet_t1 = np.asarray(
        registry.get_at_time(WET_MASK, next_time), dtype=bool
    )[:nreal]
    newly_wet = np.where(~wet_t & wet_t1)[0]
    if newly_wet.size == 0:
        return x_full

    x_arr = np.asarray(x_full.values)

    # Build a gather-concentration vector so the upstream value carried
    # by water arriving at a newly-wet cell is c[t] of the wet-at-t
    # donor, not the post-pin 0 the solver writes for wet-to-dry cells
    # (those will exist once Unit C lands the LHS rule-1 pinning; this
    # is forward-compatible).
    wet_to_dry = wet_t & ~wet_t1
    if wet_to_dry.any():
        c_t = np.asarray(
            registry.get_at_time(constituent_name, current_time)
        )[:nreal]
        gather_conc = x_arr[:nreal].copy()
        gather_conc[wet_to_dry] = c_t[wet_to_dry]
    else:
        gather_conc = x_arr[:nreal]

    adv_t = np.asarray(registry.get_at_time(FLOW_ACROSS_FACE, current_time))
    try:
        adv_t1 = np.asarray(
            registry.get_at_time(FLOW_ACROSS_FACE, next_time)
        )
    except (KeyError, ValueError):
        adv_t1 = None
    ef = np.asarray(registry.get(EDGE_FACE_CONNECTIVITY))
    bc_input = np.asarray(next_constituent_value.values)

    # Track first-pass reconstructions so a second pass can use them as
    # upstream sources for wetting fronts.
    reconstructed = np.zeros(nreal, dtype=bool)
    neighbor_wet = wet_t  # mask-active path; canonical always has WET_MASK here

    def _gather_signed(i, adv):
        face1_in = np.where((ef[:, 1] == i) & (adv > 0))[0]
        face2_in = np.where((ef[:, 0] == i) & (adv < 0))[0]
        weights: list = []
        concs: list = []
        for e in face1_in:
            j = int(ef[e, 0])
            if j < nreal:
                if neighbor_wet[j] or reconstructed[j]:
                    weights.append(float(adv[e]))
                    concs.append(float(gather_conc[j]))
            else:
                weights.append(float(adv[e]))
                concs.append(float(bc_input[j]))
        for e in face2_in:
            j = int(ef[e, 1])
            if j < nreal:
                if neighbor_wet[j] or reconstructed[j]:
                    weights.append(float(-adv[e]))
                    concs.append(float(gather_conc[j]))
            else:
                weights.append(float(-adv[e]))
                concs.append(float(bc_input[j]))
        return weights, concs

    def _gather_any_wet_neighbor(i):
        edges = np.where((ef[:, 0] == i) | (ef[:, 1] == i))[0]
        weights: list = []
        concs: list = []
        for e in edges:
            f1, f2 = int(ef[e, 0]), int(ef[e, 1])
            j = f2 if f1 == i else f1
            if j < nreal:
                if neighbor_wet[j] or reconstructed[j]:
                    weights.append(1.0)
                    concs.append(float(gather_conc[j]))
            else:
                weights.append(1.0)
                concs.append(float(bc_input[j]))
        return weights, concs

    for _ in range(2):
        for i in newly_wet:
            candidate = None
            sources = [_gather_signed(i, adv_t)]
            if adv_t1 is not None:
                sources.append(_gather_signed(i, adv_t1))
            sources.append(_gather_any_wet_neighbor(i))
            for weights, concs in sources:
                if not weights:
                    continue
                w = np.asarray(weights)
                if w.sum() <= 0:
                    continue
                avg = float(np.average(concs, weights=w))
                if avg > 0:
                    candidate = avg
                    break
            if candidate is not None and candidate > x_arr[i]:
                x_arr[i] = candidate
                reconstructed[i] = True
                gather_conc[i] = candidate

    # Write the lifted values back into the DataArray.
    x_full.values[:nreal] = x_arr[:nreal]
    return x_full

class TransportEngine:
    def __init__(self, registry: VariableRegistry):
        # initialize left hand side of transport equation
        self.lhs = LHS(registry)

    def run(
        self,
        registry: VariableRegistry,
        current_time: datetime,
        time_step: timedelta,
        constituents: dict[str, Constituent],
        mass_flux_calculation: bool,
    ):
        """Run the transport engine."""
        # update the left hand side of the matrix
        self.lhs.update_values(
            registry,
            current_time,
            time_step,
        )

        # define compressed sparse row matrix for LHS
        real_cell_count = registry.get(NUMBER_OF_REAL_CELLS)
        A = csr_matrix(
            (self.lhs.coefficients, (self.lhs.rows, self.lhs.columns)),
            shape = (real_cell_count, real_cell_count)
        )

        # loop through all constituents
        for constituent_name, constituent in constituents.items():
            constituent_value = registry.get_at_time(constituent_name, current_time)
            next_constituent_value = registry.get_at_time(constituent_name, current_time + time_step)
            # update right hand side of the matrix
            constituent.rhs.update_values(
                registry=registry,
                current_time=current_time,
                time_step=time_step,
                constituent_name=constituent_name,
            )
        
            # solve
            x = linalg.spsolve(A, constituent.rhs.values)
            x_full = xr.DataArray(np.zeros(constituent_value.shape), coords=constituent_value.coords)
            x_full[:len(x)] = x

            # Phase-D Unit B: lift the c~0 newly-wet artifact (opt-in
            # via Unit A's wet_dry_metric; no-op when WET_MASK is
            # absent from the registry).
            x_full = reconstruct_newly_wet(
                registry=registry,
                current_time=current_time,
                time_step=time_step,
                constituent_name=constituent_name,
                x_full=x_full,
                next_constituent_value=next_constituent_value,
            )

            # update the value in the registry
            mask = np.isnan(next_constituent_value)
            registry.set_at_time(
                constituent_name,
                current_time + time_step,
                next_constituent_value.where(~mask, other=x_full)
            )

        
    