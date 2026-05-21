from typing import Optional, Tuple
import warnings
import numpy as np
from scipy.sparse import csr_matrix, linalg
from datetime import datetime, timedelta
import xarray as xr

from clearwater_riverine.linalg import LHS
from clearwater_riverine.variables import (
    CHANGE_IN_TIME,
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


def drain_newly_dry(
    registry: VariableRegistry,
    current_time: datetime,
    time_step: timedelta,
    constituent_name: str,
) -> Tuple[np.ndarray, float]:
    """Pre-solve wet->dry mass handoff (Phase-D Unit C-beta).

    Symmetric counterpart to :func:`reconstruct_newly_wet`. For each
    cell ``i`` with ``WET_MASK[t] = True`` and ``WET_MASK[t+1] = False``
    (a wet->dry transition):

      1. For each outflow face from ``i`` to a wet-at-t+1 neighbour
         ``j``, add ``f_ij * c_i[t]`` to ``drain_source[j]``. Units are
         mass per unit time; the implicit solve integrates over ``dt``.
      2. Donor mass not carried by face flow to wet neighbours --
         phantom volume that disappears from ``i`` without crossing any
         face (RAS continuity residual: infiltration, evaporation,
         hydraulic time-step truncation), plus volume flowed to a
         dry-at-t+1 neighbour (which drains itself on its own
         iteration), plus volume that left via a ghost face (already
         accounted for by the existing BC outflow term) -- is returned
         as ``lost`` for the caller to accumulate into
         ``mass_lost_to_dry``.

    The LHS rule-3 amendment zeros the (i, j) advection coupling for
    wet-dry edges, so this drain rate provides the only path for
    ``i -> j`` mass transfer. The cell's own row is pinned to identity
    with RHS = 0 by the LHS rule-1 contribution, so no post-solve
    overwrite of cell ``i`` is required.

    Opt-in via Unit A's ``wet_dry_metric``. When ``WET_MASK`` is not in
    the registry, this routine returns ``(zeros(nreal), 0.0)`` so the
    legacy non-mask code path stays free of overhead.

    Args:
        registry: The variable registry holding ``VOLUME``, ``WET_MASK``,
            ``FLOW_ACROSS_FACE``, ``EDGE_FACE_CONNECTIVITY``,
            ``NUMBER_OF_REAL_CELLS``, ``CHANGE_IN_TIME``, and the
            constituent's own concentration array.
        current_time: The ``t`` timestamp. Volume, mask, advection, and
            constituent concentration at this time gate which cells
            qualify and supply the donor concentration.
        time_step: The simulation timestep. ``current_time + time_step``
            is the ``t+1`` timestamp; the mask there decides which
            cells transition wet->dry.
        constituent_name: Name of the constituent being drained. Its
            concentration at ``current_time`` is the donor value
            ``c_i[t]`` carried by each outflowing face.

    Returns:
        drain_source: shape ``(nreal,)``. Per-cell mass-rate
            contribution to be added to the RHS before ``spsolve``.
            Only entries corresponding to wet recipients of a wet->dry
            donor are non-zero.
        lost: scalar; the per-step mass that could not be apportioned
            to a wet face neighbour. The caller adds this to
            ``mass_lost_to_dry`` for the constituent.
    """
    nreal = int(registry.get(NUMBER_OF_REAL_CELLS))
    drain_source = np.zeros(nreal)

    if WET_MASK not in registry:
        return drain_source, 0.0

    next_time = current_time + time_step
    wet_t = np.asarray(
        registry.get_at_time(WET_MASK, current_time), dtype=bool
    )
    wet_t1 = np.asarray(
        registry.get_at_time(WET_MASK, next_time), dtype=bool
    )

    # Trigger set: cells wet at t, dry at t+1 -- restricted to real cells.
    going_dry = np.flatnonzero(wet_t & ~wet_t1)
    going_dry = going_dry[going_dry < nreal]
    if going_dry.size == 0:
        return drain_source, 0.0

    adv_t = np.asarray(registry.get_at_time(FLOW_ACROSS_FACE, current_time))
    V_t = np.asarray(registry.get_at_time(VOLUME, current_time))[:nreal]
    c_t = np.asarray(
        registry.get_at_time(constituent_name, current_time)
    )[:nreal]
    # Phase F (2026-05-21): use the caller's scalar time_step instead
    # of float(registry.get(CHANGE_IN_TIME)) so the array-dt path
    # (allowed by the relaxed cadence guard on non-uniform RAS HDFs)
    # does not raise. The current-step dt is what the drain integrates
    # against.
    dt_sec = float(time_step.total_seconds())
    ef = np.asarray(registry.get(EDGE_FACE_CONNECTIVITY))  # (nedge, 2)

    lost = 0.0
    # Per-cell loop. ``going_dry`` is typically a handful of cells per
    # timestep (boundary of the wet region), so the explicit loop is
    # fine and keeps the apportionment logic readable.
    for i in going_dry:
        i = int(i)
        edges = np.where((ef[:, 0] == i) | (ef[:, 1] == i))[0]
        if edges.size == 0:
            # Isolated cell -- no path for the mass to go.
            lost += float(V_t[i] * c_t[i])
            continue

        # For each edge incident on i: outflow magnitude (from i) and
        # the neighbour index.
        f_to_wet_neighbor = []
        neighbors = []
        for e in edges:
            e = int(e)
            f1 = int(ef[e, 0])
            f2 = int(ef[e, 1])
            if f1 == i:
                j = f2
                f_out = float(adv_t[e])   # adv > 0 means face1 -> face2
            else:
                j = f1
                f_out = float(-adv_t[e])  # adv < 0 means face2 -> face1
            if f_out <= 0:
                continue  # inflow or zero on this edge
            # Skip wet-dry routes to dry neighbours: routing to a
            # neighbour that is itself going dry would recreate the
            # artifact this pass exists to eliminate.
            if j < nreal and not wet_t1[j]:
                continue
            # Ghost neighbour outflow is already accounted for by the
            # existing ghost-cell outflow term; do not double-credit.
            if j >= nreal:
                continue
            f_to_wet_neighbor.append(f_out)
            neighbors.append(j)

        total_f = sum(f_to_wet_neighbor)
        c_i_val = float(c_t[i])
        M_i = float(V_t[i] * c_i_val)
        if total_f > 0:
            # Transfer only the mass that actually flows through faces
            # to wet neighbours: per-edge mass-rate = f_ij * c_i. Any
            # donor mass not carried by physical face flow to a wet
            # neighbour is routed to ``mass_lost_to_dry``. This is the
            # physically honest accounting: mass goes with the water
            # that carried it, and water that vanished outside the face
            # network cannot deposit mass into a downstream wet cell.
            for f_out, j in zip(f_to_wet_neighbor, neighbors):
                drain_source[j] += f_out * c_i_val
            mass_to_wet_via_faces = total_f * dt_sec * c_i_val
            unaccounted = M_i - mass_to_wet_via_faces
            if unaccounted > 0:
                lost += unaccounted
        else:
            # No outflow to wet neighbours -- entire donor mass is lost.
            # Includes the all-neighbours-dry and ghost-only-outflow
            # cases.
            lost += M_i

    return drain_source, lost


def zero_dry_initial_conditions(
    registry: VariableRegistry,
    constituents: dict,
    current_time: datetime,
) -> dict:
    """Zero IC mass loaded into sub-threshold cells (Phase-D Unit C-gamma).

    For each extensive constituent, cells with
    ``WET_MASK[current_time] = False`` carry zero concentration after
    this call. Any IC mass loaded into a sub-threshold cell at
    ``current_time`` is returned in the per-constituent total so the
    caller can fold it into ``mass_lost_to_dry``.

    **Intensive scalars** (e.g. water temperature, indicated by
    ``constituent.is_intensive = True``) are skipped: zeroing a
    temperature to ``T = 0`` in a sub-threshold cell is non-physical
    (represents "ice cold" rather than "no value") and can cascade
    through coupled physics when the cell becomes wet later. Intensive
    constituents keep their IC values in sub-threshold cells. The
    ``is_intensive`` attribute is consulted via ``getattr`` with a
    default of ``False``, so this routine is forward-compatible with
    Unit D's introduction of the flag.

    Opt-in via Unit A's ``wet_dry_metric``. When ``WET_MASK`` is not in
    the registry, this routine returns an empty dict so the legacy
    code path is unchanged.

    The caller decides whether to invoke this routine: it is **not**
    auto-invoked by ``TransportEngine``. On the fork the IC zeroing is
    gated by an explicit ``zero_dry_initial_conditions`` model-level
    flag (default off) introduced in Unit D; the canonical port keeps
    that gate at the call site rather than embedding it in the engine.

    Args:
        registry: The variable registry holding ``WET_MASK``,
            ``VOLUME``, and each constituent's array.
        constituents: ``dict[str, Constituent]`` mapping constituent
            name to the canonical ``Constituent`` object. The object
            is consulted via ``getattr(c, "is_intensive", False)`` to
            decide whether to skip.
        current_time: The simulation-start timestamp at which the IC
            mask and IC concentrations are read.

    Returns:
        ``dict[str, float]`` mapping constituent name to the IC mass
        that was zeroed out of sub-threshold cells. Only constituents
        with non-zero loss appear in the dict.
    """
    if WET_MASK not in registry:
        return {}

    wet0 = np.asarray(
        registry.get_at_time(WET_MASK, current_time), dtype=bool
    )
    if wet0.all():
        return {}

    nreal = int(registry.get(NUMBER_OF_REAL_CELLS))
    V0_full = np.asarray(registry.get_at_time(VOLUME, current_time))
    dry_mask_real = ~wet0[:nreal]
    if not dry_mask_real.any():
        return {}

    lost_by_name: dict = {}
    for name, constituent in constituents.items():
        if getattr(constituent, "is_intensive", False):
            # Intensive scalars keep their IC values; see docstring.
            continue
        c0_da = registry.get_at_time(name, current_time)
        c0_full = np.asarray(c0_da)
        # Real-cell IC mass loaded into sub-threshold cells.
        ic_mass_lost = float(
            np.sum(V0_full[:nreal][dry_mask_real] * c0_full[:nreal][dry_mask_real])
        )
        if ic_mass_lost > 0:
            lost_by_name[name] = ic_mass_lost
        # Zero the dry real cells in place via set_at_time. Ghost cells
        # are left untouched -- they carry boundary-condition values.
        new_c0 = c0_da.copy()
        # Use boolean indexing on the underlying array; preserve coords.
        full_mask = np.zeros(c0_full.shape, dtype=bool)
        full_mask[:nreal] = dry_mask_real
        new_vals = new_c0.values.copy()
        new_vals[full_mask] = 0.0
        new_c0.values[:] = new_vals
        registry.set_at_time(name, current_time, new_c0)

    return lost_by_name


def emit_mass_loss_warning(
    mass_lost_to_dry: dict,
    constituents: dict,
    threshold: Optional[float] = 0.01,
) -> None:
    """End-of-run wet-dry mass-loss warning (Phase-D Unit C-gamma).

    Compares per-constituent total ``mass_lost_to_dry`` against
    ``threshold * bc_inflow_mass`` and emits ``warnings.warn`` for each
    extensive constituent that breaches. Constituents that lost mass
    but had zero BC inflow warn unconditionally (typical signal: IC
    mass loaded into sub-threshold cells and zeroed by
    :func:`zero_dry_initial_conditions`).

    No-op cases:
      - ``threshold is None`` -- the user explicitly disabled the
        warning.
      - ``mass_lost_to_dry`` is empty -- no losses recorded (the
        Unit-A opt-out path: ``WET_MASK`` not in registry).
      - A constituent's recorded losses sum to zero.
      - A constituent flagged ``is_intensive`` -- the warning's
        denominator (BC inflow MASS) has the wrong units for an
        intensive scalar like temperature.

    Args:
        mass_lost_to_dry: ``dict[str, list[float] | float | ndarray]``
            mapping constituent name to per-step loss entries. ``np.sum``
            yields the total. Typically ``TransportEngine.mass_lost_to_dry``
            populated by Unit C-beta plus any IC contribution from
            :func:`zero_dry_initial_conditions` folded in by the caller.
        constituents: ``dict[str, Constituent]`` mapping constituent
            name to the canonical ``Constituent`` object. The object's
            ``rhs.bc_inflow_mass`` accumulator (a list of per-step
            inflow masses, populated by ``RHS._ghost_cell``) supplies
            the denominator; ``is_intensive`` decides whether to skip.
        threshold: Fraction of total BC inflow above which a warning
            is emitted. Defaults to ``0.01`` (1%). Pass ``None`` to
            disable.
    """
    if threshold is None:
        return
    if not mass_lost_to_dry:
        return
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError(
            "threshold must be >= 0 or None, "
            f"got {threshold}"
        )
    for name, lost_entries in mass_lost_to_dry.items():
        total_lost = float(np.sum(np.asarray(lost_entries, dtype=float)))
        if total_lost <= 0:
            continue
        constituent = constituents.get(name)
        if constituent is None or not hasattr(constituent, "rhs"):
            continue
        if getattr(constituent, "is_intensive", False):
            # Intensive scalars (e.g. temperature) should never reach
            # this branch because IC zeroing, drain mass logging, and
            # the post-solve wet-dry leak diagnostic all skip them.
            # Guard anyway: the BC inflow MASS denominator has the
            # wrong units for an intensive scalar.
            continue
        bc_inflow_mass = getattr(constituent.rhs, "bc_inflow_mass", [])
        total_inflow = float(
            np.sum(np.asarray(bc_inflow_mass, dtype=float))
        )
        if total_inflow <= 0:
            # No BC inflow to compare against -- warn unconditionally
            # if any mass was lost (otherwise the loss is silent).
            warnings.warn(
                f"Constituent {name!r}: {total_lost:.4g} mass units "
                f"routed to mass_lost_to_dry with zero BC inflow over "
                f"the run. Likely IC mass loaded into sub-threshold "
                f"cells. Review wet_dry_threshold or IC inputs.",
                UserWarning,
                stacklevel=2,
            )
            continue
        fraction = total_lost / total_inflow
        if fraction > threshold:
            warnings.warn(
                f"Constituent {name!r}: {total_lost:.4g} mass units "
                f"({100 * fraction:.2f}% of BC inflow {total_inflow:.4g}) "
                f"routed to mass_lost_to_dry, exceeding the "
                f"{100 * threshold:.2f}% threshold. Indicates wet->dry "
                f"events with no wet outflow path or IC mass loaded "
                f"into sub-threshold cells. Tighten wet_dry_threshold "
                f"or review the source of the loss.",
                UserWarning,
                stacklevel=2,
            )


class TransportEngine:
    def __init__(
        self,
        registry: VariableRegistry,
        reconstruct_newly_wet: bool = True,
    ):
        # initialize left hand side of transport equation. ``self.lhs``
        # remains the extensive LHS (``is_intensive=False``). When any
        # constituent in a ``run()`` call is intensive, the engine
        # lazily builds a second LHS keyed by ``is_intensive=True``
        # below.
        self.lhs = LHS(registry)
        self._lhs_intensive: LHS | None = None
        # Phase-D Unit C-beta: per-constituent mass-loss accumulator.
        # Lazily populated -- one list entry appended per ``run()`` call
        # that produces a non-zero loss for that constituent. Stays an
        # empty dict for runs that do not opt into the wet/dry mask
        # (Unit A), so the legacy code path is observable as an empty
        # accumulator without any extra branching at every step.
        self.mass_lost_to_dry: dict[str, list] = {}
        # Phase F (2026-05-21): opt-out for the newly-wet-cell
        # reconstruction pass. See ClearwaterRiverine.__init__ for the
        # rationale; this engine-level flag is the propagation point.
        self._reconstruct_newly_wet: bool = bool(reconstruct_newly_wet)

    def run(
        self,
        registry: VariableRegistry,
        current_time: datetime,
        time_step: timedelta,
        constituents: dict[str, Constituent],
        mass_flux_calculation: bool,
    ):
        """Run the transport engine."""
        # Phase-D Unit D1: build the LHS matrices once per
        # ``is_intensive`` flag in use this step. The extensive LHS
        # (``self.lhs``) is always rebuilt -- it is the common case and
        # the legacy/no-wet-mask path. The intensive LHS is built only
        # when at least one constituent in this run is flagged
        # intensive, so models with no temperature constituent pay no
        # extra cost.
        any_intensive = any(
            getattr(c, "is_intensive", False) for c in constituents.values()
        )

        self.lhs.update_values(
            registry,
            current_time,
            time_step,
        )
        real_cell_count = registry.get(NUMBER_OF_REAL_CELLS)
        A_extensive = csr_matrix(
            (self.lhs.coefficients, (self.lhs.rows, self.lhs.columns)),
            shape=(real_cell_count, real_cell_count),
        )

        if any_intensive:
            if self._lhs_intensive is None:
                self._lhs_intensive = LHS(registry)
            self._lhs_intensive.update_values(
                registry,
                current_time,
                time_step,
                is_intensive=True,
            )
            A_intensive = csr_matrix(
                (
                    self._lhs_intensive.coefficients,
                    (self._lhs_intensive.rows, self._lhs_intensive.columns),
                ),
                shape=(real_cell_count, real_cell_count),
            )
        else:
            A_intensive = None

        # loop through all constituents
        for constituent_name, constituent in constituents.items():
            # Phase-D Unit D1: pick the LHS / A built with this
            # constituent's intensive-ness flag. Extensive default
            # preserves all prior behaviour.
            is_intensive = bool(getattr(constituent, "is_intensive", False))
            A = A_intensive if is_intensive else A_extensive
            lhs_for_constituent = (
                self._lhs_intensive if is_intensive else self.lhs
            )
            constituent_value = registry.get_at_time(constituent_name, current_time)
            next_constituent_value = registry.get_at_time(constituent_name, current_time + time_step)

            # Phase-D Unit C-beta: pre-solve wet->dry mass handoff.
            # Computes a per-cell mass-rate source for cells going dry
            # at t+1, adds it to the RHS so the implicit solve carries
            # the redistributed mass in the same step, and returns the
            # per-step ``lost`` scalar (donor mass that could not be
            # apportioned to a wet face neighbour). Opt-in via Unit A's
            # ``wet_dry_metric``: when ``WET_MASK`` is absent, both
            # outputs are zero and the legacy behaviour is preserved.
            drain_source, drain_lost = drain_newly_dry(
                registry=registry,
                current_time=current_time,
                time_step=time_step,
                constituent_name=constituent_name,
            )

            # update right hand side of the matrix
            constituent.rhs.update_values(
                registry=registry,
                current_time=current_time,
                time_step=time_step,
                constituent_name=constituent_name,
            )
            # Inject the drain source. ``drain_source`` is already
            # shaped to ``(nreal,)`` -- the same shape as ``rhs.values``
            # -- and zeros out on the legacy path.
            constituent.rhs.values[:] = constituent.rhs.values + drain_source

            # Phase F T2-B (2026-05-21): per-constituent first-order
            # decay. When ``constituent.decay_rate > 0`` (1/s, set from
            # the per-day config value at Constituent init), add
            # ``k * V[t+1]`` to the LHS diagonal so the implicit solve
            # produces ``c[t+1] = c[t] / (1 + k*dt)`` in the steady-
            # advection limit. Build a per-constituent A copy to avoid
            # mutating the shared LHS matrix; no-op for conservative
            # transport (default ``decay_rate=0``).
            decay_rate = float(getattr(constituent, "decay_rate", 0.0) or 0.0)
            if decay_rate > 0.0:
                volume_next = np.asarray(
                    registry.get_at_time(VOLUME, current_time + time_step)
                )[: int(real_cell_count)]
                diag_add = decay_rate * volume_next
                A_solve = A.copy()
                A_solve.setdiag(A_solve.diagonal() + diag_add)
            else:
                A_solve = A

            # solve
            x = linalg.spsolve(A_solve, constituent.rhs.values)
            x_full = xr.DataArray(np.zeros(constituent_value.shape), coords=constituent_value.coords)
            x_full[:len(x)] = x

            # Phase-D Unit C-beta (continued): post-solve accumulation
            # of the rule-3 wet-dry edge leak diagnostic exposed by
            # ``LHS.update_values``. On edges where the donor side is
            # wet at t+1 and the recipient is dry-at-t+1, the LHS
            # includes the donor's diagonal advection contribution so
            # the implicit solve sinks mass at rate ``|adv| * c[t+1,
            # donor]``. That mass has left the wet domain (the dry
            # recipient has no water to hold it) and is logged here as
            # an honest accounting of the wet-side outflow loss. The
            # diagnostic is read from this constituent's own LHS so an
            # intensive scalar (which uses the intensive-LHS cache key
            # that emits empty leak arrays) does not pick up the
            # extensive constituent's leak entries. Unit D1.
            leak_total = 0.0
            wd_donors = getattr(lhs_for_constituent, "wet_dry_leak_donors", None)
            if wd_donors is not None and wd_donors.size > 0:
                wd_abs_adv = lhs_for_constituent.wet_dry_leak_abs_adv
                # Phase F (2026-05-21): use the caller's scalar time_step
                # so the array-dt path does not raise.
                dt_sec = float(time_step.total_seconds())
                leak_total = float(
                    np.sum(wd_abs_adv * x[wd_donors]) * dt_sec
                )

            # Phase-D Unit D1: skip the mass_lost_to_dry accumulator
            # for intensive constituents. The diagnostic is summed as
            # mg-equivalent for concentration species; for an intensive
            # scalar like temperature the same quantity is V*T (heat-
            # content units) which does not sum with the other
            # constituents' loss totals or interact with the
            # end-of-run warning denominator (BC inflow MASS).
            step_lost = float(drain_lost) + float(leak_total)
            if step_lost > 0 and not is_intensive:
                self.mass_lost_to_dry.setdefault(
                    constituent_name, []
                ).append(step_lost)

            # Phase-D Unit B: lift the c~0 newly-wet artifact (opt-in
            # via Unit A's wet_dry_metric; no-op when WET_MASK is
            # absent from the registry).
            # Phase F (2026-05-21): also gated by the engine-level
            # ``reconstruct_newly_wet`` flag. Default True preserves
            # the Phase-D correctness behaviour; set False to match
            # the streaming repo's reference-run configuration on
            # dry-start RAS HDFs where the pass is O(N x edges) and
            # newly-wet cells overwhelmingly lack a qualifying
            # upstream neighbour anyway.
            if self._reconstruct_newly_wet:
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

        
    