"""Phase-D Unit B: newly-wet reconstruction tests.

Unit-tests ``reconstruct_newly_wet`` directly against hand-built
registries that exercise the five behaviors the fork's
``TestReconstructNewlyWet`` covers, plus a no-op-when-opt-out test
specific to canonical's opt-in gate:

  1. No-op when ``WET_MASK`` is absent (Unit-A opt-out is bit-identical).
  2. No-op when no cell transitions from dry to wet.
  3. Lifts a newly-wet cell to an upstream-flow-weighted average.
  4. "Only lift, never lower" -- a non-zero solver value is preserved.
  5. Two-pass propagation: a wetting front where the only wet neighbour
     of a newly-wet cell is itself reconstructed in the same step.

The hand-built fixtures avoid spinning up a full ClearwaterRiverine
model so the test focuses on the reconstruction logic in isolation.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.transport import reconstruct_newly_wet
from clearwater_riverine.variables import (
    EDGE_FACE_CONNECTIVITY,
    FLOW_ACROSS_FACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    WET_MASK,
)


# --- fixture helpers -------------------------------------------------------


T0 = datetime(2026, 1, 1, 0, 0, 0)
T1 = T0 + timedelta(seconds=60)


def _make_time_var(values_t, values_t1, *, space_dim):
    """DataArrayVariable with a 2-slot time coord."""
    arr = np.stack([np.asarray(values_t), np.asarray(values_t1)], axis=0)
    return DataArrayVariable(
        xr.DataArray(
            arr,
            dims=("time", space_dim),
            coords={"time": [T0, T1], space_dim: np.arange(arr.shape[1])},
        ),
        space_dimension=space_dim,
    )


def _make_registry(
    *,
    nreal,
    nghost,
    edges,                # list of (face_a, face_b) pairs
    flow_t,               # array (nedge,)
    flow_t1,              # array (nedge,)
    volume_t,             # array (nface,)
    volume_t1,            # array (nface,)
    wet_t=None,           # array (nface,) bool, or None to skip WET_MASK
    wet_t1=None,
    tracer_t=None,        # array (nface,) for c[t]; defaults to zeros
):
    """Build a minimal registry that ``reconstruct_newly_wet`` can read."""
    nface = nreal + nghost
    ef = np.asarray(edges, dtype=int)
    assert ef.shape == (len(edges), 2)
    registry = VariableRegistry()
    registry.register(NUMBER_OF_REAL_CELLS, FloatVariable(float(nreal)))
    # Edge-face connectivity (static; no time dim).
    registry.register(
        EDGE_FACE_CONNECTIVITY,
        DataArrayVariable(
            xr.DataArray(ef, dims=("nedge", "2"), coords={"nedge": np.arange(len(ef))}),
            space_dimension="nedge",
        ),
    )
    registry.register(
        FLOW_ACROSS_FACE,
        _make_time_var(flow_t, flow_t1, space_dim="nedge"),
    )
    registry.register(
        VOLUME,
        _make_time_var(volume_t, volume_t1, space_dim="nface"),
    )
    if wet_t is not None and wet_t1 is not None:
        registry.register(
            WET_MASK,
            _make_time_var(wet_t, wet_t1, space_dim="nface"),
        )
    # Tracer concentration at t (only consulted for the gather-conc
    # wet-to-dry swap; default to zeros since the tests below don't
    # exercise that branch).
    tracer_t = np.zeros(nface) if tracer_t is None else np.asarray(tracer_t)
    # We need a 2-slot tracer in the registry because get_at_time
    # selects by time; default the t+1 slot to NaN so the "BC overlay"
    # mask in the test won't matter.
    tracer_t1 = np.full(nface, np.nan)
    registry.register(
        "tracer",
        _make_time_var(tracer_t, tracer_t1, space_dim="nface"),
    )
    return registry


def _make_x_full_and_next(nface, x_values):
    """Build the post-solve x_full DataArray and a next_constituent_value
    that has NaN at interior cells (so the mass-balance write would
    use x_full's values there) and a configurable BC at ghost cells."""
    x = xr.DataArray(
        np.asarray(x_values, dtype=float),
        dims=("nface",),
        coords={"nface": np.arange(nface)},
    )
    nxt = xr.DataArray(
        np.full(nface, np.nan, dtype=float),
        dims=("nface",),
        coords={"nface": np.arange(nface)},
    )
    return x, nxt


# --- tests -----------------------------------------------------------------


def test_noop_when_wet_mask_absent():
    """Opt-out path: no WET_MASK in registry -> reconstruction is a
    no-op and the existing chunked/non-chunked guard suite stays
    bit-identical."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[1.0], flow_t1=[1.0],
        volume_t=[10.0, 0.0], volume_t1=[10.0, 10.0],
        # wet_t/wet_t1 omitted -> WET_MASK not registered
    )
    x, nxt = _make_x_full_and_next(2, [100.0, 0.0])
    out = reconstruct_newly_wet(
        registry, T0, timedelta(seconds=60), "tracer", x, nxt,
    )
    np.testing.assert_array_equal(out.values, np.array([100.0, 0.0]))


def test_noop_when_no_newly_wet_cells():
    """All cells wet at both times -> no newly-wet, no change."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[1.0], flow_t1=[1.0],
        volume_t=[10.0, 10.0], volume_t1=[10.0, 10.0],
        wet_t=[True, True], wet_t1=[True, True],
    )
    x, nxt = _make_x_full_and_next(2, [100.0, 50.0])
    out = reconstruct_newly_wet(
        registry, T0, timedelta(seconds=60), "tracer", x, nxt,
    )
    np.testing.assert_array_equal(out.values, np.array([100.0, 50.0]))


def test_lifts_newly_wet_cell_from_signed_inflow():
    """Cell 0 is wet at t with c=100; cell 1 is newly wet (dry at t,
    wet at t+1); positive face flow from 0->1 at time t. Solver
    returned ~0 for cell 1 (the artifact). Reconstruction should
    lift cell 1 to ~100."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],         # edge from face 0 to face 1
        flow_t=[2.5],           # positive -> flow goes from 0 to 1
        flow_t1=[2.5],
        volume_t=[10.0, 0.0],   # cell 1 dry at t
        volume_t1=[10.0, 5.0],  # cell 1 wet at t+1
        wet_t=[True, False],
        wet_t1=[True, True],
    )
    x, nxt = _make_x_full_and_next(2, [100.0, 0.0])
    out = reconstruct_newly_wet(
        registry, T0, timedelta(seconds=60), "tracer", x, nxt,
    )
    np.testing.assert_array_equal(out.values, np.array([100.0, 100.0]))


def test_only_lift_never_lower():
    """If the solver already produced a value larger than the gather
    candidate, the reconstruction must NOT lower it. Solver wrote 150
    at cell 1; gather candidate from upstream is 100; expected: 150
    preserved."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[2.5], flow_t1=[2.5],
        volume_t=[10.0, 0.0], volume_t1=[10.0, 5.0],
        wet_t=[True, False],
        wet_t1=[True, True],
    )
    x, nxt = _make_x_full_and_next(2, [100.0, 150.0])
    out = reconstruct_newly_wet(
        registry, T0, timedelta(seconds=60), "tracer", x, nxt,
    )
    np.testing.assert_array_equal(out.values, np.array([100.0, 150.0]))


def test_two_pass_wetting_front():
    """Three cells in a line; cell 0 wet at t (c=100); cells 1 and 2
    both newly-wet at t+1. Flow goes 0->1 and 1->2 at time t. Cell 2's
    only wet neighbour at t is cell 1, which is itself newly-wet;
    the second pass picks up cell 1's just-reconstructed value as the
    upstream source for cell 2."""
    registry = _make_registry(
        nreal=3, nghost=0,
        edges=[(0, 1), (1, 2)],
        flow_t=[2.0, 2.0], flow_t1=[2.0, 2.0],
        volume_t=[10.0, 0.0, 0.0],
        volume_t1=[10.0, 5.0, 5.0],
        wet_t=[True, False, False],
        wet_t1=[True, True, True],
    )
    x, nxt = _make_x_full_and_next(3, [100.0, 0.0, 0.0])
    out = reconstruct_newly_wet(
        registry, T0, timedelta(seconds=60), "tracer", x, nxt,
    )
    # Cell 1 should lift to ~100 (from cell 0). Cell 2 should lift to
    # ~100 on the second pass (cell 1's reconstructed value).
    np.testing.assert_array_equal(out.values, np.array([100.0, 100.0, 100.0]))
