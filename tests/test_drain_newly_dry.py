"""Phase-D Unit C-beta: pre-solve wet->dry mass handoff tests.

Unit-tests ``drain_newly_dry`` directly against hand-built registries
that exercise the cases the fork's ``TestDrainNewlyDry`` covers, plus
a no-op-when-opt-out test specific to canonical's Unit-A opt-in gate:

  1. No-op when ``WET_MASK`` is absent (Unit-A opt-out is bit-identical).
  2. No-op when no cell transitions from wet to dry.
  3. Outflow to a wet neighbour: drain source carries ``f * c``; the
     unaccounted donor mass (``V*c - f*dt*c``) is returned as ``lost``.
  4. Isolated cell with no edges: the full donor mass is ``lost``.
  5. Outflow only to a dry-at-t+1 neighbour: nothing drains; full
     donor mass is ``lost``.
  6. Outflow only via a ghost face: skipped (already accounted for by
     the existing ghost-cell outflow term); full donor mass is ``lost``.
  7. Net inflow on the only edge: no outflow, full donor mass ``lost``.

The hand-built fixtures avoid spinning up a full ``ClearwaterRiverine``
model so the test focuses on the drain logic in isolation.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.transport import drain_newly_dry
from clearwater_riverine.variables import (
    CHANGE_IN_TIME,
    EDGE_FACE_CONNECTIVITY,
    FLOW_ACROSS_FACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    WET_MASK,
)


# --- fixture helpers -------------------------------------------------------


T0 = datetime(2026, 1, 1, 0, 0, 0)
DT_SEC = 60.0
T1 = T0 + timedelta(seconds=DT_SEC)


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
    volume_t,             # array (nface,)
    volume_t1,            # array (nface,)
    wet_t=None,           # array (nface,) bool, or None to skip WET_MASK
    wet_t1=None,
    tracer_t=None,        # array (nface,); defaults to ones * 50
):
    """Build a minimal registry that ``drain_newly_dry`` can read."""
    nface = nreal + nghost
    ef = np.asarray(edges, dtype=int)
    assert ef.shape == (len(edges), 2)
    registry = VariableRegistry()
    registry.register(NUMBER_OF_REAL_CELLS, FloatVariable(int(nreal)))
    registry.register(CHANGE_IN_TIME, FloatVariable(DT_SEC))
    # Edge-face connectivity (static; no time dim).
    registry.register(
        EDGE_FACE_CONNECTIVITY,
        DataArrayVariable(
            xr.DataArray(
                ef,
                dims=("nedge", "2"),
                coords={"nedge": np.arange(len(ef))},
            ),
            space_dimension="nedge",
        ),
    )
    registry.register(
        FLOW_ACROSS_FACE,
        _make_time_var(flow_t, flow_t, space_dim="nedge"),
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
    # Tracer concentration; default to 50 in real cells, 0 in ghosts.
    if tracer_t is None:
        tracer_t = np.zeros(nface)
        tracer_t[:nreal] = 50.0
    else:
        tracer_t = np.asarray(tracer_t, dtype=float)
    tracer_t1 = np.zeros(nface)
    registry.register(
        "tracer",
        _make_time_var(tracer_t, tracer_t1, space_dim="nface"),
    )
    return registry


# --- tests -----------------------------------------------------------------


def test_noop_when_wet_mask_absent():
    """Opt-out path: no WET_MASK -> drain is a no-op."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[1.0],
        volume_t=[10.0, 10.0],
        volume_t1=[10.0, 10.0],
        # wet_t/wet_t1 omitted -> WET_MASK not registered
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    np.testing.assert_array_equal(drain_source, np.zeros(2))
    assert lost == 0.0


def test_noop_when_no_wet_to_dry_transition():
    """All cells wet at both times -> nothing to drain."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[1.0],
        volume_t=[10.0, 10.0],
        volume_t1=[10.0, 10.0],
        wet_t=[True, True],
        wet_t1=[True, True],
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    np.testing.assert_array_equal(drain_source, np.zeros(2))
    assert lost == 0.0


def test_outflow_to_wet_neighbour_drains_face_rate():
    """Cell 0 wet -> dry at t+1 with positive flow 0 -> 1. Cell 1 stays
    wet. drain_source[1] = f * c[t,0]. Unaccounted residual
    V_0[t] * c_0[t] - f * dt * c_0[t] goes to ``lost``."""
    # Cell 0: V[t]=100, c=50, V[t+1]=0 (going dry). Cell 1 wet at both.
    # f = 2.5 m^3/s, dt = 60 s. Per-step face mass = 2.5 * 60 * 50 = 7500.
    # Donor mass M_0 = 100 * 50 = 5000. mass_to_wet_via_faces = 7500.
    # unaccounted = M_0 - 7500 = -2500 < 0 -> lost = 0 (the face
    # over-drains the donor; physically the implicit solve will draw
    # the remainder from the wet cell, which is the documented
    # behaviour). Use a smaller flow so we see a positive residual:
    # f = 1.0 -> mass_to_wet_via_faces = 1.0 * 60 * 50 = 3000;
    # unaccounted = 5000 - 3000 = 2000.
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[1.0],            # positive: 0 -> 1
        volume_t=[100.0, 100.0],
        volume_t1=[0.0, 100.0],  # cell 0 dry at t+1
        wet_t=[True, True],
        wet_t1=[False, True],
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    # drain_source[1] = f * c_0[t] = 1.0 * 50 = 50.0
    assert drain_source[0] == pytest.approx(0.0)
    assert drain_source[1] == pytest.approx(50.0)
    # unaccounted = V*c - f*dt*c = 100*50 - 1*60*50 = 5000 - 3000 = 2000
    assert lost == pytest.approx(2000.0)


def test_isolated_cell_routes_all_to_mass_lost():
    """Cell 0 going dry, no edges incident -> full donor mass lost."""
    # Two cells but no edge between them (edges list empty would fail
    # the assertion; use a self-edge on cell 1 or a degenerate edge
    # that does NOT touch cell 0). Cleanest: 3 cells, one edge between
    # 1 and 2; cell 0 is the isolated wet->dry donor.
    registry = _make_registry(
        nreal=3, nghost=0,
        edges=[(1, 2)],
        flow_t=[0.0],
        volume_t=[100.0, 10.0, 10.0],
        volume_t1=[0.0, 10.0, 10.0],
        wet_t=[True, True, True],
        wet_t1=[False, True, True],
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    np.testing.assert_array_equal(drain_source, np.zeros(3))
    # M_0 = V * c = 100 * 50 = 5000
    assert lost == pytest.approx(5000.0)


def test_outflow_only_to_dry_neighbour_routes_to_mass_lost():
    """Cell 0 going dry with outflow only to a neighbour also going
    dry. Drain skips the wet-dry route to a dry recipient; entire
    donor mass goes to ``lost``."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[1.0],
        volume_t=[100.0, 10.0],
        volume_t1=[0.0, 0.0],    # both go dry at t+1
        wet_t=[True, True],
        wet_t1=[False, False],
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    np.testing.assert_array_equal(drain_source, np.zeros(2))
    # Cell 0's full mass routes to lost; cell 1 has its own donor
    # contribution (V*c = 10*50 = 500) -- it has only an inflow on its
    # edge (negative direction for cell 1), so no outflow path.
    # lost_total = 5000 + 500 = 5500.
    assert lost == pytest.approx(5500.0)


def test_outflow_only_to_ghost_routes_to_mass_lost():
    """Cell 0 going dry with outflow only via a ghost face. Ghost
    outflow is already accounted for by the existing BC outflow term,
    so drain_newly_dry treats the donor's ghost outflow as ``lost``
    from the drain's perspective (the BC term will move the mass)."""
    registry = _make_registry(
        nreal=1, nghost=1,
        edges=[(0, 1)],          # edge 0 connects cell 0 (real) -> 1 (ghost)
        flow_t=[1.0],            # positive: 0 -> 1 (ghost)
        volume_t=[100.0, 0.0],
        volume_t1=[0.0, 0.0],
        wet_t=[True, True],
        wet_t1=[False, True],    # cell 0 dries; ghost slot ignored
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    np.testing.assert_array_equal(drain_source, np.zeros(1))
    # The drain skips the ghost recipient -> no face apportionment ->
    # full donor mass M_0 = 100 * 50 = 5000 goes to lost.
    assert lost == pytest.approx(5000.0)


def test_net_inflow_on_only_edge_routes_to_mass_lost():
    """Cell 0 going dry with the only incident edge carrying flow
    INTO cell 0. No outflow path; full donor mass lost."""
    registry = _make_registry(
        nreal=2, nghost=0,
        edges=[(1, 0)],          # edge from cell 1 to cell 0
        flow_t=[2.0],            # positive: 1 -> 0 (inflow to dying cell 0)
        volume_t=[100.0, 10.0],
        volume_t1=[0.0, 10.0],
        wet_t=[True, True],
        wet_t1=[False, True],
    )
    drain_source, lost = drain_newly_dry(
        registry, T0, timedelta(seconds=DT_SEC), "tracer",
    )
    np.testing.assert_array_equal(drain_source, np.zeros(2))
    # M_0 = 100 * 50 = 5000 lost (no outflow path from dying cell 0).
    assert lost == pytest.approx(5000.0)
