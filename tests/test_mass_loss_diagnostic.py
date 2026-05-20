"""Phase-D Unit C-gamma: IC zeroing, BC inflow accumulator, end-of-run warning.

Three pieces tested in isolation against hand-built registries / mock
constituent objects:

  - ``RHS.bc_inflow_mass`` populates from ``_ghost_cell(flowing_in=True)``
    with the per-step ``|adv| * c_ghost * dt`` for each inflow edge.
  - ``zero_dry_initial_conditions`` zeroes IC mass loaded into
    sub-threshold cells for extensive constituents and returns the loss
    by constituent.
  - ``emit_mass_loss_warning`` issues ``UserWarning`` when a
    constituent's recorded loss exceeds a fraction of its BC inflow,
    and warns unconditionally when BC inflow is zero.

The hand-built fixtures avoid spinning up a full ``ClearwaterRiverine``
model so the tests focus on the C-gamma logic in isolation.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import xarray as xr

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.linalg import RHS
from clearwater_riverine.transport import (
    emit_mass_loss_warning,
    zero_dry_initial_conditions,
)
from clearwater_riverine.variables import (
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    DIFFUSION_COEFFICIENT,
    EDGE_FACE_CONNECTIVITY,
    EDGE_VELOCITY,
    FLOW_ACROSS_FACE,
    NEDGE,
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    WET_MASK,
)


# --- fixture helpers -------------------------------------------------------


T0 = datetime(2026, 1, 1, 0, 0, 0)
DT_SEC = 60.0
T1 = T0 + timedelta(seconds=DT_SEC)


def _make_time_var(values_t, values_t1, *, space_dim):
    arr = np.stack([np.asarray(values_t), np.asarray(values_t1)], axis=0)
    return DataArrayVariable(
        xr.DataArray(
            arr,
            dims=("time", space_dim),
            coords={"time": [T0, T1], space_dim: np.arange(arr.shape[1])},
        ),
        space_dimension=space_dim,
    )


def _make_rhs_registry(
    *,
    nreal,
    nghost,
    edges,
    edge_velocity,
    flow,
    tracer_t1,
    tracer_t=None,
    diff_coef=None,
):
    """Build a registry that exercises ``RHS._ghost_cell``.

    Diffusion is suppressed by setting ``DIFFUSION_COEFFICIENT = 0``.
    """
    nface = nreal + nghost
    ef = np.asarray(edges, dtype=int)
    nedge = len(edges)
    if tracer_t is None:
        tracer_t = np.zeros(nface)
    diff_coef = (
        np.zeros(nedge) if diff_coef is None else np.asarray(diff_coef)
    )

    registry = VariableRegistry()
    registry.register(NUMBER_OF_REAL_CELLS, FloatVariable(int(nreal)))
    registry.register(NFACE, FloatVariable(nface))
    registry.register(NEDGE, FloatVariable(nedge))
    registry.register(CHANGE_IN_TIME, FloatVariable(DT_SEC))
    registry.register(DIFFUSION_COEFFICIENT, FloatVariable(0.0))

    ef_da = xr.DataArray(
        ef, dims=("nedge", "2"),
        coords={"nedge": np.arange(nedge)},
    )
    registry.register(
        EDGE_FACE_CONNECTIVITY,
        DataArrayVariable(ef_da, space_dimension="nedge"),
    )
    registry.register(
        EDGE_VELOCITY,
        _make_time_var(edge_velocity, edge_velocity, space_dim="nedge"),
    )
    registry.register(
        FLOW_ACROSS_FACE,
        _make_time_var(flow, flow, space_dim="nedge"),
    )
    registry.register(
        COEFFICIENT_TO_DIFFUSION_TERM,
        _make_time_var(diff_coef, diff_coef, space_dim="nedge"),
    )
    registry.register(
        "tracer",
        _make_time_var(tracer_t, tracer_t1, space_dim="nface"),
    )
    return registry


def _make_ic_registry(
    *,
    nreal,
    nghost,
    wet_t=None,
    volume_t=None,
    tracers: dict = None,
):
    """Minimal registry for ``zero_dry_initial_conditions``."""
    nface = nreal + nghost
    if volume_t is None:
        volume_t = np.ones(nface) * 10.0
    registry = VariableRegistry()
    registry.register(NUMBER_OF_REAL_CELLS, FloatVariable(int(nreal)))
    if wet_t is not None:
        # Two-slot time DataArray; only t=T0 is read.
        registry.register(
            WET_MASK,
            _make_time_var(wet_t, wet_t, space_dim="nface"),
        )
    registry.register(
        VOLUME,
        _make_time_var(volume_t, volume_t, space_dim="nface"),
    )
    tracers = tracers or {}
    for name, values in tracers.items():
        registry.register(
            name,
            _make_time_var(values, values, space_dim="nface"),
        )
    return registry


# --- RHS.bc_inflow_mass accumulator ---------------------------------------


def test_bc_inflow_mass_initialized_empty():
    """A fresh RHS exposes an empty ``bc_inflow_mass`` list."""
    registry = _make_rhs_registry(
        nreal=1, nghost=1,
        edges=[(0, 1)],
        edge_velocity=[-1.0],
        flow=[1.0],
        tracer_t1=[0.0, 50.0],
    )
    rhs = RHS(registry)
    assert rhs.bc_inflow_mass == []


def test_bc_inflow_mass_accumulates_inflow_edge():
    """Inflow ghost edge (velocity < 0) contributes ``|adv| * c_ghost * dt``."""
    registry = _make_rhs_registry(
        nreal=1, nghost=1,
        edges=[(0, 1)],
        edge_velocity=[-1.0],  # negative -> inflow per np.less condition
        flow=[2.0],            # |adv| = 2.0
        tracer_t1=[0.0, 50.0], # ghost concentration at t+1
    )
    rhs = RHS(registry)
    rhs._ghost_cell(
        registry, T0, timedelta(seconds=DT_SEC), "tracer", flowing_in=True,
    )
    # step mass = |adv| * c_ghost * dt = 2.0 * 50.0 * 60.0 = 6000.0
    assert rhs.bc_inflow_mass == [pytest.approx(6000.0)]


def test_bc_inflow_mass_skips_outflow_edge():
    """Outflow edge (velocity > 0) does not contribute to the
    inflow accumulator. Only the diffusion branch runs and is gated
    here by ``DIFFUSION_COEFFICIENT = 0``."""
    registry = _make_rhs_registry(
        nreal=1, nghost=1,
        edges=[(0, 1)],
        edge_velocity=[1.0],   # positive -> outflow per np.greater condition
        flow=[2.0],
        tracer_t1=[0.0, 50.0],
    )
    rhs = RHS(registry)
    rhs._ghost_cell(
        registry, T0, timedelta(seconds=DT_SEC), "tracer", flowing_in=False,
    )
    assert rhs.bc_inflow_mass == []


def test_bc_inflow_mass_appends_one_entry_per_call():
    """Two inflow calls -> two entries (one per ``run()``-equivalent step)."""
    registry = _make_rhs_registry(
        nreal=1, nghost=1,
        edges=[(0, 1)],
        edge_velocity=[-1.0],
        flow=[2.0],
        tracer_t1=[0.0, 50.0],
    )
    rhs = RHS(registry)
    rhs._ghost_cell(
        registry, T0, timedelta(seconds=DT_SEC), "tracer", flowing_in=True,
    )
    rhs._ghost_cell(
        registry, T0, timedelta(seconds=DT_SEC), "tracer", flowing_in=True,
    )
    assert len(rhs.bc_inflow_mass) == 2
    assert rhs.bc_inflow_mass[0] == pytest.approx(6000.0)
    assert rhs.bc_inflow_mass[1] == pytest.approx(6000.0)


# --- zero_dry_initial_conditions ------------------------------------------


def test_zero_dry_initial_conditions_noop_when_wet_mask_absent():
    """Opt-out path: no WET_MASK -> empty dict, registry untouched."""
    registry = _make_ic_registry(
        nreal=2, nghost=0,
        # wet_t omitted -> WET_MASK not registered
        volume_t=[10.0, 10.0],
        tracers={"tracer": [50.0, 75.0]},
    )
    constituents = {"tracer": SimpleNamespace()}  # extensive (no flag)
    lost = zero_dry_initial_conditions(registry, constituents, T0)
    assert lost == {}
    # Registry constituent values unchanged.
    c0 = np.asarray(registry.get_at_time("tracer", T0))
    np.testing.assert_array_equal(c0, np.array([50.0, 75.0]))


def test_zero_dry_initial_conditions_noop_when_all_wet():
    """All cells wet at t0 -> nothing to zero."""
    registry = _make_ic_registry(
        nreal=2, nghost=0,
        wet_t=[True, True],
        volume_t=[10.0, 10.0],
        tracers={"tracer": [50.0, 75.0]},
    )
    constituents = {"tracer": SimpleNamespace()}
    lost = zero_dry_initial_conditions(registry, constituents, T0)
    assert lost == {}
    c0 = np.asarray(registry.get_at_time("tracer", T0))
    np.testing.assert_array_equal(c0, np.array([50.0, 75.0]))


def test_zero_dry_initial_conditions_zeroes_dry_cells_and_returns_mass():
    """Sub-threshold cell at t0 has its IC concentration zeroed and the
    IC mass returned in the per-constituent total."""
    registry = _make_ic_registry(
        nreal=2, nghost=0,
        wet_t=[True, False],     # cell 1 dry at t0
        volume_t=[10.0, 5.0],
        tracers={"tracer": [50.0, 75.0]},
    )
    constituents = {"tracer": SimpleNamespace()}
    lost = zero_dry_initial_conditions(registry, constituents, T0)
    # IC mass on cell 1 = V*c = 5*75 = 375
    assert lost == {"tracer": pytest.approx(375.0)}
    c0 = np.asarray(registry.get_at_time("tracer", T0))
    np.testing.assert_array_equal(c0, np.array([50.0, 0.0]))


def test_zero_dry_initial_conditions_skips_intensive():
    """Intensive constituent (``is_intensive=True``) is left alone."""
    registry = _make_ic_registry(
        nreal=2, nghost=0,
        wet_t=[True, False],
        volume_t=[10.0, 5.0],
        tracers={
            "tracer": [50.0, 75.0],            # extensive
            "temperature": [15.0, 4.0],        # intensive
        },
    )
    constituents = {
        "tracer": SimpleNamespace(),                       # is_intensive default False
        "temperature": SimpleNamespace(is_intensive=True),
    }
    lost = zero_dry_initial_conditions(registry, constituents, T0)
    # Tracer was zeroed and mass logged. Temperature was skipped.
    assert "tracer" in lost
    assert "temperature" not in lost
    T = np.asarray(registry.get_at_time("temperature", T0))
    np.testing.assert_array_equal(T, np.array([15.0, 4.0]))


# --- emit_mass_loss_warning -----------------------------------------------


def _mock_constituent(*, bc_inflow_mass=None, is_intensive=False):
    rhs = SimpleNamespace(bc_inflow_mass=list(bc_inflow_mass or []))
    return SimpleNamespace(rhs=rhs, is_intensive=is_intensive)


def test_emit_mass_loss_warning_noop_when_threshold_none():
    constituents = {"tracer": _mock_constituent(bc_inflow_mass=[100.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_mass_loss_warning(
            mass_lost_to_dry={"tracer": [5.0]},
            constituents=constituents,
            threshold=None,
        )
    assert len(caught) == 0


def test_emit_mass_loss_warning_noop_when_empty_loss():
    constituents = {"tracer": _mock_constituent(bc_inflow_mass=[100.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_mass_loss_warning(
            mass_lost_to_dry={}, constituents=constituents, threshold=0.01,
        )
    assert len(caught) == 0


def test_emit_mass_loss_warning_fires_when_fraction_exceeds_threshold():
    # Loss = 50, BC inflow = 1000 -> 5% -> > 1% threshold -> warn.
    constituents = {"tracer": _mock_constituent(bc_inflow_mass=[1000.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_mass_loss_warning(
            mass_lost_to_dry={"tracer": [30.0, 20.0]},  # sum = 50
            constituents=constituents,
            threshold=0.01,
        )
    assert len(caught) == 1
    w = caught[0]
    assert issubclass(w.category, UserWarning)
    assert "tracer" in str(w.message)
    assert "5.00%" in str(w.message)


def test_emit_mass_loss_warning_silent_when_fraction_below_threshold():
    # Loss = 5, BC inflow = 1000 -> 0.5% -> below 1% threshold -> silent.
    constituents = {"tracer": _mock_constituent(bc_inflow_mass=[1000.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_mass_loss_warning(
            mass_lost_to_dry={"tracer": [5.0]},
            constituents=constituents,
            threshold=0.01,
        )
    assert len(caught) == 0


def test_emit_mass_loss_warning_unconditional_when_no_bc_inflow():
    # Loss > 0 but BC inflow = 0 -> warn unconditionally (typical IC
    # loss signal).
    constituents = {"tracer": _mock_constituent(bc_inflow_mass=[])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_mass_loss_warning(
            mass_lost_to_dry={"tracer": [10.0]},
            constituents=constituents,
            threshold=0.01,
        )
    assert len(caught) == 1
    assert "zero BC inflow" in str(caught[0].message)


def test_emit_mass_loss_warning_skips_intensive():
    """Intensive constituent's recorded loss is not warned about (the
    denominator's units are wrong for a scalar like temperature)."""
    constituents = {
        "temperature": _mock_constituent(
            bc_inflow_mass=[0.0], is_intensive=True,
        ),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_mass_loss_warning(
            mass_lost_to_dry={"temperature": [9999.0]},
            constituents=constituents,
            threshold=0.01,
        )
    assert len(caught) == 0
