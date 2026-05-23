"""Regression tests for the wet-transition NaN-propagation fix.

Covers Patches 1 and 3 of ``design/wet_transition_nan_fix.md``:

  - Patch 1 (root cause): ``_ghost_cell`` sanitizes ghost-cell
    concentrations -- any NaN ghost value defaults to 0.0 rather than
    multiplying ``diffusion_face`` to produce a poisoned RHS.

  - Patch 3 (defensive): ``RHS.update_values`` emits a ``UserWarning``
    when the assembled RHS contains any NaN entries.

Patch 2 (NaN-aware lift in ``reconstruct_newly_wet``) is covered by
the additions to ``test_newly_wet_reconstruction.py``.
"""
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import clearwater_riverine as cwr
from clearwater_riverine.variables import (
    EDGE_FACE_CONNECTIVITY,
    NUMBER_OF_REAL_CELLS,
)


DATA = Path(__file__).parent / "data"
SIMPLE = DATA / "simple_test_cases"
PLAN02 = SIMPLE / "plan02_2x1"
PLAN02_HDF = "clearWaterTestCases.p02.hdf"
PLAN08 = SIMPLE / "plan08_10x5Rf_tidal_multiBndry_isle"
PLAN08_HDF = "clearWaterTestCases.p08.hdf"

_RAS_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/"
    "Unsteady Time Series/Time Date Stamp"
)


def _hdf_time_bounds(hdf_path: Path):
    with h5py.File(hdf_path, "r") as f:
        raw = f[_RAS_TIME_PATH][()]
    stamps = pd.to_datetime(
        pd.Series(raw).str.decode("utf8"), format="%d%b%Y %H:%M:%S"
    )
    return stamps.iloc[0], stamps.iloc[-1]


def _make_config(tmp_path, plan_dir, hdf_name, ic_value=100.0, bc_value=100.0):
    """Minimal canonical config with a single ``tracer`` constituent."""
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    cfg = {
        "model": {
            "simulation_directory": str(plan_dir),
            "hydrodynamic_input": hdf_name,
            "start_datetime": str(start),
            "end_datetime": str(end),
            "diffusion_coefficient": 0.01,
            "output_variables": [],
            "mass_flux_calculation": True,
            "calculated_variables": {
                "wetted_surface_area": False,
                "average_depth": False,
                "maximum_depth": False,
            },
        },
        "constituents": {
            "tracer": {
                "initial_conditions": {
                    "provider": "float",
                    "data": {"value": float(ic_value)},
                },
                "boundary_conditions": {
                    "provider": "float",
                    "data": {"value": float(bc_value)},
                },
            }
        },
    }
    cfg_path = tmp_path / "riverine.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


# ---------------------------------------------------------------------------
# Patch 1: _ghost_cell sanitizes NaN ghost concentrations
# ---------------------------------------------------------------------------


def _inject_nan_into_ghost_constituent(model, constituent_name="tracer"):
    """Overwrite a ghost cell's constituent value with NaN at every
    timestep, simulating the regression scenario where a user's BC CSV
    leaves some ghosts unspecified and the registry holds NaN.

    Returns the index of the ghost cell that was poisoned."""
    edge_face = np.asarray(
        model.registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data()
    )
    nreal = int(model.registry.get_variable(NUMBER_OF_REAL_CELLS).get_data())
    ef2 = edge_face[:, 1]
    ghost_idx = int(ef2[np.where(ef2 >= nreal)[0][0]])
    var = model.registry.get_variable(constituent_name).get_data()
    # ``var`` is an xr.DataArray (time, nface). Overwrite every time
    # slot at the ghost cell with NaN to reproduce the unspecified-BC
    # condition consistently across the run.
    var.loc[{"nface": ghost_idx}] = np.nan
    return ghost_idx


def test_patch1_ghost_cell_sanitizes_nan_does_not_poison_interior(tmp_path):
    """End-to-end: inject NaN into one ghost cell's tracer value, then
    drive the canonical model. Pre-Patch-1 this NaN multiplied the
    diffusion face contribution in ``_ghost_cell(flowing_in=False)``
    and propagated through the RHS + spsolve to corrupt the interior
    tracer field. Post-Patch-1 the ghost NaN is replaced with 0.0
    before the multiplication and the interior field stays finite."""
    cfg_path = _make_config(tmp_path, PLAN08, PLAN08_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        mass_loss_warn_threshold=None,
    )
    ghost_idx = _inject_nan_into_ghost_constituent(model, "tracer")
    nreal = int(model.registry.get_variable(NUMBER_OF_REAL_CELLS).get_data())
    assert ghost_idx >= nreal

    # Drive one transport step; pre-fix this propagated NaN through
    # spsolve to many interior cells. The check is on the freshly
    # solved t=1 slice rather than the whole timeline (later timesteps
    # are still NaN from the IC fill_value and that is expected).
    model.update()

    tracer = np.asarray(
        model.registry.get_variable("tracer").get_data()
    )
    # tracer shape is (time, nface). Index 0 is the IC (filled at
    # construct time), index 1 is the result of the update we just
    # drove. Interior values at index 1 must be finite -- Patch 1
    # replaces the ghost NaN with 0 inside ``_ghost_cell``, so the
    # RHS stays finite and spsolve produces a finite c[t=1].
    interior_t1 = tracer[1, :nreal]
    n_nan_interior = int(np.isnan(interior_t1).sum())
    assert n_nan_interior == 0, (
        f"Patch 1 regression: {n_nan_interior} interior tracer values "
        f"are NaN at t=1 after a ghost-NaN injection. Expected: 0."
    )


def test_patch1_preserves_finite_ghost_contributions(tmp_path):
    """Patch 1 must only replace NaN ghosts with 0; finite ghost
    values (the common case) must be passed through unchanged.
    plan02 has BC = 100 at every ghost; the interior tracer must
    receive finite (positive) contributions from the boundary."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF, bc_value=100.0)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        mass_loss_warn_threshold=None,
    )
    model.run()
    tracer = np.asarray(model.registry.get_variable("tracer").get_data())
    nreal = int(model.registry.get_variable(NUMBER_OF_REAL_CELLS).get_data())
    interior = tracer[:, :nreal]
    assert np.isfinite(interior).all()
    # plan02 with BC = 100 everywhere should leave the interior positive.
    final = interior[-1]
    assert (final > 0).all(), (
        "Patch 1 must not zero out the contributions of finite ghost values"
    )


# ---------------------------------------------------------------------------
# Patch 3: RHS.update_values warns when NaN reaches the assembled RHS
# ---------------------------------------------------------------------------


def test_patch3_rhs_nan_warning_fires(tmp_path, monkeypatch):
    """Force NaN into the assembled RHS by monkey-patching
    ``RHS.__calculate_rhs`` to return a vector containing NaN, then
    confirm that ``update_values`` raises a ``UserWarning`` mentioning
    the constituent name. This guards against future regressions where
    a new code path silently lets NaN reach the RHS (the original
    failure mode pre-Patch-1)."""
    from clearwater_riverine.linalg import RHS

    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        mass_loss_warn_threshold=None,
    )
    rhs = model._constituents["tracer"].rhs
    # Replace __calculate_rhs with a stub that returns a NaN-tainted
    # vector matching the expected shape.
    nan_vec = np.zeros_like(rhs.values)
    nan_vec[0] = np.nan

    def _fake_calculate(self, *args, **kwargs):
        return nan_vec

    monkeypatch.setattr(
        RHS, "_RHS__calculate_rhs", _fake_calculate, raising=True
    )

    # Use the model's actual first timestamp + dt so registry.get_at_time
    # lookups land on the constituent's real time axis.
    mesh = model.mesh
    t0 = pd.Timestamp(mesh.time.values[0]).to_pydatetime()
    dt = (
        pd.Timestamp(mesh.time.values[1]) - pd.Timestamp(mesh.time.values[0])
    ).to_pytimedelta()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rhs.update_values(
            registry=model.registry,
            current_time=t0,
            time_step=dt,
            constituent_name="tracer",
        )

    rhs_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "RHS assembly produced" in str(w.message)
    ]
    assert len(rhs_warnings) == 1, (
        f"Expected 1 RHS NaN UserWarning; got {len(rhs_warnings)}: "
        f"{[str(w.message) for w in caught]}"
    )
    msg = str(rhs_warnings[0].message)
    assert "tracer" in msg, "warning message must name the constituent"
    assert "design/wet_transition_nan_fix.md" in msg, (
        "warning must reference the design memo for context"
    )


def test_patch3_rhs_no_warning_when_clean(tmp_path):
    """The RHS NaN warning is a tripwire; it must stay silent on a
    healthy run. Drives plan02 with no NaN injection and confirms no
    ``RHS assembly produced`` warning is emitted."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        mass_loss_warn_threshold=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.run()
    rhs_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "RHS assembly produced" in str(w.message)
    ]
    assert rhs_warnings == [], (
        f"Patch 3 fired on a clean run: {[str(w.message) for w in rhs_warnings]}"
    )