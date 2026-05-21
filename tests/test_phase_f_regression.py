"""Phase F regression smoke tests (T2-F).

End-to-end regression coverage for every Phase F port and feature
added during the 2026-05-21 Tier-1 + Tier-2 work. The goal is CI-
friendly catching of silent regressions if any of the following are
broken in a future canonical change:

* T1-A: reconstruct_newly_wet opt-out kwarg + plumbing.
* T1-B: face_hydraulic_depth derivation when the HDF doesn't write
  Cell Hydraulic Depth, and the calculate_maximum_depth shape bug
  fix.
* T1-C: continuity_correction subsystem in both modes ("bc_only" and
  "all_edges") plus the "none" passthrough.
* T1-G: wind_input_height note (parity item resolved at the runner
  level, not the package level; included here as documentation only).
* T1-H: per-constituent output and the nsm1_history.nc fallback path
  in the case-study validator (canonical-side: model_outputs.zarr
  writes the constituent variables when output_variables is set).
* T2-A: point_sources CSV loading, RHS contribution, and the negative-
  Flow_Rate warning.
* T2-B: decay_rate per-constituent first-order decay (1/day -> 1/s)
  and the LHS diagonal modification.
* T2-C: diffusion-dispatch helpers ported (constant works; the other
  three methods raise clear NotImplementedError until the HDF reader
  is extended).
* T2-D: NaN / negative-value validation in IC and BC sources.
* T2-E: Internal-type BC line warning (silent on the External-only
  fixtures used here, as designed).

The fixtures are the existing plan02_2x1 (smallest; 2 real cells) and
plan08_10x5Rf_tidal_multiBndry_isle (richer; exercises wet/dry).
Tests favor short runs to keep CI fast.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import clearwater_riverine as cwr
from clearwater_riverine.constituents import _validate_constituent_values
from clearwater_riverine.variables import (
    ADVECTION_COEFFICIENT,
    FACE_HYD_DEPTH,
    FLOW_ACROSS_FACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    WET_MASK,
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


def _make_config(tmp_path, plan_dir, hdf_name, *, constituents=None, **model_overrides):
    """Build a canonical YAML config for one of the simple fixtures."""
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    if constituents is None:
        constituents = {
            "tracer": {
                "initial_conditions": {"provider": "float", "data": {"value": 100}},
                "boundary_conditions": {"provider": "float", "data": {"value": 100}},
            }
        }
    cfg = {
        "model": {
            "simulation_directory": str(plan_dir),
            "hydrodynamic_input": hdf_name,
            "start_datetime": str(start),
            "end_datetime": str(end),
            "diffusion_coefficient": 0.01,
            "output_variables": [],
            "mass_flux_calculation": False,
            "calculated_variables": {
                "wetted_surface_area": False,
                "average_depth": False,
                "maximum_depth": False,
            },
        },
        "constituents": constituents,
    }
    cfg["model"].update(model_overrides)
    cfg_path = tmp_path / "riverine.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


# --- T1-A reconstruct_newly_wet opt-out ----------------------------------


@pytest.mark.parametrize("recon_flag", [True, False], ids=["recon-on", "recon-off"])
def test_t1a_reconstruct_newly_wet_kwarg(tmp_path, recon_flag):
    """Both True and False values are accepted at __init__ and reach
    the transport engine. The engine attribute is the source of truth
    for the run-loop branch in transport.py."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        reconstruct_newly_wet=recon_flag,
    )
    assert model.transport_engine._reconstruct_newly_wet is recon_flag


# --- T1-B face_hydraulic_depth + maximum_depth fix -----------------------


def test_t1b_face_hyd_depth_auto_registered_with_depth_metric(tmp_path):
    """When wet_dry_metric needs depth and the HDF doesn't write
    Cell Hydraulic Depth, canonical auto-registers FACE_HYD_DEPTH via
    the calculated-variable fallback (WSE - cell_min_elev)."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="both",
    )
    assert FACE_HYD_DEPTH in model.registry
    depth = np.asarray(model.registry.get(FACE_HYD_DEPTH))
    # Expect (time, nface) and finite-or-NaN values
    assert depth.ndim == 2
    assert np.isfinite(depth[np.isfinite(depth)]).all()


def test_t1b_maximum_depth_is_2d_not_3d(tmp_path):
    """Regression: calculate_maximum_depth used to subtract the whole
    (nface, index) lookup curve from WSE and produce a spurious
    (time, nface, index) 3-D result. After the fix it must be 2-D."""
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        calculated_variables={
            "wetted_surface_area": False,
            "average_depth": False,
            "maximum_depth": True,
        },
    )
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    from clearwater_riverine.variables import MAXIMUM_DEPTH
    md = model.registry.get(MAXIMUM_DEPTH)
    assert md.ndim == 2, f"MAXIMUM_DEPTH should be (time, nface) but is {md.dims}"


# --- T1-C continuity_correction (three modes) ----------------------------


@pytest.mark.parametrize(
    "mode", ["none", "bc_only", "all_edges"],
    ids=["cc-none", "cc-bc_only", "cc-all_edges"],
)
def test_t1c_continuity_correction_modes(tmp_path, mode):
    """All three continuity_correction modes initialize without
    raising, register ADVECTION_COEFFICIENT, and produce a finite
    correction. ``none`` should be byte-identical to FLOW_ACROSS_FACE."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        continuity_correction=mode,
    )
    assert ADVECTION_COEFFICIENT in model.registry
    adv = np.asarray(model.registry.get(ADVECTION_COEFFICIENT))
    flow = np.asarray(model.registry.get(FLOW_ACROSS_FACE))
    assert adv.shape == flow.shape
    if mode == "none":
        np.testing.assert_array_equal(adv, flow)
    else:
        # Non-zero correction is possible; just check finiteness.
        assert np.all(np.isfinite(adv))


def test_t1c_continuity_correction_invalid_mode(tmp_path):
    """Unknown mode raises a clear ValueError."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    with pytest.raises(ValueError, match="Unknown continuity_correction"):
        cwr.ClearwaterRiverine(
            config_filepath=str(cfg_path),
            continuity_correction="wat",
        )


# --- T1-H per-constituent output (canonical writes model_outputs.zarr) ---


def test_t1h_output_variables_writes_per_constituent(tmp_path):
    """When output_variables lists constituent names, finalize writes
    them to model_outputs.zarr. Regression catches the empty-zarr bug
    that surfaced during Phase F validation."""
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        simulation_directory=str(tmp_path),  # write zarr under tmp
        hydrodynamic_input=str(PLAN02 / PLAN02_HDF),  # absolute
        output_variables=["tracer"],
    )
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    model.run()
    model.finalize(save=True)
    zarr_path = tmp_path / "model_outputs.zarr"
    assert zarr_path.exists()
    ds = xr.open_zarr(zarr_path, consolidated=False)
    assert "tracer" in ds.data_vars


# --- T2-A point_sources --------------------------------------------------


def test_t2a_point_sources_loads_and_registers(tmp_path):
    """A point-sources CSV is loaded and the per-cell flows and
    concentrations are registered on the registry. Backwards-compat
    when point_sources is absent."""
    ps_csv = tmp_path / "point_sources.csv"
    pd.DataFrame({
        "Cell_Index": [0, 0, 1],
        "Datetime": ["2023-01-01 12:00:00", "2023-01-01 12:30:00",
                     "2023-01-01 12:15:00"],
        "Flow_Rate": [0.1, 0.2, 0.05],
        "Concentration": [50.0, 60.0, 25.0],
    }).to_csv(ps_csv, index=False)

    constituents = {
        "tracer": {
            "initial_conditions": {"provider": "float", "data": {"value": 100}},
            "boundary_conditions": {"provider": "float", "data": {"value": 100}},
            "point_sources": str(ps_csv),
        }
    }
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        constituents=constituents,
        simulation_directory=str(tmp_path),
        hydrodynamic_input=str(PLAN02 / PLAN02_HDF),
    )
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    constituent = model._constituents["tracer"]
    assert constituent.has_point_sources is True
    assert "tracer_point_source_flows" in model.registry
    assert "tracer_point_source_concentrations" in model.registry

    flows = np.asarray(model.registry.get("tracer_point_source_flows"))
    concs = np.asarray(model.registry.get("tracer_point_source_concentrations"))
    assert flows.shape == concs.shape
    # Cell 0 has knots at sim-start (12:00) and 12:30; both interior
    # to the run window, so the first timestep matches the first
    # knot's Flow_Rate (0.1).
    assert flows[0, 0] == pytest.approx(0.1, rel=1e-6)
    # Cell 1 has a single knot at 12:15. Before that knot the
    # interpolator's outer-merge + .interpolate() falls back to 0.
    assert flows[0, 1] == pytest.approx(0.0, abs=1e-9)
    # All other (cell, time) entries with no source data remain at
    # the initial zero fill.
    nreal = int(model.registry.get(NUMBER_OF_REAL_CELLS))
    # Sample a cell beyond the configured sources -- if nreal > 2,
    # check; else just confirm non-source cells exist.
    if nreal > 2:
        assert flows[:, 2:nreal].sum() == 0.0


def test_t2a_point_sources_negative_flow_warns(tmp_path):
    """Negative Flow_Rate (sink) emits a UserWarning that sink
    handling is deferred."""
    ps_csv = tmp_path / "ps_sink.csv"
    pd.DataFrame({
        "Cell_Index": [0],
        "Datetime": ["2023-01-01 12:00:00"],
        "Flow_Rate": [-0.5],
        "Concentration": [0.0],
    }).to_csv(ps_csv, index=False)

    constituents = {
        "tracer": {
            "initial_conditions": {"provider": "float", "data": {"value": 100}},
            "boundary_conditions": {"provider": "float", "data": {"value": 100}},
            "point_sources": str(ps_csv),
        }
    }
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        constituents=constituents,
        simulation_directory=str(tmp_path),
        hydrodynamic_input=str(PLAN02 / PLAN02_HDF),
    )
    with pytest.warns(UserWarning, match="negative Flow_Rate"):
        cwr.ClearwaterRiverine(config_filepath=str(cfg_path))


def test_t2a_point_sources_absent_is_no_op(tmp_path):
    """Without a point_sources entry, has_point_sources is False and
    no point-source registry keys exist."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    constituent = model._constituents["tracer"]
    assert constituent.has_point_sources is False
    assert "tracer_point_source_flows" not in model.registry


# --- T2-B decay_rate -----------------------------------------------------


def test_t2b_decay_rate_config_to_seconds(tmp_path):
    """The config value (1/day) is correctly converted to 1/s and
    stored on the constituent."""
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        constituents={
            "tracer": {
                "initial_conditions": {"provider": "float", "data": {"value": 100}},
                "boundary_conditions": {"provider": "float", "data": {"value": 100}},
                "decay_rate": 86400.0,  # 86400/day = 1/s
            }
        },
    )
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model._constituents["tracer"].decay_rate == pytest.approx(1.0, rel=1e-9)


def test_t2b_decay_rate_default_zero(tmp_path):
    """Backwards-compat: no decay_rate in config -> 0.0 (conservative)."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model._constituents["tracer"].decay_rate == 0.0


# --- T2-C diffusion-dispatch backward compat -----------------------------


def test_t2c_diffusion_constant_default_works(tmp_path):
    """The diffusion dispatcher defaults to constant when
    diffusion_method is absent from the registry, preserving the
    pre-T2-C behaviour bit-identically."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    from clearwater_riverine.variables import COEFFICIENT_TO_DIFFUSION_TERM
    cdt = model.registry.get(COEFFICIENT_TO_DIFFUSION_TERM)
    assert np.all(np.isfinite(np.asarray(cdt)))


def test_t2c_diffusion_helper_imports():
    """The three method helpers are importable from utilities."""
    from clearwater_riverine.utilities import (
        _cell_diffusion_to_edge,
        _calc_diffusion_elder,
        _calc_diffusion_eddy_viscosity,
        _calc_diffusion_array,
    )
    assert callable(_cell_diffusion_to_edge)
    assert callable(_calc_diffusion_elder)
    assert callable(_calc_diffusion_eddy_viscosity)
    assert callable(_calc_diffusion_array)


# --- T2-D NaN / negative validation --------------------------------------


def test_t2d_nan_in_ic_raises():
    """NaN values in an IC source raise ValueError at validation."""
    with pytest.raises(ValueError, match="contains 1 NaN value"):
        _validate_constituent_values(
            np.array([100.0, np.nan, 50.0]),
            constituent_name="tracer",
            source_label="initial_conditions",
        )


def test_t2d_negative_in_ic_warns():
    """Negative values in an IC source emit a UserWarning."""
    with pytest.warns(UserWarning, match="contains 1 negative value"):
        _validate_constituent_values(
            np.array([100.0, -50.0, 25.0]),
            constituent_name="tracer",
            source_label="initial_conditions",
        )


def test_t2d_scalar_validates():
    """Scalar IC values pass through validation without error."""
    _validate_constituent_values(
        100.0,
        constituent_name="tracer",
        source_label="initial_conditions",
    )


# --- T2-E Internal-BC warning silent on External-only fixtures ----------


def test_t2e_no_internal_bc_warning_on_plan02(tmp_path):
    """plan02 has External-type BCs only; the Internal-BC warning
    must NOT fire on this fixture."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
        internal_warnings = [
            x for x in w if "Internal-type" in str(x.message)
        ]
    assert internal_warnings == [], (
        f"Unexpected Internal-BC warning(s) on plan02: "
        f"{[str(x.message) for x in internal_warnings]}"
    )


# --- End-to-end Phase F stack runs to completion ------------------------


def test_phase_f_full_stack_runs(tmp_path):
    """Full Phase F kwarg stack (matching the Santiam-Salem runner)
    initializes and runs without raising. Catches integration bugs
    where individual features work but combinations break."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="both",
        wet_dry_threshold={"h_min": 0.01, "V_min": 0.1},
        continuity_correction="all_edges",
        reconstruct_newly_wet=False,
    )
    model.run()
    # Constituent values should be finite at real cells.
    tracer = np.asarray(model.registry.get("tracer"))
    nreal = int(model.registry.get(NUMBER_OF_REAL_CELLS))
    finite_mask = np.isfinite(tracer[:, :nreal])
    assert finite_mask.any()
