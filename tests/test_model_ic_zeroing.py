"""Phase-D Unit D2: model-level IC-zeroing opt-in and end-of-run warning.

Tests the ``ClearwaterRiverine`` wire-up that ties together the
Unit-C-gamma helpers (``zero_dry_initial_conditions``,
``emit_mass_loss_warning``) with the model's existing
``zero_dry_initial_conditions`` / ``mass_loss_warn_threshold`` kwargs.

The wire-up has two halves:

  1. After ``TransportEngine`` construction in ``__init__``, when
     ``zero_dry_initial_conditions=True`` AND ``WET_MASK`` is in the
     registry (Unit-A opt-in), call
     ``zero_dry_initial_conditions(...)`` and fold the returned IC
     loss into ``self.transport_engine.mass_lost_to_dry``.

  2. In ``finalize()``, call ``emit_mass_loss_warning(...)`` with the
     engine's accumulator, the model's constituents, and the model's
     ``mass_loss_warn_threshold``.

Both halves are gated by their respective default-off kwargs so
runs that do not opt into Phase-D wet/dry behaviour are bit-identical
to the pre-D2 path.

Uses the smallest available plan (plan02) for fast model construction.
"""
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import clearwater_riverine as cwr
from clearwater_riverine.variables import WET_MASK

DATA = Path(__file__).parent / "data"
SIMPLE = DATA / "simple_test_cases"
PLAN02 = SIMPLE / "plan02_2x1"
PLAN02_HDF = "clearWaterTestCases.p02.hdf"

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


def _make_config(tmp_path, **model_overrides):
    """Build a minimal plan02 config and write it to tmp_path."""
    start, end = _hdf_time_bounds(PLAN02 / PLAN02_HDF)
    cfg = {
        "model": {
            "simulation_directory": str(PLAN02),
            "hydrodynamic_input": PLAN02_HDF,
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
                "initial_conditions": {"provider": "float", "data": {"value": 100}},
                "boundary_conditions": {"provider": "float", "data": {"value": 100}},
            }
        },
    }
    cfg["model"].update(model_overrides)
    cfg_path = tmp_path / "riverine.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


# --- IC-zeroing wire-up tests ---------------------------------------------


def test_default_no_ic_zeroing_and_empty_accumulator(tmp_path):
    """Default kwargs (``zero_dry_initial_conditions=False``,
    ``wet_dry_metric=None``): ``WET_MASK`` is not registered, the IC
    zeroing call is skipped, and the engine's accumulator stays empty
    pre-run. Legacy bit-identical path."""
    cfg_path = _make_config(tmp_path)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert WET_MASK not in model.registry
    assert model.transport_engine.mass_lost_to_dry == {}


def test_opt_in_without_wet_mask_is_silent_noop(tmp_path):
    """``zero_dry_initial_conditions=True`` without
    ``wet_dry_metric`` set: the gate ``WET_MASK in registry`` is
    false, so the call is skipped silently (no-op preserves the
    no-Unit-A behaviour)."""
    cfg_path = _make_config(tmp_path)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        zero_dry_initial_conditions=True,
    )
    assert WET_MASK not in model.registry
    assert model.transport_engine.mass_lost_to_dry == {}


def test_opt_in_with_wet_mask_invokes_ic_zeroing(tmp_path, monkeypatch):
    """``zero_dry_initial_conditions=True`` + ``wet_dry_metric=...``:
    the IC-zeroing helper is called once at the model's start
    timestamp with the model's registry and constituents. Spies on the
    helper to verify the call, sidestepping the need for a fixture
    that has sub-threshold cells at t=0. Patches the aliased name
    ``_zero_dry_initial_conditions_fn`` on the model module because
    the bare name shadows the ``__init__`` kwarg inside the class."""
    from clearwater_riverine import model as model_module

    calls = []
    real_fn = model_module._zero_dry_initial_conditions_fn

    def spy(registry, constituents, current_time):
        calls.append((registry, set(constituents.keys()), current_time))
        return real_fn(registry, constituents, current_time)

    monkeypatch.setattr(
        model_module, "_zero_dry_initial_conditions_fn", spy,
    )

    cfg_path = _make_config(tmp_path)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        zero_dry_initial_conditions=True,
    )
    assert WET_MASK in model.registry
    assert len(calls) == 1
    # Constituent set matches what we configured.
    assert calls[0][1] == {"tracer"}
    # The call's current_time equals the model's start datetime.
    assert calls[0][2] == model._start_datetime


def test_opt_in_with_wet_mask_skipped_when_flag_off(tmp_path, monkeypatch):
    """``zero_dry_initial_conditions=False`` (default) + WET_MASK
    registered: the helper is NOT called. Just because Unit A is
    enabled does not imply Unit D2's IC zeroing happens."""
    from clearwater_riverine import model as model_module

    calls = []
    real_fn = model_module._zero_dry_initial_conditions_fn

    def spy(registry, constituents, current_time):
        calls.append(current_time)
        return real_fn(registry, constituents, current_time)

    monkeypatch.setattr(
        model_module, "_zero_dry_initial_conditions_fn", spy,
    )

    cfg_path = _make_config(tmp_path)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        # zero_dry_initial_conditions defaults to False
    )
    assert WET_MASK in model.registry
    assert calls == []


# --- end-of-run warning tests ---------------------------------------------


def test_finalize_warns_when_threshold_exceeded(tmp_path):
    """``mass_loss_warn_threshold`` is consumed by ``finalize``:
    inject a synthetic loss with zero BC inflow and verify the
    warning fires on finalize (the zero-inflow unconditional-warn
    branch)."""
    cfg_path = _make_config(tmp_path)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        mass_loss_warn_threshold=0.01,
    )
    # Pre-run injection: simulate a constituent loss without any
    # BC inflow being recorded (``rhs.bc_inflow_mass`` stays empty).
    model.transport_engine.mass_lost_to_dry["tracer"] = [42.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.finalize()
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    relevant = [m for m in msgs if "tracer" in m and "mass_lost_to_dry" in m]
    assert len(relevant) == 1


def test_finalize_silent_when_threshold_none(tmp_path):
    """``mass_loss_warn_threshold=None`` disables the warning even
    when synthetic losses are present."""
    cfg_path = _make_config(tmp_path)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        mass_loss_warn_threshold=None,
    )
    model.transport_engine.mass_lost_to_dry["tracer"] = [42.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.finalize()
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    relevant = [m for m in msgs if "tracer" in m and "mass_lost_to_dry" in m]
    assert len(relevant) == 0
