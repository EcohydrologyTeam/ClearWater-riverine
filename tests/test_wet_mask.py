"""Phase-D Unit A: wet/dry mask scaffolding.

Foundational, additive, no behavior change. Validates:
- ``compute_wet_mask`` pure function for each ``metric`` option.
- Default ``wet_dry_metric=None`` is opt-out: WET_MASK is NOT registered
  (the existing 35-pass/10-skip mass-balance suite is bit-identical to
  the pre-Unit-A state on the default path).
- Opt-in ``wet_dry_metric="volume"`` registers WET_MASK with the same
  shape as VOLUME, in both non-chunked and chunked modes (the chunked
  path re-populates the mask per chunk via ``__load_new_chunk``).

Solver/transport does NOT consume WET_MASK yet -- that lands in Phase-D
Units B/C. This file gates only the scaffolding contract.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import clearwater_riverine as cwr
from clearwater_riverine.utilities import compute_wet_mask
from clearwater_riverine.variables import VOLUME, WET_MASK

# Reuse the established guard fixtures.
from test_final_mass import (
    PLANS, SKIP, _build_model, _even_chunk_size, _hdf_time_bounds,
)


# --- compute_wet_mask pure-function unit tests ------------------------------


def test_compute_wet_mask_volume_metric():
    v = np.array([0.0, 0.05, 0.2, 1.0])
    np.testing.assert_array_equal(
        compute_wet_mask(v, metric="volume", V_min=0.1),
        np.array([False, False, True, True]),
    )


def test_compute_wet_mask_depth_metric():
    v = np.zeros(4)  # ignored under depth-only
    d = np.array([0.0, 0.005, 0.02, 0.1])
    np.testing.assert_array_equal(
        compute_wet_mask(v, d, metric="depth", h_min=0.01),
        np.array([False, False, True, True]),
    )


def test_compute_wet_mask_both_metric_requires_both_above_threshold():
    # Only the last cell is above BOTH thresholds.
    v = np.array([0.0, 0.5, 0.05, 1.0])    # > V_min: T, T, F, T
    d = np.array([0.0, 0.005, 0.5, 0.1])   # > h_min: F, F, T, T
    np.testing.assert_array_equal(
        compute_wet_mask(v, d, metric="both", h_min=0.01, V_min=0.1),
        np.array([False, False, False, True]),
    )


def test_compute_wet_mask_depth_required_when_metric_needs_it():
    v = np.array([1.0])
    with pytest.raises(ValueError, match="requires the depth"):
        compute_wet_mask(v, metric="depth")
    with pytest.raises(ValueError, match="requires the depth"):
        compute_wet_mask(v, metric="both")


def test_compute_wet_mask_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric="):
        compute_wet_mask(np.array([1.0]), metric="foo")


def test_compute_wet_mask_preserves_xarray_dataarray():
    v = xr.DataArray(
        np.array([[0.0, 1.0], [0.5, 0.05]]),
        dims=("time", "nface"),
    )
    out = compute_wet_mask(v, metric="volume", V_min=0.1)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "nface")
    np.testing.assert_array_equal(
        out.values, np.array([[False, True], [True, False]])
    )


# --- model integration: opt-out (default) and opt-in ------------------------


def _build_wet_mask_model(plan_dir, hdf_name, diff_coef, tmp_path, *,
                          chunk_size=None, wet_dry_metric="volume",
                          wet_dry_threshold=None):
    """Build a model with the wet-dry kwargs set (NOT through the config
    file -- the config writer doesn't pass them yet)."""
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    model_cfg = {
        "simulation_directory": str(tmp_path),
        "hydrodynamic_input": str((plan_dir / hdf_name).resolve()),
        "start_datetime": str(start),
        "end_datetime": str(end),
        "diffusion_coefficient": diff_coef,
        "output_variables": [],
        "mass_flux_calculation": False,
        "calculated_variables": {
            "wetted_surface_area": False,
            "average_depth": False,
            "maximum_depth": False,
        },
    }
    if chunk_size is not None:
        model_cfg["chunk_size"] = chunk_size
    cfg = {
        "model": model_cfg,
        "constituents": {
            "tracer": {
                "initial_conditions": {
                    "provider": "float", "data": {"value": 100}
                },
                "boundary_conditions": {
                    "provider": "float", "data": {"value": 100}
                },
            }
        },
    }
    cfg_path = tmp_path / "riverine_wet_mask.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric=wet_dry_metric,
        wet_dry_threshold=wet_dry_threshold,
    )


def test_default_wet_dry_metric_does_not_register_wet_mask(tmp_path):
    """Bit-identical opt-out: existing guard config (no wet-dry kwarg)
    leaves the registry unchanged. ``_build_model`` builds the model
    exactly as the established 35/10 suite does -- WET_MASK must be
    absent. Establishes the regression-safety contract for Unit A."""
    plan_dir, hdf_name, diff = PLANS["plan02"]
    model = _build_model(plan_dir, hdf_name, diff, tmp_path)
    model.run()
    assert WET_MASK not in model.registry, (
        "Default wet_dry_metric=None must NOT register WET_MASK; "
        "the existing guard config is the bit-identical opt-out path."
    )


def test_opt_in_registers_wet_mask_matching_volume_shape(tmp_path):
    plan_dir, hdf_name, diff = PLANS["plan02"]
    model = _build_wet_mask_model(
        plan_dir, hdf_name, diff, tmp_path, wet_dry_metric="volume"
    )
    model.run()
    assert WET_MASK in model.registry
    wm = np.asarray(model.registry.get(WET_MASK))
    vol = np.asarray(model.registry.get(VOLUME))
    assert wm.shape == vol.shape, (wm.shape, vol.shape)
    assert wm.dtype == bool, wm.dtype


def test_opt_in_chunked_repopulates_per_chunk(tmp_path):
    """Chunked mode: WET_MASK must be present at the END of the run with
    the LAST chunk's shape. ``__load_new_chunk`` re-registers it for
    every chunk window, so on a run that crosses chunk boundaries the
    final registry holds the chunk-aligned mask."""
    plan_dir, hdf_name, diff = PLANS["plan02"]
    chunk_size, _m = _even_chunk_size(plan_dir, hdf_name, diff, tmp_path)
    assert chunk_size is not None, "plan02 must have an even >=2-chunk split"
    model = _build_wet_mask_model(
        plan_dir, hdf_name, diff, tmp_path,
        chunk_size=chunk_size, wet_dry_metric="volume",
    )
    model.run()
    assert WET_MASK in model.registry
    wm = np.asarray(model.registry.get(WET_MASK))
    vol = np.asarray(model.registry.get(VOLUME))
    # Shape must match the last chunk's volume (chunked mode keeps only
    # the last chunk's window resident post-run).
    assert wm.shape == vol.shape, (wm.shape, vol.shape)
    assert wm.dtype == bool
