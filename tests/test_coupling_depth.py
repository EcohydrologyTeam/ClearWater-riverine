"""Change A (Riverine MeshView-compat, 2026-05-30): on-demand,
precedence-based ``coupling_depth``.

The v3 ClearWater-modules coupling needs the cell *mean* water-column
depth. CWR resolves it by precedence and computes it ONLY on demand --
standalone transport runs are byte-unchanged. These tests lock in the
contract the modules repo builds to:

  ``enable_coupling_depth()`` (public, idempotent) enables + seed-
  computes the resolved depth and registers it under the string
  ``'coupling_depth'``, refreshed per chunk while enabled.

Coverage:
  1. Precedence:
       * RAS "Cell Hydraulic Depth" present -> ``coupling_depth`` equals
         the RAS-read values (branch 1 beats branch 2).
       * absent + elevation-volume lookups present -> equals
         ``volume / wetted_surface_area`` (branch 2).
       * neither -> equals ``WSE - bed`` and a UserWarning is emitted
         (branch 3).
  2. On-demand: WITHOUT ``enable_coupling_depth()`` the registered-
     variable set and ``coupling_depth`` absence match a baseline
     standalone init -- nothing new is computed.
  3. Per-chunk: after ``enable_coupling_depth()`` a >=2-chunk run has a
     correct ``coupling_depth`` at the FIRST and LAST timestep of EACH
     chunk (guards the wsa-per-chunk dependency refresh that otherwise
     raises an xarray AlignmentError at chunk 2).
  4. ``is_chunked`` True/False.

Fixtures reuse the existing simple_test_cases plans (see
``test_phase_d_integration.py`` / ``test_final_mass.py`` for the same
config-builder + chunk-split helpers). The plan02 fixture lacks RAS
"Cell Hydraulic Depth" but has the elevation-volume lookups, so it
resolves via branch 2 (volume/wsa) -- identical to the validated manual
coupling path. Branch 1 (RAS-present) is exercised by injecting a known
RAS array under the model's private stash key, since no test fixture HDF
ships the optional dataset.
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
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_riverine.variables import (
    COUPLING_DEPTH,
    CHANGE_IN_TIME,
    FACE_HYD_DEPTH,
    LOOKUP_VOLUME,
    LOOKUP_WETTED_SURFACE_AREA,
    VOLUME,
    WETTED_SURFACE_AREA,
)


DATA = Path(__file__).parent / "data"
SIMPLE = DATA / "simple_test_cases"
PLAN02 = SIMPLE / "plan02_2x1"
PLAN02_HDF = "clearWaterTestCases.p02.hdf"

_RAS_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/"
    "Unsteady Time Series/Time Date Stamp"
)

# Private stash key the model registers the RAS Cell Hydraulic Depth
# under (mirrors model.__ras_cell_hyd_depth_stash). Kept in sync here so
# the branch-1 injection test can plant a known RAS array.
_RAS_STASH = "_ras_cell_hydraulic_depth"


def _hdf_time_bounds(hdf_path: Path):
    with h5py.File(hdf_path, "r") as f:
        raw = f[_RAS_TIME_PATH][()]
    stamps = pd.to_datetime(
        pd.Series(raw).str.decode("utf8"), format="%d%b%Y %H:%M:%S"
    )
    return stamps.iloc[0], stamps.iloc[-1]


def _make_config(tmp_path, plan_dir, hdf_name, **model_overrides):
    """Build a canonical YAML config for a simple fixture.

    ``calculated_variables`` is OMITTED entirely -- the standalone path
    Change A must leave untouched (no coupling depth unless enabled).
    """
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    cfg = {
        "model": {
            "simulation_directory": str(plan_dir),
            "hydrodynamic_input": hdf_name,
            "start_datetime": str(start),
            "end_datetime": str(end),
            "diffusion_coefficient": 0.01,
            "output_variables": [],
            "mass_flux_calculation": False,
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


def _even_chunk_size(plan_dir, hdf_name, tmp_path):
    """Derive a chunk_size giving an exact, even >=2-chunk split.

    Mirrors ``test_final_mass._even_chunk_size``: probe a non-chunked
    model for its uniform timestep, then pick a chunk window that splits
    the step count into 2 (or 3) equal chunks with >=2 slots each.
    Returns ``(chunk_size_str, n_chunks)`` or ``(None, None)``.
    """
    probe_cfg = _make_config(tmp_path, plan_dir, hdf_name)
    probe = cwr.ClearwaterRiverine(config_filepath=str(probe_cfg))
    dt_s = float(probe.registry.get_variable(CHANGE_IN_TIME).get_data())
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    n_steps = round((end - start).total_seconds() / dt_s)
    m = next((k for k in (2, 3) if n_steps % k == 0 and n_steps // k >= 2), None)
    if m is None:
        return None, None
    return str(pd.Timedelta(seconds=dt_s) * (n_steps // m)), m


def _expected_volume_over_wsa(registry):
    """volume / wetted_surface_area where wsa > 0 (else 0)."""
    vol = np.asarray(registry.get_variable(VOLUME).get_data())
    wsa = np.asarray(registry.get_variable(WETTED_SURFACE_AREA).get_data())
    return np.where(wsa > 0, vol / np.where(wsa > 0, wsa, 1.0), 0.0)


# --- 1. precedence -------------------------------------------------------


def test_precedence_volume_over_wsa_when_lookups_present(tmp_path):
    """plan02 has no RAS Cell Hydraulic Depth but HAS the elevation-volume
    lookups, so after ``enable_coupling_depth()`` the resolved
    ``coupling_depth`` is branch 2: ``volume / wetted_surface_area``."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))

    # Sanity: this fixture really lacks the RAS dataset (drives branch 2).
    assert model.ras_cell_hydraulic_depth_available is False

    model.enable_coupling_depth()
    assert COUPLING_DEPTH in model.registry
    assert COUPLING_DEPTH in model.mesh  # MeshView item access for the bridge
    # The wsa branch self-registers wetted_surface_area.
    assert WETTED_SURFACE_AREA in model.registry

    depth = np.asarray(model.registry.get_variable(COUPLING_DEPTH).get_data())
    expected = _expected_volume_over_wsa(model.registry)
    assert depth.shape == expected.shape
    ntime = depth.shape[0]
    for t in (0, ntime // 2, ntime - 1):
        np.testing.assert_allclose(depth[t], expected[t], rtol=1e-9, atol=1e-12)
    assert not np.isnan(depth).any()

    # MeshView returns the same array the v3 process bridges as ``depth``.
    np.testing.assert_array_equal(np.asarray(model.mesh[COUPLING_DEPTH]), depth)


def test_precedence_ras_cell_hydraulic_depth_wins(tmp_path):
    """When RAS Cell Hydraulic Depth is available, branch 1 wins:
    ``coupling_depth`` equals the RAS-read values, NOT volume/wsa.

    No test fixture HDF ships the optional dataset, so simulate
    availability by planting a known RAS array under the model's private
    stash key and flipping the read-time flag, then enabling coupling
    depth. The resolver must prefer it over the volume/wsa branch."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))

    # Build a distinctive RAS array shaped like VOLUME (time, nface) that
    # cannot be confused with volume/wsa.
    vol = model.registry.get_variable(VOLUME).get_data()
    ras_values = xr.full_like(vol.astype(float), 7.0)
    model.registry.register(
        _RAS_STASH, DataArrayVariable(ras_values, space_dimension="nface")
    )
    # Flip the read-time availability flag (name-mangled private attr).
    model._ClearwaterRiverine__ras_cell_hydraulic_depth_available = True

    model.enable_coupling_depth()
    assert COUPLING_DEPTH in model.registry
    assert model.ras_cell_hydraulic_depth_available is True

    depth = np.asarray(model.registry.get_variable(COUPLING_DEPTH).get_data())
    np.testing.assert_array_equal(depth, np.asarray(ras_values))
    # And it must NOT equal the volume/wsa branch (would mean precedence
    # was ignored).
    if WETTED_SURFACE_AREA in model.registry:
        assert not np.allclose(depth, _expected_volume_over_wsa(model.registry))


def test_precedence_wse_minus_bed_warns_when_no_lookups(tmp_path):
    """Neither RAS depth nor the elevation-volume lookups available ->
    branch 3: ``coupling_depth`` equals ``WSE - bed`` and a UserWarning
    is emitted that this is max depth / overestimates the mean."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))

    # Tear down everything the volume/wsa branch depends on, mimicking a
    # config that never had the lookup tables.
    for name in (WETTED_SURFACE_AREA, LOOKUP_VOLUME, LOOKUP_WETTED_SURFACE_AREA):
        if name in model.registry:
            model.registry.unregister(name)
    assert model.ras_cell_hydraulic_depth_available is False

    from clearwater_riverine.utilities import calculate_face_hyd_depth

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.enable_coupling_depth()  # must not raise
    assert COUPLING_DEPTH in model.registry

    depth = np.asarray(model.registry.get_variable(COUPLING_DEPTH).get_data())
    expected = np.asarray(calculate_face_hyd_depth(model.registry).get_data())
    np.testing.assert_array_equal(depth, expected)

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(COUPLING_DEPTH in m and "maximum" in m.lower() for m in msgs), (
        f"expected a WSE-bed/max-depth warning; got {msgs}"
    )


# --- 2. on-demand (standalone byte-unchanged) ----------------------------


def test_coupling_depth_absent_without_enable(tmp_path):
    """WITHOUT ``enable_coupling_depth()``, ``coupling_depth`` is never
    computed or registered, and no coupling-depth side effects leak into
    the registry (the private RAS stash and an auto wetted_surface_area
    from the wsa branch must both be absent)."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model.coupling_depth_enabled is False
    assert COUPLING_DEPTH not in model.registry
    assert COUPLING_DEPTH not in model.mesh
    # Coupling-depth-only side effects must not appear standalone.
    assert _RAS_STASH not in model.registry
    assert WETTED_SURFACE_AREA not in model.registry


def test_standalone_registry_set_unchanged_by_change(tmp_path):
    """The registered-variable set of a standalone (non-coupled) init
    must NOT contain any coupling-depth artifacts. This is the on-demand
    guarantee: nothing new is computed unless coupling is enabled."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    registered = set(model.registry._registry.keys())
    for artifact in (COUPLING_DEPTH, _RAS_STASH):
        assert artifact not in registered, (
            f"{artifact} leaked into a standalone init"
        )


# --- 3. per-chunk refresh ------------------------------------------------


def test_coupling_depth_correct_each_chunk(tmp_path):
    """Chunked (>=2-chunk) run with coupling depth enabled:
    ``coupling_depth`` is present and equals ``volume / wsa`` at the
    FIRST and LAST timestep of EACH chunk. Guards the per-chunk
    wetted_surface_area dependency refresh -- without it chunk 2's
    ``volume / wsa`` raises an xarray AlignmentError (chunk-1 wsa axis vs
    chunk-2 volume axis)."""
    chunk_size, n_chunks = _even_chunk_size(PLAN02, PLAN02_HDF, tmp_path)
    if chunk_size is None:
        pytest.skip("plan02 step count has no exact >=2-chunk split")

    # ``output_variables=[]`` so the run does NOT exercise the
    # ChunkedZarrDataStore write path (broken in this env by an
    # xarray/zarr version skew, orthogonal to the refresh under test).
    # Drive ``update()`` manually so ``__load_new_chunk`` ->
    # ``__update_calculated_variables`` fires at each chunk boundary.
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        simulation_directory=str(tmp_path),
        hydrodynamic_input=str((PLAN02 / PLAN02_HDF).resolve()),
        output_variables=[],
        chunk_size=chunk_size,
    )
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model.is_chunked
    model.enable_coupling_depth()

    seen_chunks = 0
    last_window = None
    while model.current_time < model._end_datetime:
        vol_da = model.registry.get_variable(VOLUME).get_data()
        times = vol_da.time.values
        window = (times[0], times[-1])
        if window != last_window:
            assert COUPLING_DEPTH in model.registry, (
                f"coupling_depth absent on chunk window {window}"
            )
            depth = np.asarray(
                model.registry.get_variable(COUPLING_DEPTH).get_data()
            )
            expected = _expected_volume_over_wsa(model.registry)
            assert depth.shape == expected.shape
            for t in (0, depth.shape[0] - 1):
                np.testing.assert_allclose(
                    depth[t], expected[t], rtol=1e-9, atol=1e-12,
                    err_msg=f"depth mismatch at slot {t} of chunk {window}",
                )
            seen_chunks += 1
            last_window = window
        model.update()

    assert seen_chunks >= n_chunks, (
        f"expected >= {n_chunks} chunk windows, observed {seen_chunks}"
    )


# --- 4. is_chunked accessor ----------------------------------------------


def test_is_chunked_false_for_unchunked_config(tmp_path):
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model.is_chunked is False
    assert model.chunk_size is None


def test_is_chunked_true_for_chunked_config(tmp_path):
    chunk_size, _ = _even_chunk_size(PLAN02, PLAN02_HDF, tmp_path)
    if chunk_size is None:
        pytest.skip("plan02 step count has no exact >=2-chunk split")
    cfg_path = _make_config(
        tmp_path, PLAN02, PLAN02_HDF,
        simulation_directory=str(tmp_path),
        hydrodynamic_input=str((PLAN02 / PLAN02_HDF).resolve()),
        output_variables=[],
        chunk_size=chunk_size,
    )
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model.is_chunked is True
    assert model.chunk_size is not None


# --- 5. enable_coupling_depth idempotence --------------------------------


def test_enable_coupling_depth_idempotent(tmp_path):
    """Calling ``enable_coupling_depth()`` twice is a no-op refresh: the
    flag stays True and ``coupling_depth`` is unchanged."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    model.enable_coupling_depth()
    first = np.asarray(model.registry.get_variable(COUPLING_DEPTH).get_data())
    model.enable_coupling_depth()
    assert model.coupling_depth_enabled is True
    second = np.asarray(model.registry.get_variable(COUPLING_DEPTH).get_data())
    np.testing.assert_array_equal(first, second)


def test_model_constructs_with_calculated_variables_none(tmp_path):
    """Regression for the config path that leaves ``calculated_variables``
    as ``None`` (key absent): the model must construct without the
    ``NoneType.items()`` crash the un-normalized dict would cause."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    assert model is not None
