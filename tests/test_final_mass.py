"""Domain mass-balance **regression guard**.

Rewritten for the refactored ``clearwater_data`` / ``VariableRegistry`` API.
The pre-refactor suite drove ``postproc_util._run_simulation`` ->
``ClearwaterRiverine(ras_file_path=...)`` -> ``.simulate_wq()`` / ``.mesh`` --
all removed by the refactor, so every test errored at setup on canonical
``main``. This rebuilds the suite on the config-driven constructor +
``model.calculate_mass_balance()``.

Role of this suite (decided 2026-05-18 from the Phase-B calibration probe; see
``design/streaming_chunking_implementation_plan.md`` §4 Phase B):

  This is a **regression GUARD**, not a PORT-1/PORT-2 discriminating detector.
  Measured on canonical: every plan run with a uniform tracer (IC = BC = 100)
  conserves mass to ~0.005% -- including the tidal multi-boundary plans
  (06/07/08). A uniform field has no concentration structure for the
  wet->dry / multi-BC leak to express, and the closure metric is a
  self-consistency check two model-derived quantities can shift together.
  So this suite cannot *discriminate* the wet->dry / multi-BC bug; that
  discriminating detector (the fork's bound-vs-RAS-continuity-residual metric
  and/or a non-uniform / forced dry-wet fixture) is a separate Phase-D
  deliverable. What this suite *does* guarantee: it catches gross breakage
  (it would have caught the F0 LHS-assembly crash) and any conservation
  regression introduced while re-basing the streaming layer or applying the
  numerical ports. Tolerances are set well inside the observed margins so a
  genuine regression -- orders of magnitude larger -- trips the guard.

Scenario: each plan runs a single ``tracer`` constituent, IC = BC = 100
everywhere (``provider: float`` reproduces the uniform-100 legacy fixtures and
sidesteps the new CSVDataSource column-naming contract).
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import clearwater_riverine as cwr
from clearwater_riverine.variables import CHANGE_IN_TIME

DATA = Path(__file__).parent / "data"
SIMPLE = DATA / "simple_test_cases"

# RAS2D unsteady output timestamp dataset (same path the package's
# RASHDFDataSource.__parse_datetimes uses).
_RAS_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/"
    "Unsteady Time Series/Time Date Stamp"
)

# --- Regression-guard tolerances --------------------------------------------
# Probe (2026-05-18, canonical steissberg-riverine-merged @ 0d7aff9):
#   worst |mass_percent_error|        = 0.0071 %  (plan05)
#   worst |modeled - uniform100|/dom  ~ 5e-5      (plan06)
# Guards set ~7-20x inside those margins -- all plans pass comfortably now;
# a real conservation break is orders of magnitude larger and will trip them.
GUARD_TOL_PCT = 0.05          # |mass_percent_error| must stay below 0.05 %
GUARD_ANSWER_REL = 1e-3       # modeled domain end-mass vs uniform-100 answer

# plan_key -> (data_dir, hdf_filename, diffusion_coefficient)
PLANS = {
    "plan01": (SIMPLE / "plan01_10x5", "clearWaterTestCases.p01.hdf", 0.01),
    "plan02": (SIMPLE / "plan02_2x1", "clearWaterTestCases.p02.hdf", 0.01),
    "plan03": (SIMPLE / "plan03_2x1", "clearWaterTestCases.p03.hdf", 0.01),
    "plan04": (SIMPLE / "plan04_10x5_fullBndry", "clearWaterTestCases.p04.hdf", 0.01),
    "plan05": (SIMPLE / "plan05_10x5_tidal_fullBndry", "clearWaterTestCases.p05.hdf", 0.01),
    "plan06": (SIMPLE / "plan06_10x5_tidal_multiBndry", "clearWaterTestCases.p06.hdf", 0.01),
    "plan07": (SIMPLE / "plan07_10x5_tidal_multiBndry_isle", "clearWaterTestCases.p07.hdf", 0.01),
    "plan08": (SIMPLE / "plan08_10x5Rf_tidal_multiBndry_isle", "clearWaterTestCases.p08.hdf", 0.01),
    "plan11": (DATA / "sumwere_test_cases" / "plan11_stormSurge", "clearWaterTestCases.p11.hdf", 0.01),
}

SKIP = {
    # Legacy: @pytest.mark.skip in the pre-refactor suite.
    "plan03": "Legacy: only needed in limited circumstances; slow.",
    # Storm-surge, ~large; not yet verified under the rewritten recipe.
    # Tracked as a Phase-B follow-up before adding to the guard.
    "plan11": "Storm-surge (slow); unverified under new recipe -- Phase-B follow-up.",
}


def _hdf_time_bounds(hdf_path: Path):
    """Full (first, last) output timestamp for a RAS2D plan HDF."""
    with h5py.File(hdf_path, "r") as f:
        raw = f[_RAS_TIME_PATH][()]
    stamps = pd.to_datetime(
        pd.Series(raw).str.decode("utf8"), format="%d%b%Y %H:%M:%S"
    )
    return stamps.iloc[0], stamps.iloc[-1]


def _run_plan(plan_dir: Path, hdf_name: str, diff_coef: float, tmp_path: Path):
    """Build a config, construct + run the model for the full HDF range."""
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    cfg = {
        "model": {
            "simulation_directory": str(plan_dir),
            "hydrodynamic_input": hdf_name,
            "start_datetime": str(start),
            "end_datetime": str(end),
            "diffusion_coefficient": diff_coef,
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
    cfg_path = tmp_path / "riverine.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    model.run()
    return model


@pytest.mark.parametrize("plan_key", list(PLANS))
def test_mass_conservation_guard(plan_key, tmp_path):
    if plan_key in SKIP:
        pytest.skip(SKIP[plan_key])
    plan_dir, hdf_name, diff = PLANS[plan_key]
    model = _run_plan(plan_dir, hdf_name, diff, tmp_path)

    g = model.calculate_mass_balance("tracer", calculate_answer=False)["global"]
    a = model.calculate_mass_balance(
        "tracer", calculate_answer=True, answer_value=100
    )["global"]

    pct = abs(float(g["mass_percent_error"].values[0]))
    modeled = float(g["mass_end_modeled"].values[0])
    answer = float(a["mass_end_modeled"].values[0])

    # Guard 1: boundary-flux closure self-consistency.
    assert pct < GUARD_TOL_PCT, (
        f"{plan_key}: mass-balance closure error {pct:.5f}% exceeds the "
        f"{GUARD_TOL_PCT}% regression guard "
        f"(mass_end_modeled={modeled:.6g}, "
        f"mass_end_calculated={float(g['mass_end_calculated'].values[0]):.6g})"
    )
    # Guard 2: modeled domain end-mass tracks the uniform-100 analytical mass.
    assert modeled == pytest.approx(answer, rel=GUARD_ANSWER_REL), (
        f"{plan_key}: modeled end-mass {modeled:.6g} drifted from the "
        f"uniform-100 answer {answer:.6g} beyond rel={GUARD_ANSWER_REL}"
    )


# --- Chunked-mode sibling guard (Phase-C C2) --------------------------------
# The non-chunked guard above never sets ``chunk_size``, so the standardized
# blessed output path -- ``__transport_chunked`` / ``__load_new_chunk`` /
# ``__finalize_chunk`` -> ``ChunkedZarrDataStore.write_chunk(region="auto")``,
# Zarr v3 -- had ZERO test coverage. C2 establishes that missing oracle (which
# C3, the release/checkpoint re-base living entirely in this path, will need).


def _build_model(plan_dir, hdf_name, diff_coef, tmp_path, *,
                 chunk_size=None, mass_flux=True):
    """Construct (do NOT run) a model with an ISOLATED output store.

    ``simulation_directory`` is ``tmp_path`` and ``hydrodynamic_input`` is the
    ABSOLUTE HDF path. ``clearwater_data.io.config`` joins them as
    ``simulation_directory / hydrodynamic_input``; pathlib discards the left
    side when the right is absolute, so the HDF still resolves from
    ``plan_dir`` while ``model_outputs.zarr`` lands under ``tmp_path`` -- no
    collision with the non-chunked guard and no test-data-dir pollution.

    When ``chunk_size`` is given, ``output_variables=["tracer"]`` so
    ``__finalize_chunk`` actually drives
    ``ChunkedZarrDataStore.write_chunk(region="auto")``; the non-chunked
    guard's ``output_variables=[]`` would make that write loop a no-op.
    """
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    model_cfg = {
        "simulation_directory": str(tmp_path),
        "hydrodynamic_input": str((plan_dir / hdf_name).resolve()),
        "start_datetime": str(start),
        "end_datetime": str(end),
        "diffusion_coefficient": diff_coef,
        "output_variables": ["tracer"] if chunk_size is not None else [],
        "mass_flux_calculation": mass_flux,
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
                "initial_conditions": {"provider": "float", "data": {"value": 100}},
                "boundary_conditions": {"provider": "float", "data": {"value": 100}},
            }
        },
    }
    cfg_path = tmp_path / (
        "riverine_chunked.yml" if chunk_size is not None else "riverine_probe.yml"
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cwr.ClearwaterRiverine(config_filepath=str(cfg_path))


def _even_chunk_size(plan_dir, hdf_name, diff, tmp_path):
    """Derive a chunk_size giving an EXACT, even >=2-chunk split.

    Returns ``(chunk_size_str, m)``, or ``(None, None)`` if the plan's step
    count has no clean split (caller skips -- non-even chunk boundaries are a
    separate C4 ``__init_chunks [1:-1]`` concern). A non-chunked probe (the
    loud guard only fires for chunked + mass_flux) yields the model's own
    ``CHANGE_IN_TIME``, which A3 requires ``chunk_size`` to be a multiple of.
    """
    probe = _build_model(plan_dir, hdf_name, diff, tmp_path)
    dt_s = float(probe.registry.get(CHANGE_IN_TIME))
    start, end = _hdf_time_bounds(plan_dir / hdf_name)
    n_steps = round((end - start).total_seconds() / dt_s)
    m = next(
        (k for k in (3, 2) if n_steps % k == 0 and n_steps // k >= 2), None
    )
    if m is None:
        return None, None
    return str(pd.Timedelta(seconds=dt_s) * (n_steps // m)), m


@pytest.mark.parametrize("plan_key", list(PLANS))
def test_chunked_mass_balance_closure(plan_key, tmp_path):
    """Chunked mass balance conserves to the SAME standard as non-chunked (C3a).

    Replaces the C2 loud-guard assertion: the 6th defect (chunked
    ``__finalize_chunk`` re-registers ``{constituent}_mass_flux``;
    ``_calculate_mass_flux`` has no cross-chunk accumulation) is fixed by
    folding each chunk's boundary contribution + start/end domain snapshots
    into a cross-chunk accumulator that ``calculate_global_mass_balance``
    consumes. Chunk windows overlap one slot, so per-slot FLOW drops the
    shared trailing slot on interior chunks and the final chunk keeps it;
    per-transition mass flux partitions exactly -- so every timestep is
    counted once and the chunked totals must reproduce the non-chunked ones.

    The two guards are identical to the non-chunked test, so passing both
    here proves chunked == non-chunked via the shared uniform-100 answer.
    ``chunk_size`` gives an exact even >=2-chunk split (the ``[1:-1]``
    non-even edge is a separate C4 concern), so the run genuinely crosses
    chunk boundaries.
    """
    if plan_key in SKIP:
        pytest.skip(SKIP[plan_key])
    plan_dir, hdf_name, diff = PLANS[plan_key]

    chunk_size, m = _even_chunk_size(plan_dir, hdf_name, diff, tmp_path)
    if chunk_size is None:
        pytest.skip(
            f"{plan_key}: step count has no exact >=2-chunk split with "
            f">=2 slots/chunk; non-even chunk boundaries are a C4 "
            f"(__init_chunks [1:-1]) concern, out of scope for this oracle."
        )

    model = _build_model(
        plan_dir, hdf_name, diff, tmp_path,
        chunk_size=chunk_size, mass_flux=True,
    )
    model.run()

    g = model.calculate_mass_balance("tracer", calculate_answer=False)["global"]
    a = model.calculate_mass_balance(
        "tracer", calculate_answer=True, answer_value=100
    )["global"]
    pct = abs(float(g["mass_percent_error"].values[0]))
    modeled = float(g["mass_end_modeled"].values[0])
    answer = float(a["mass_end_modeled"].values[0])

    assert pct < GUARD_TOL_PCT, (
        f"{plan_key} [chunked x{m}, chunk_size={chunk_size}]: chunked "
        f"closure error {pct:.5f}% exceeds the {GUARD_TOL_PCT}% guard -- "
        f"cross-chunk mass continuity is not faithful to non-chunked"
    )
    assert modeled == pytest.approx(answer, rel=GUARD_ANSWER_REL), (
        f"{plan_key} [chunked x{m}]: chunked modeled end-mass "
        f"{modeled:.6g} drifted from the uniform-100 answer {answer:.6g} "
        f"beyond rel={GUARD_ANSWER_REL}"
    )


@pytest.mark.parametrize("plan_key", list(PLANS))
def test_chunked_v3_write_path_sound(plan_key, tmp_path):
    """The standardized v3 chunked write path reproduces the non-chunked field.

    Isolated from the C3-owned mass-flux-continuity gap by running with
    ``mass_flux_calculation=False`` (so the loud guard does not fire and no
    per-chunk flux is registered). This drives the full multi-chunk
    ``__transport_chunked`` / ``__load_new_chunk`` / ``__finalize_chunk`` ->
    ``ChunkedZarrDataStore.write_chunk(region="auto")`` (Zarr v3) path -- which
    no test had ever driven to completion.

    Oracle: the chunked WRITTEN store must equal the NON-CHUNKED in-memory
    field cell-for-cell, on the time slots the chunked run actually wrote.
    Self-calibrating against the model's true (ghost-inclusive) field -- a
    uniform IC=BC=100 tracer is 100 only in real cells; ghost/boundary cells
    are legitimately 0 after the IC in BOTH modes, so a "== 100" oracle would
    be wrong. ``region="auto"`` leaves trailing rolling-boundary slots NaN (a
    C3/D4 concern), so only finite written rows are compared. No closure
    assertion: the boundary-flux metric needs the disabled mass flux.
    """
    if plan_key in SKIP:
        pytest.skip(SKIP[plan_key])
    plan_dir, hdf_name, diff = PLANS[plan_key]

    chunk_size, m = _even_chunk_size(plan_dir, hdf_name, diff, tmp_path)
    if chunk_size is None:
        pytest.skip(
            f"{plan_key}: step count has no exact >=2-chunk split with "
            f">=2 slots/chunk; non-even chunk boundaries are a C4 "
            f"(__init_chunks [1:-1]) concern, out of scope for this oracle."
        )

    # Non-chunked reference field (the established-green guard path).
    ref_model = _build_model(plan_dir, hdf_name, diff, tmp_path, mass_flux=False)
    ref_model.run()
    ref = np.asarray(ref_model.registry.get("tracer"))

    # Chunked run through the v3 write path.
    model = _build_model(
        plan_dir, hdf_name, diff, tmp_path,
        chunk_size=chunk_size, mass_flux=False,
    )
    model.run()

    store = tmp_path / "model_outputs.zarr"
    assert store.exists(), (
        f"{plan_key} [chunked x{m}]: chunked run wrote no Zarr store"
    )
    ds = xr.open_zarr(store, consolidated=False)
    assert "tracer" in ds, (
        f"{plan_key} [chunked x{m}]: 'tracer' missing from the v3 store"
    )
    out = ds["tracer"].values

    # Compare index-aligned (both runs share the identical global uniform
    # time grid -- same HDF, dt, start/end), on rows the chunked store wrote
    # fully (finite). Trailing NaN rolling-boundary slots are the known
    # C3/D4 concern and are excluded from the soundness oracle.
    n = min(ref.shape[0], out.shape[0])
    ref, out = ref[:n], out[:n]
    written = np.isfinite(out).all(axis=tuple(range(1, out.ndim)))
    n_written = int(written.sum())
    assert n_written >= 2, (
        f"{plan_key} [chunked x{m}]: only {n_written} fully-written time "
        f"rows in the v3 store; chunked write produced no usable output"
    )
    mism = ~np.isclose(out[written], ref[written], rtol=1e-6, atol=1e-6)
    assert not mism.any(), (
        f"{plan_key} [chunked x{m}]: v3 chunked write diverges from the "
        f"non-chunked field at {int(mism.sum())}/{mism.size} written cells "
        f"(max |Δ|={float(np.nanmax(np.abs(out[written] - ref[written]))):.4g}); "
        f"chunked transport/write is not faithful to non-chunked"
    )


@pytest.mark.parametrize("plan_key", list(PLANS))
def test_chunked_resume_equivalence(plan_key, tmp_path):
    """Checkpoint at a chunk boundary + resume == uninterrupted (C3b).

    Drives the full end-to-end resume contract:
      1. Run the chunked plan uninterrupted -> closure C0 (reference).
      2. Build a separate chunked run, advance past the first chunk
         boundary, ``model.checkpoint(dir)``, tear down.
      3. ``ClearwaterRiverine.from_checkpoint(config, dir)`` rebuilds with
         the existing output store preserved (clearwater_data
         ``init_template=False``), restores the C3a accumulator + resume
         timestamp, stages per-constituent boundary-slot ICs, loads the
         resume chunk's hydrodynamic window. ``run()`` continues to
         ``end_datetime``.
      4. Assert resumed closure C2 matches C0 within tight tolerance.

    The chunked transport+write path is deterministic (proven by the C2
    oracle), so resume should produce a bit-identical (or within FP
    noise) final mass-balance number. Plans with no exact >=2-chunk
    even split are skipped to isolate this oracle from the C4
    ``__init_chunks [1:-1]`` edge.
    """
    if plan_key in SKIP:
        pytest.skip(SKIP[plan_key])
    plan_dir, hdf_name, diff = PLANS[plan_key]

    chunk_size, m = _even_chunk_size(plan_dir, hdf_name, diff, tmp_path)
    if chunk_size is None:
        pytest.skip(
            f"{plan_key}: step count has no exact >=2-chunk split with "
            f">=2 slots/chunk; non-even chunk boundaries are a C4 "
            f"(__init_chunks [1:-1]) concern, out of scope for this oracle."
        )

    # 1. Uninterrupted chunked run (reference).
    ref_dir = tmp_path / "uninterrupted"
    ref_dir.mkdir()
    ref_model = _build_model(
        plan_dir, hdf_name, diff, ref_dir,
        chunk_size=chunk_size, mass_flux=True,
    )
    ref_model.run()
    g_ref = ref_model.calculate_mass_balance(
        "tracer", calculate_answer=False
    )["global"]
    pct_ref = abs(float(g_ref["mass_percent_error"].values[0]))
    modeled_ref = float(g_ref["mass_end_modeled"].values[0])

    # 2. Separate chunked run; advance past the first chunk boundary.
    cp_dir = tmp_path / "with_checkpoint"
    cp_dir.mkdir()
    interim_model = _build_model(
        plan_dir, hdf_name, diff, cp_dir,
        chunk_size=chunk_size, mass_flux=True,
    )
    first_boundary = interim_model._ClearwaterRiverine__chunk_ends[0]
    while interim_model._ClearwaterRiverine__current_time < first_boundary:
        interim_model.update()
    interim_model.update()  # the update that crosses the boundary
    assert (
        interim_model._ClearwaterRiverine__last_finalized_boundary
        == first_boundary
    ), "checkpoint precondition: first boundary should be finalized"

    ckpt = tmp_path / "ckpt"
    interim_model.checkpoint(ckpt)
    assert (ckpt / "checkpoint.json").exists()
    assert (ckpt / "resume_state.npz").exists()

    # Drop references to the original model -- simulate a real resume
    # (separate process) so any reliance on in-memory state would show.
    interim_config = list(cp_dir.glob("*.yml"))[0]
    del interim_model

    # 3. Resume + run to end.
    resumed = cwr.ClearwaterRiverine.from_checkpoint(
        str(interim_config), str(ckpt)
    )
    resumed.run()
    g_res = resumed.calculate_mass_balance(
        "tracer", calculate_answer=False
    )["global"]
    pct_res = abs(float(g_res["mass_percent_error"].values[0]))
    modeled_res = float(g_res["mass_end_modeled"].values[0])

    # 4. Resumed closure ~== uninterrupted (deterministic; expect tight).
    assert pct_res == pytest.approx(pct_ref, abs=1e-9, rel=1e-9), (
        f"{plan_key} [chunked x{m}]: resumed closure pct {pct_res:.6g} "
        f"differs from uninterrupted {pct_ref:.6g} -- resume is not "
        f"faithful to an uninterrupted chunked run"
    )
    assert modeled_res == pytest.approx(modeled_ref, rel=1e-9), (
        f"{plan_key} [chunked x{m}]: resumed modeled end-mass "
        f"{modeled_res:.8g} drifted from uninterrupted "
        f"{modeled_ref:.8g} beyond rel=1e-9"
    )
