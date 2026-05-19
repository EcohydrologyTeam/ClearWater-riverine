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
import pandas as pd
import pytest
import yaml

import clearwater_riverine as cwr

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
