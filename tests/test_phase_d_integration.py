"""Phase-D integration: full opt-in stack on a wet-dry-active plan.

End-to-end test that exercises Units A through D2 together on plan08
(the multi-boundary tidal plan with active wet-dry dynamics). The
goal is a smoke check that the gated paths cooperate -- it is not a
discriminating mass-balance test; the conservation guard in
``test_final_mass.py`` already plays that role.

What this covers
----------------
Per the project memo's Phase-D decomposition:

  - Unit A (wet-mask scaffolding) -- ``wet_dry_metric="volume"``
    registers ``WET_MASK`` so all downstream gates open.
  - Unit B (newly-wet reconstruction) -- transparently active; no
    direct assertion (would require a fixture with a known c~0
    artifact).
  - Unit C-alpha (LHS wet-dry edge filter + leak diagnostic) --
    ``transport_engine.lhs.wet_dry_leak_donors`` reflects the most
    recent step.
  - Unit C-beta (drain + ``mass_lost_to_dry``) -- the engine's
    accumulator may carry entries for the tracer.
  - Unit C-gamma (``bc_inflow_mass``, ``zero_dry_initial_conditions``,
    ``emit_mass_loss_warning``) -- the RHS accumulator carries
    per-step BC inflow entries; the IC-zeroing helper is invoked at
    init; the warning fires only when the fraction exceeds threshold.
  - Unit D1 (``is_intensive`` flag + engine LHS cache) -- the engine
    builds a single (extensive) LHS for an all-extensive run; the
    intensive cache stays at ``None``.
  - Unit D2 (model-level IC-zeroing opt-in + finalize warning) --
    ``transport_engine`` property exposes the accumulator;
    ``finalize`` invokes ``emit_mass_loss_warning``.

Ghost-edge mass-flux note
-------------------------
The fork's Step-4 ``_mass_flux`` ghost patch substitutes ghost-side
BC concentrations into ``neighbor_concentration`` because the fork
keeps ``mesh[name]`` ghost slots at NaN and stores BC values in a
separate ``input_array``. On canonical the BC values are written
directly into ``registry.get(name)``'s ghost slots by
``Constituent.set_boundary_conditions``, so the ghost-edge entries
of ``tracer_mass_flux`` are well-defined (zero NaN) without the
patch. This test asserts that property to lock it in as a contract:
if the canonical's BC-write changes and breaks ghost-edge flux
values, this test catches it. No port of the fork's patch is
required.
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
    WET_MASK,
)


DATA = Path(__file__).parent / "data"
SIMPLE = DATA / "simple_test_cases"
PLAN08 = SIMPLE / "plan08_10x5Rf_tidal_multiBndry_isle"
PLAN08_HDF = "clearWaterTestCases.p08.hdf"
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


def _make_config(tmp_path, plan_dir, hdf_name, **model_overrides):
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
                "initial_conditions": {"provider": "float", "data": {"value": 100}},
                "boundary_conditions": {"provider": "float", "data": {"value": 100}},
            }
        },
    }
    cfg["model"].update(model_overrides)
    cfg_path = tmp_path / "riverine.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


# --- end-to-end stack -----------------------------------------------------


@pytest.mark.parametrize(
    "plan_dir,hdf_name",
    [(PLAN02, PLAN02_HDF), (PLAN08, PLAN08_HDF)],
    ids=["plan02-simple", "plan08-tidal-wet-dry"],
)
def test_phase_d_full_opt_in_runs_to_completion(tmp_path, plan_dir, hdf_name):
    """Build a model with the full Phase-D opt-in stack and run it
    end-to-end. The run should complete, leave the registry
    consistent, and let ``finalize()`` execute without raising. The
    end-of-run warning is suppressed (threshold=None) so its semantics
    are exercised separately."""
    cfg_path = _make_config(tmp_path, plan_dir, hdf_name)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        zero_dry_initial_conditions=True,
        mass_loss_warn_threshold=None,
    )
    assert WET_MASK in model.registry
    # Engine exposed via the D2 property.
    engine = model.transport_engine
    assert engine is not None
    # No intensive constituents in this fixture; the D1 cache stays None.
    assert engine._lhs_intensive is None

    # Run + finalize; mass-loss warning is gated off by threshold=None.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.run()
    relevant = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning) and "mass_lost_to_dry" in str(w.message)
    ]
    assert relevant == []

    # BC inflow accumulator (Unit C-gamma) carries entries from the
    # ghost-cell injection path.
    bc_acc = model._constituents["tracer"].rhs.bc_inflow_mass
    assert isinstance(bc_acc, list)
    # On plans 02 and 08 the BC is constant ``100`` so at least one
    # step has positive inflow.
    assert any(v > 0 for v in bc_acc)


def test_phase_d_warning_emits_on_synthetic_loss(tmp_path):
    """``mass_loss_warn_threshold > 0`` + a synthetic
    ``mass_lost_to_dry`` injection: the warning fires on finalize.
    Uses plan02 for a fast model build; the run itself is not invoked
    so the test is fast (no time-stepping)."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        mass_loss_warn_threshold=0.01,
    )
    # Inject loss + leave the BC accumulator empty so the
    # zero-inflow unconditional-warn branch fires deterministically.
    model.transport_engine.mass_lost_to_dry["tracer"] = [50.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.finalize()
    relevant = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "tracer" in str(w.message)
    ]
    assert len(relevant) == 1


# --- ghost-edge mass-flux contract ----------------------------------------


def test_canonical_mass_flux_has_no_nan_at_ghost_edges(tmp_path):
    """Canonical's ``Constituent._calculate_mass_flux`` does not need
    the fork's Step-4 ghost-patch because canonical writes BC values
    directly into ``registry.get(name)``'s ghost slots. This test
    locks that contract in so a future change to the BC-write path
    that breaks ghost-edge flux values fails loudly here."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(config_filepath=str(cfg_path))
    model.run()
    mass_flux = np.asarray(model.registry.get("tracer_mass_flux"))
    nreal = int(model.registry.get(NUMBER_OF_REAL_CELLS))
    ef2 = np.asarray(model.registry.get(EDGE_FACE_CONNECTIVITY))[:, 1]
    ghost_edges = np.where(ef2 >= nreal)[0]
    assert ghost_edges.size > 0, "plan02 has boundary edges; ghost_edges must be non-empty"
    ghost_flux = mass_flux[:, ghost_edges]
    nan_count = int(np.isnan(ghost_flux).sum())
    assert nan_count == 0, (
        f"Canonical ghost-edge mass flux carries {nan_count} NaN entries "
        f"out of {ghost_flux.size}. If the BC-write path on Constituent "
        f"changed, the fork's _mass_flux ghost-patch may now be required."
    )


# --- engine diagnostics surface -------------------------------------------


def test_transport_engine_exposes_phase_d_diagnostics(tmp_path):
    """The TransportEngine exposes the Phase-D diagnostic surface
    callers and tests rely on: ``mass_lost_to_dry``, ``lhs`` (with
    ``wet_dry_leak_donors`` / ``wet_dry_leak_abs_adv`` /
    ``dry_cells_t1`` populated after the first ``update_values``
    call), and ``_lhs_intensive`` (None when no intensive constituent
    is present).

    The C-alpha leak-diagnostic attrs are populated by
    ``LHS.update_values`` -- they do not exist at fresh construction.
    Driving one ``model.run()`` step would be enough; this test runs
    the whole plan02 simulation (small, fast) so the diagnostic
    surface is exercised under the full Phase-D stack."""
    cfg_path = _make_config(tmp_path, PLAN02, PLAN02_HDF)
    model = cwr.ClearwaterRiverine(
        config_filepath=str(cfg_path),
        wet_dry_metric="volume",
        mass_loss_warn_threshold=None,
    )
    engine = model.transport_engine
    # Mass-loss accumulator (Unit C-beta) -- available from
    # ``__init__`` before any time-step.
    assert hasattr(engine, "mass_lost_to_dry")
    assert isinstance(engine.mass_lost_to_dry, dict)
    assert hasattr(engine, "lhs")
    # D1 intensive-LHS cache slot (None until a run touches it with
    # an intensive constituent in the constituent set).
    assert hasattr(engine, "_lhs_intensive")
    assert engine._lhs_intensive is None
    # Drive the simulation so ``LHS.update_values`` populates the
    # C-alpha diagnostic attrs.
    model.run()
    # After the run, the leak-diagnostic attrs reflect the last
    # solved step. WET_MASK is in the registry so they are
    # populated (not empty by absence of the gate).
    assert hasattr(engine.lhs, "wet_dry_leak_donors")
    assert hasattr(engine.lhs, "wet_dry_leak_abs_adv")
    assert hasattr(engine.lhs, "dry_cells_t1")
