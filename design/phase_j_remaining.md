# Phase J — Remaining Items

Deferred from Phase G/H/I (2026-05-21). All items are non-blocking for
the Santiam-Salem validation and the current dry-cell-fix / reintegration
branches. They range from blocked-on-external to minor cleanup.

## J-1: Internal-Cells Full BC Mass Routing

**Priority:** HIGH (correctness)
**Status:** BLOCKED on HEC-RAS data availability

The RAS HDF "Internal Cells" dataset identifies which cells lie along an
Internal BC line, but RAS does not expose per-cell BC flow attribution
(i.e., what fraction of the BC-line inflow enters each internal cell).
Without that attribution, the model cannot route BC-derived tracer mass
into the correct cells — it can only warn that Internal BCs exist
(current behavior, via `io/hdf.py` warning).

**Path forward:** When HEC exports per-cell BC flow attribution (or a
workaround such as face-velocity decomposition becomes viable), implement
full mass injection at Internal BC cells. The scaffold exists:
`io/hdf.py` already reads Internal Cells into `mesh.internal_cells`.

**See:** `design/internal_bc_audit.md` (T2-E audit, 2026-05-21).

## J-2: Deferred MINOR / NIT Items from Phase G/H/I Audit

Items below were inventoried in `design/phase_g_h_i_design_spec.md` and
rated as low-impact or low-urgency. They do not affect numerical
correctness for current case studies.

| ID   | Category | Description |
|------|----------|-------------|
| F4   | MINOR    | `utilities.py` NaN guard covers only `_apply_continuity_correction`; extend to LHS/RHS entry points |
| F10  | MINOR    | Point-source CSV loader lacks header validation; silent column-name mismatch possible |
| F11  | MINOR    | `calculate_face_hyd_depth` uses `.fillna(0)` — confirm this is physically correct for dry faces |
| F15  | MINOR    | `ZarrDataStore.close()` is a no-op; should flush pending writes if any |
| F17  | NIT      | `_load_point_sources` re-reads CSV every chunk; cache on first load |
| F18  | NIT      | `Constituent.__repr__` could include `is_intensive` flag for debugging |
| F19  | NIT      | `transport.py` logs per-step timing to stdout; add a `verbose` flag |
| F20  | NIT      | `linalg.py` warning "NaN in LHS" fires once per step; rate-limit or aggregate |
| NIT2 | NIT      | `variables.py` MANNINGS_N registered but never used by transport (informational only) |
| NIT4 | NIT      | `io/hdf.py` optional temporal variable reads (`FACE_VEL_X/Y`, `EDDY_VISCOSITY`) have no fallback message |
| NIT5 | NIT      | Several `design/*.md` files reference streaming-fork commit hashes; update to canonical equivalents |

## J-3: Substrate v0.4 Release (clearwater_data on PyPI)

**Priority:** MEDIUM (developer experience)
**Status:** Waiting on team readiness

Currently `clearwater_data` is installed from GitHub
(`steissberg-clearwater-data-chunked-reader` branch) in CI. When the
team is ready, cut a v0.4 release to PyPI so the CI `pip install` step
and downstream users can use a versioned dependency instead of a branch
pin.

**Prerequisites:**
- Merge `steissberg-clearwater-data-chunked-reader` to `main`
- Verify all downstream repos (ClearWater-riverine, modules-streaming)
  pass CI against the PyPI release
- Update `pyproject.toml` in ClearWater-riverine to declare the
  versioned dependency

## J-4: TSM Calibration Parameter Audit

**Priority:** LOW (calibration, not correctness)
**Status:** Open investigation

The v3 TSM wind-function coefficients `a=0.3`, `b=1.5`, `c=2.0` were
inherited from the v1/Fortran implementation with a `/1e6` normalisation
convention. The relationship to CE-QUAL-W2's SI-unit coefficients
(`9.2`, `0.46`) and the Brady-Graves-Geyer formulation needs a
documented cross-walk. The current defaults produce validated results
(Salem T bias -0.30 C, RMSE 0.62 C with wind_input_height=10m), but the
provenance chain from literature to code should be documented for
reproducibility.

**See:** `design/clearwater_modules_v3_tsm_wind_function_specification.md`
(in the modules repo).

## J-5: Willamette Full-Domain Validation

**Priority:** MEDIUM (scientific)
**Status:** Blocked on data regeneration

The Santiam-Salem subset validation is complete. The full Willamette
domain (Corvallis-Salem-Albany) validation requires regenerating
synthetic data that was lost (see `design/willamette_validation_plan.md`
Section 13). Stage scripts 01-07 and surviving source data (full USGS
RAS HDF, NWIS data, derived_bcs_2008) are available for regeneration.
