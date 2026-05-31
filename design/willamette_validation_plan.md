# Willamette / Santiam-Salem Validation Plan for Canonical ClearWater-Riverine

**Date opened:** 2026-05-20
**Status:** F1 committed & pushed (`9483fd0`); F2a script written (canonical runner uncommitted in modules-phase2-ESM-streaming); F2b execution blocked by lost-data discovery 2026-05-20 (see Section 13). Next action: Stage 01 regeneration.
**Owner:** ClearWater-Riverine reintegration team
**Purpose:** Reproduce the fork's Santiam-Salem Sep-2008 validation on the canonical `steissberg-riverine-merged` branch of `ClearWater-riverine`, locking in real-corridor scale evidence for the Phase-D forward-port.

This is the **Phase F** validation deliverable. Phases A through E are complete and pushed (see `design/all_phases_complete.md`); the close-out memo's Section 8 explicitly lists this Willamette run as a separate follow-on, not part of Phase D itself.

The user requested this single memo to centralize tracking. Everything below is reconstructible from the run-provenance JSONs, the modules-phase2-ESM-streaming case-study directory, and the conversation context dumps received 2026-05-20.

---

## 1. Goal

Demonstrate that the curated forward-port of the streaming fork onto canonical `ClearWater-riverine` produces the same Santiam-Salem Sep-2008 Salem-station validation metrics the fork produced. The single most important number is the Salem temperature bias and RMSE; the secondary numbers are the per-constituent biases for DOX, NH4, NO3, TIP, and Ap at the same station.

If canonical reproduces those metrics to within numerical tolerance, the forward-port is real-corridor validated. If it does not, the discrepancy points at one of (a) the canonical-specific Phase-D code paths, (b) the API translation between fork and canonical, or (c) the orchestrator coupling, and we triage from there.

---

## 2. Reference baseline (locked 2026-05-20)

The reference run is **`v3_smoke_15day_wind10m_final_mumax_1_3`** in the modules-phase2-ESM-streaming case-study directory. The screenshot the user provided 2026-05-20 transcribes its Salem-station metrics:

| Variable | bias | RMSE | n | notes |
|---|---|---|---|---|
| water_temp_C | -0.2970 | 0.6235 | 3 | rounds to -0.30 / 0.62 (the headline metric) |
| DOX | -0.0074 | 0.3116 | 3 | was +0.116 in pre-fix; now near zero |
| NH4 | -0.0169 | 0.0169 | 1 | was +0.010 (sign flipped, same magnitude) |
| NO3 | +0.0399 | 0.0399 | 1 | identical to original wind10m |
| TIP | -0.0097 | 0.0150 | 2 | identical to original wind10m |
| Ap | -0.7358 | 0.7358 | 1 | was -0.86 / -0.82 / -0.80; slight improvement |
| Alk | n=0 | -- | -- | no observations |

Source files on disk (verified 2026-05-20):

- `…/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/output/v3_smoke_15day_wind10m_final_mumax_1_3/validation_2008/validation_metrics.csv`
- `…/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/output/v3_smoke_15day_wind10m_final_mumax_1_3/run_provenance.json`

The screenshot (2026-05-13) at `~/Documents/Screenshots/Screenshot 2026-05-13 at 05.57.02.png` shows the mu_max calibration sweep (1.0, 1.3, 1.5) all producing identical Salem T bias / RMSE (-0.30 / 0.62) because T does not depend on mu_max. `mu=1.3` was the calibrated production choice (matches obs Sep p75 cleanly).

A later run (`v3_smoke_15day_wind10m_velfix_mumax_1_5`, 2026-05-18, `a7e868c`) supersedes the `final` series by fixing the ESM constant-0.5 m/s velocity gap (now derives per-cell velocity from RAS2D edge velocities with a depth-based guard). The velfix run should produce the same T metric because the ESM velocity gap fix does not propagate back into TSM or transport. The validation_2008 directory was not yet generated for the velfix run as of 2026-05-20.

**Compare canonical against `final_mumax_1_3` metrics above** as the locked baseline.

---

## 3. Pinned configuration (from `final_mumax_1_3`'s `run_provenance.json`)

### Repositories and commits

| Repository | Branch | Commit pinned by reference run | Where it is sys.path-injected |
|---|---|---|---|
| ClearWater-Riverine-streaming (the fork, source of reference) | `dry-cell-fix` | `df6650f` (EPA-validated) | `/Users/todd/GitHub/ecohydrology/ClearWater-riverine-dev/src` in `08_run_coupled_v3_smoke.py` |
| ClearWater-modules-phase2-ESM-streaming | `streaming` | `a688d5b` (2026-05-13, "docs: wind10m bug-hunt …") | run from repo working dir |
| ClearWater-modules-streaming | `streaming` | (need to verify by reading the runner script's `sys.path` line) | injected via runner script |
| ClearWater-modules-phase2-ESM | (separate repo for ESM core) | (need to verify; runner uses `/Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM/src`) | injected via runner script |
| ClearWater-data | `steissberg-clearwater-data-chunked-reader` | `e328900` (PR companion) | depended on by canonical |

For the canonical-side replication, the swap is:

- `ClearWater-riverine-dev/src` → `ClearWater-riverine/src` (with `steissberg-riverine-merged` checked out at `100eacc` or later)
- Everything else pinned as above

### Data paths

| Variable | Path | Notes |
|---|---|---|
| HDF (hydraulics) | `/Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/data/synthetic/santiam_salem_subset_2008-09_hourly.p01.hdf` | Option C: real USGS Sep-2008 unsteady, spatially subset to 158k cells, transcoded to CW-Riverine HDF. NOT rating-curve synthesized. |
| Nutmet CSV | `/Users/todd/LargeProjects/ClearWater-riverine-case-study-Willamette/data/observed/derived_bcs_2008/synthetic_nutrients_meteorology_2008-09__obs_T.csv` | The filename says "synthetic" but the contents are observation-derived (WQP grab samples + NWIS unit-value temperature). Filename is a documented hazard; a rename has been done in some other-session work but back-compat symlink remains. |
| Met CSV | `/Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/data/synthetic/met_KSLE_2008-09_hourly.csv` | Observed KSLE (Salem airport) hourly meteorology. |
| Original RAS model (full mesh) | `/Users/todd/LargeProjects/ClearWater-riverine-case-study-Willamette/data/hecras_model/Santiam_Salem/Santiam_Salem.p01.hdf` | 5.7 GB, ~2,035,430-cell USGS validated model; only used for the Stage-01 subset extraction. Not consumed at runtime by the Stage-08 validation. |

### Run arguments (from `run_provenance.json`)

```
window_start = 2008-09-01 00:00:00
n_days = 15  (so window_end = 2008-09-15 00:00:00)
n_cells = 159,634   (158,037 real)
n_constituents = 13 (Ap, NH4, NO3, TIP, DOX, water_temp_C, plus 7 seed_<species> tracers)
diffusion_coefficient = 0.1
algae_mu_max_20 = 1.3
ic_water_temp = 17.35  (deg C)
ic_nh4 = 0.02; ic_no3 = 0.137; ic_tip = 0.029; ic_dox = 9.4; ic_ap = 1.6
reconstruct_newly_wet = false
continuity_correction = "all_edges"
wind_input_height = 10.0  m
no_sediment = false   (synthetic sediment is active)
master_step_hours = 1.0
tsm_step_hours = 1.0
nsm1_step_hours = 1.0
esm_step_hours = 24.0
transport_step_hours = 1.0   (auto-detected from HDF spacing; coincides with master step)
```

### Constituents

- 6 physical constituents: Ap, NH4, NO3, TIP, DOX, water_temp_C
- 7 seed tracers (one per species): potamogeton, ludwigia, wapato, pacific_willow, black_cottonwood, oregon_ash, reed_canarygrass
- water_temp_C carries `is_intensive=True` (single-line fork orchestrator change; required to match the -0.30 / 0.62 metric)

### Salem station

- Site ID: OREGONDEQ-10555-ORDEQ
- Lat / Lon (WGS84): 44.94386 / -123.04527
- Nearest mesh cell index: 86521 (UTM10N 496426.0, 4976497.1; distance 218.6 m from station)
- Validator: `case_studies/santiam_salem/scripts/09_validate_2008_obs.py` with `--validation-mode neighborhood_median --radius-m 500` (default ~500 m spatial median across wet cells)

---

## 4. The blocking discovery: fork-vs-canonical API gap

The Stage-08 runner (`08_run_coupled_v3_smoke.py`) is built for the fork's `ClearWater-Riverine-streaming` API, which differs from canonical's `ClearWater-riverine` API in several ways. The fork's API is xarray-Dataset-centric (`model.mesh`); canonical's API is registry-centric (`model.registry`). Substituting one for the other is not a `sys.path` one-liner.

### Specific divergences

| Concern | Fork (`ClearWater-Riverine-streaming`) | Canonical (`ClearWater-riverine`) |
|---|---|---|
| Mesh storage | `model.mesh` (xarray Dataset, time x face) | `model.registry` (VariableRegistry) |
| Constituent dict | `model.constituent_dict[name]` (Constituent objs) | `model._constituents[name]` (Constituent objs); D2 added `model.transport_engine` property |
| Reading a value at time t | `model.mesh["water_temp_C"].isel(time=t)` | `model.registry.get_at_time("water_temp_C", t)` |
| Writing a value at time t | `model.mesh["water_temp_C"].loc[{"time": t, "nface": …}] = arr` | `model.registry.set_at_time("water_temp_C", t, arr)` |
| Mesh-level metadata | `model.mesh.attrs[NUMBER_OF_REAL_CELLS]`, `model.mesh.sizes["time"]` | `model.registry.get(NUMBER_OF_REAL_CELLS)` (scalar); time count via the constituent's time coord |
| `is_intensive` plumbing | Constituent constructor reads from `constituent_config["is_intensive"]`; persisted on `mesh[name].attrs` | Same: `Constituent.__init__` reads from `constituent_config["is_intensive"]` (D1 commit `7f61582`) |
| Update entry | `model.update(update_concentration={...})` | `model.update()` (kwargs differ; needs verification) |

### Stage-08 usage of the fork API

`08_run_coupled_v3_smoke.py` accesses `transport.mesh` extensively for both reads and writes:

- Line 1666: `transport.mesh["water_temp_C"].loc[{"time": ..., "nface": ...}] = T_for_mesh.astype(...)` (writes evolved temperature back into the mesh at each step)
- Multiple `transport.mesh[...]` reads for `face_velocity`, `wse`, `depth`, `nface`, constituent state, etc.
- Multiple `transport.constituent_dict[name]` and `transport.mesh.attrs[...]` reads

A clean drop-in of `transport = cwr.ClearwaterRiverine(...)` with canonical will `AttributeError` on the first `transport.mesh` access.

---

## 5. Options evaluated

| # | Approach | Scope | Time | Risk | Validation surface |
|---|---|---|---|---|---|
| 1 | Add a fork-compat shim on canonical: a `mesh` property exposing a write-through xarray-Dataset view of the registry, plus a `constituent_dict` alias | Modest: ~50-100 lines on canonical `model.py`; new commit on `steissberg-riverine-merged` | 2-4 hours | Write-through semantics are subtle; needs unit tests. 94-test suite stays green because nothing currently touches the shim. | Full (all 6 metric rows) |
| 2 | Adapt Stage-08 for canonical's API: line-by-line translate `transport.mesh[X]` to `registry.get_at_time(X, t)` etc. | Substantial: ~150-300 edits in a new `08_run_coupled_v3_smoke_canonical.py` | 1-2 days | Fork-specific behaviors may not have clean registry equivalents and need invention; diff harder to review | Full |
| 3 | Write a minimal canonical-API runner: Transport + TSM only (skip NSM1 + ESM) | Smallest: ~300-500 lines new script | 4-6 hours | Cannot reproduce DOX / NH4 / NO3 / TIP / Ap metrics | Reduced (only T) |
| 4 | Defer | None | 0 | Pushes Phase F to a later session | None |

### Decision (2026-05-20)

**Option 1** chosen by the user. Reasoning:

- Keeps the fork's calibrated orchestrator and its full validation surface intact.
- Additive on canonical; existing 94-test suite stays green because nothing in the existing tests touches the shim.
- The shim is the natural "merge-to-streaming" enabler too; not throwaway.
- Bounded: the fork's `transport.mesh` usage is a finite set; once the shim covers those access patterns, the script runs.

---

## 6. Phase F plan: the compat-shim path

### F1 (canonical): build the compat shim on `steissberg-riverine-merged`

1. Catalog every `transport.mesh[X]` and `transport.constituent_dict[X]` access in `08_run_coupled_v3_smoke.py`. Produce a short table of the access patterns.
2. Add `ClearwaterRiverine.mesh` as a property returning an xarray Dataset view of the registry. Decide on read-only first; add write-through semantics for the specific keys the Stage-08 runner writes (initially: `water_temp_C` only; if the runner writes other variables, add them).
3. Add `ClearwaterRiverine.constituent_dict` as a property returning `self._constituents` (simple alias).
4. Add unit tests under `tests/test_fork_compat_shim.py` covering the read path, the write-through round-trip for `water_temp_C`, and any other access the runner needs.
5. Run the existing 94-test suite as the no-regression check.
6. Commit on `steissberg-riverine-merged`. No-Claude-attribution per project convention.

### F2 (modules-phase2-ESM-streaming): set up the canonical-test runner

1. Create a sibling output directory: `…/case_studies/santiam_salem/output/canonical_test_mumax_1_3/`
2. Copy `08_run_coupled_v3_smoke.py` to `08_run_coupled_v3_smoke_canonical.py` in the scripts directory.
3. Edit the copy: change the `CWR_SRC` `sys.path` injection from `/Users/todd/GitHub/ecohydrology/ClearWater-riverine-dev/src` to `/Users/todd/GitHub/ecohydrology/ClearWater-riverine/src`.
4. Ensure canonical's repo is at `steissberg-riverine-merged@<F1 head>`.
5. Verify that `constituent_dict["water_temp_C"]["is_intensive"] = True` is still being passed through (the fork already sets it at line 349; canonical's `Constituent.__init__` will read it).
6. Add `wet_dry_metric` parameter passed through to `cwr.ClearwaterRiverine(...)`. (TBD: probably "volume" or "both" to engage Phase D opt-in.)
7. Run with arguments matching `final_mumax_1_3`'s `run_provenance.json` exactly.
8. Wall clock expected ~12 minutes per the reference run.

### F3 (validation): run the comparator

1. Run `09_validate_2008_obs.py` against `canonical_test_mumax_1_3/`.
2. Compare the resulting `validation_metrics.csv` to the locked baseline metrics in Section 2.
3. Pass criterion (proposed): Salem T bias and RMSE each within 0.05 deg C and 0.05 deg C of the locked values. Other constituents within ~10% relative.

### F4 (report): update the close-out memo

If F3 passes, amend `design/all_phases_complete.md` Section 4 (Validation) to include the Willamette real-corridor validation result and remove the "not run as part of the reintegration test suite" caveat from Section 8.

If F3 does not pass, file the diagnostics: compare per-step canonical vs fork at the Salem cell, look for the first divergence, decide whether the divergence is real (canonical bug) or numerical (Phase-D opt-in not engaged correctly).

---

## 7. Cadence defect (deferred, tracked)

The Stage-08 orchestrator chains the transport operator-split cadence to the HDF flow-field time-axis spacing. `hdf_idx` increments once per `transport.update()`. Line 1209 in `08_run_coupled_v3_smoke.py` raises if more transport advances are requested than the HDF has indices.

This is not a bug for the validation runs because the production HDF is hourly (361 stamps over 15 days) and the master step is 1 hour, so transport cadence equals master cadence. The defect bites only with a sub-master HDF (the 30-day daily smoke product, or a hypothetical 6-hourly RAS product).

**Status:** Deferred. Not on the Willamette validation critical path. The clean fix (decouple transport from `hdf_idx`; reload flow only at HDF boundaries) would also future-proof the planned 15-minute hydraulics, but is a separate work item.

---

## 8. Other deferred items (tracked elsewhere)

- ESM uses real RAS2D edge-velocity (mean over each cell's bounding edges) in `velfix_mumax_1_5` (2026-05-18). The `final_mumax_1_3` baseline used the constant 0.5 m/s placeholder. Either baseline is fine for the canonical comparison provided we hold the configuration constant.
- Synthetic sediment is active (`no_sediment=false`). Replacement with observed sediment is a separate work item.
- Constant `ic_ap=1.6` for floating algae IC. Replacement with data-driven IC is a separate work item.
- Inflow CSV filename hazard (`synthetic_nutrients_meteorology_2008-09__obs_T.csv` contains observed data despite the "synthetic" name). Rename has been done in another session with a back-compat symlink. Confirm before launching.

---

## 9. Locked decisions (2026-05-20)

All five planning decisions were walked through one at a time and locked. F1 and F2 below proceed under these choices; revisit only if a hard blocker surfaces.

| # | Decision | Choice | Notes |
|---|---|---|---|
| 1 | Shim commit structure | **Single commit on `steissberg-riverine-merged`** | One logical unit, tests gate it, clean merge history. |
| 2 | Shim write-through scope | **Generalize for any registry variable** | The `mesh[name].loc[...] = arr` setter routes through `Constituent.set_at_time` / `registry.set_at_time`. A registered-name check errors on typos so an unknown key does not silently write to a non-existent variable. |
| 3 | `wet_dry_metric` value | **`"volume"` with `V_min=0.1 m^3`** | Matches the Phase-D regression guard configuration; engages Unit A WET_MASK and all downstream Phase-D gates. |
| 4 | `algae_mu_max_20` value | **1.3** (calibrated production choice) | Matches the `final_mumax_1_3` reference run; single 12-minute comparison rather than a three-point sweep. |
| 5 | Output directory | **`…/case_studies/santiam_salem/output/canonical_test_mumax_1_3/`** | Sibling under modules-phase2-ESM-streaming, next to the existing fork outputs. |

---

## 10. Verification commands quick-reference

```bash
# Verify canonical branch state before starting F1
cd /Users/todd/GitHub/ecohydrology/ClearWater-riverine
git status --short
git rev-parse --short HEAD       # expected: 100eacc or later
git log --oneline -3

# Verify modules-phase2-ESM-streaming reference run exists
ls /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/output/v3_smoke_15day_wind10m_final_mumax_1_3/validation_2008/validation_metrics.csv

# Verify HDF and BCs exist
ls -lh /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/data/synthetic/santiam_salem_subset_2008-09_hourly.p01.hdf
ls -lh /Users/todd/LargeProjects/ClearWater-riverine-case-study-Willamette/data/observed/derived_bcs_2008/synthetic_nutrients_meteorology_2008-09__obs_T.csv

# After F1: run the existing 94-test suite as no-regression check
cd /Users/todd/GitHub/ecohydrology/ClearWater-riverine
pixi run -e dev python -m pytest tests/ -v

# After F2: run the canonical comparator
cd /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming
python case_studies/santiam_salem/scripts/08_run_coupled_v3_smoke_canonical.py \
    --hdf /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/data/synthetic/santiam_salem_subset_2008-09_hourly.p01.hdf \
    --days 15 \
    --output-dir /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/output/canonical_test_mumax_1_3 \
    --diffusion 0.1 \
    --algae-mu-max-20 1.3 \
    --ic-water-temp 17.35 \
    --ic-nh4 0.02 --ic-no3 0.137 --ic-tip 0.029 --ic-dox 9.4 --ic-ap 1.6 \
    --wind-input-height 10.0 \
    --window-start 2008-09-01 \
    --met-csv-hourly /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/data/synthetic/met_KSLE_2008-09_hourly.csv \
    --nutmet-csv /Users/todd/LargeProjects/ClearWater-riverine-case-study-Willamette/data/observed/derived_bcs_2008/synthetic_nutrients_meteorology_2008-09__obs_T.csv \
    --continuity-correction all_edges

# After F3: run the validator
python case_studies/santiam_salem/scripts/09_validate_2008_obs.py \
    --run-dir /Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/output/canonical_test_mumax_1_3 \
    --validation-mode neighborhood_median \
    --radius-m 500
```

---

## 11. Companion documents and pointers

- `design/all_phases_complete.md`: the close-out memo for the Phase-D forward-port. Section 8 lists Willamette validation as the explicit next-step.
- `design/phase_d_complete.md`: Phase-D-specific close-out memo with the full unit ledger and per-unit validation counts.
- `design/streaming_chunking_implementation_plan.md`: the implementation plan that scoped Phases A through D.
- `…/ClearWater-modules-phase2-ESM-streaming/design/clearwater_setup_run_workflow_and_demo_compatibility.md`: cross-repository workflow analysis with the Two Hydraulic Tracks callout and the cadence-defect documentation. Contains the corrections from the 2026-05-16 audit.
- `…/ClearWater-modules-phase2-ESM-streaming/design/santiam_salem_demonstration_plan.md`: **stale** (2026-04-26); frames everything as Option A. Use only for historical context, not as the active plan.
- `~/Documents/Screenshots/Screenshot 2026-05-13 at 05.57.02.png`: the screenshot transcribed in Section 2 with the locked metrics.

---

## 12. Conversation context (received 2026-05-20)

The user provided context from prior Claude sessions covering:

- The validation track uses Option C (direct extraction of the completed USGS Sep-2008 unsteady), not Option A (rating-curve library, deferred to multi-year ESM horizons).
- Three compromises in the Option C validation: hourly vs. 15-min HDF output; four optional hydro variables (Face Flow, Cell Volume, Cell Velocity X/Y, Eddy Viscosity, Hydraulic Depth) reconstructed by CW-Riverine from volume-elevation tables rather than read directly; sparser 2008 observations than 2014.
- The v3 architecture: processes subclass `Process(ABC)` with `run(time, registry) -> None`; the v3 `Model` orchestrator is optional; Santiam-Salem deliberately bypasses it because it must multi-rate Riverine + TSM+NSM1 + ESM, which the one-shot Model loop cannot express.
- The cadence defect: real but doesn't affect Sep-2008 validation runs (hourly HDF means transport cadence equals master step which is correct). Fix proposed: decouple `transport.update()` from `hdf_idx`; reload flow only at HDF boundaries. **Deferred.**
- An ESM velocity gap (constant 0.5 m/s) was fixed in `velfix_mumax_1_5` (2026-05-18). Does not affect TSM or transport outputs, so does not change Salem T metric.
- The filename hazard: `synthetic_nutrients_meteorology_2008-09__obs_T.csv` contains observed data. Renamed in another-session work with back-compat symlink.
- The fork's orchestrator change to make the -0.30 / 0.62 metric appear was a single line: `"is_intensive": True` on water_temp_C in the `constituent_dict` passed to `cwr.ClearwaterRiverine(...)`. Canonical's `Constituent.__init__` reads the same flag (D1 commit `7f61582`), so the same one-line change reproduces the physics.

---

## 13. Data-loss event 2026-05-20 and regeneration plan

While preparing F2b execution we discovered the synthetic data the reference run consumed is gone from this machine. The user had deleted the parent folder thinking the git remote had everything, but `case_studies/santiam_salem/data/` is gitignored in both `modules-phase2-ESM` and `modules-phase2-ESM-streaming`, so the data was never pushed and the local copies were lost when the Trash was emptied. No Time Machine destination was configured.

Files lost from disk:

- `santiam_salem_subset_2008-09_hourly.p01.hdf` (the 158k-cell subset HDF the runner consumes as `--hdf`)
- `synthetic_forcing.csv` (Stage 03 output)
- `met_KSLE_2008-09_hourly.csv` and `met_KSLE_2008-09.csv` (Stage 06b NCDC download)
- `synthetic_sediment.nc` (Stage 05 output)
- `initial_layout.nc` (Stage 07 output)
- `library_subset.nc` (Stage 01 intermediate)

Sources still on disk (sufficient for full regeneration):

- Full USGS RAS HDF (5.7 GB, 2,035,430-cell mesh): `/Users/todd/LargeProjects/ClearWater-riverine-case-study-Willamette/data/hecras_model/Santiam_Salem/Santiam_Salem.p01.hdf` (Feb 2022)
- NWIS raw boundary-condition flows + temperatures at `…/Willamette/data/boundary_conditions/`
- Observation-derived BCs for Sep 2008 at `…/Willamette/data/observed/derived_bcs_2008/` (8 files: `bc_santiam_*`, `bc_upstream_water_temp_C`)
- The combined `observed_nutrients_meteorology_2008-09__obs_T.csv` (8.6 KB; intact via symlink under the legacy `synthetic_*` filename)
- All Stage 01-07 scripts in the cloned `ClearWater-modules-phase2-ESM` repo (now at `main @ d6808f5`)

### Regeneration sequence

Each Stage script lives at `…/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/scripts/`. The Stages produce inputs the runner (`08_run_coupled_v3_smoke_canonical.py`) needs.

1. **Stage 01** (`01_extract_hydraulic_library.py`): crop the 2.035M-cell RAS HDF to a 5x3 km UTM bbox (~158k cells); emit `library_subset.nc`. Input to Stage 04e and Stage 02. Expected ~5-15 min.
2. **Stage 04e** (`04e_extract_subset_timeseries.py`): extract 361 hourly stamps from the completed Sep 2008 unsteady run on the subset mesh; produces the timeseries NetCDF that Stage 06c transcodes. Expected ~5-15 min.
3. **Stage 06c** (`06c_transcode_to_hecras_hdf.py`): write the CW-Riverine-readable HDF `santiam_salem_subset_2008-09_hourly.p01.hdf`. Expected ~5-10 min.
4. **Stage 03** (`03_generate_synthetic_forcing.py`): build `synthetic_forcing.csv` from NWIS at Albany (USGS-14174000) and Santiam-at-Jefferson (USGS-14189000). Expected ~1-2 min.
5. **Stage 06b** (`06b_download_ncdc_met.py`): download Salem airport (KSLE ASOS) hourly met for Sep 2008; produces `met_KSLE_2008-09_hourly.csv`. Network-dependent; expected ~1-5 min.
6. **Stage 05** (`05_synthesize_sediment.py`): build `synthetic_sediment.nc` daily bed-change envelope. Expected ~1-5 min.
7. **Stage 07** (`07_compute_initial_layout.py`): elevation + inundation-frequency rules for the 7 species; produces `initial_layout.nc`. Expected ~1-5 min.

Stage 06 itself is skipped because its output (`observed_nutrients_meteorology_2008-09__obs_T.csv`) survived under a symlink.

Total expected wall-clock: ~30-60 min for the chain.

### Cost of the comparison shift

Regenerated files will not be bit-identical to whatever produced the locked `-0.30 / 0.62` baseline because the Stage scripts may have evolved between `a688d5b` (the reference run's git_head) and the current HEAD `d6808f5`. The Salem T metric should still land very close to `-0.30 / 0.62` because the Stage math has been stable, but the precise expectation is now:

- **Fresh fork baseline**: run the original `08_run_coupled_v3_smoke.py` against the regenerated inputs once; record its `validation_metrics.csv`. Call this `fork_repro_mumax_1_3`.
- **Canonical**: run `08_run_coupled_v3_smoke_canonical.py` against the same regenerated inputs; record its `validation_metrics.csv`. Call this `canonical_test_mumax_1_3`.
- **Compare**: fork_repro vs canonical, not canonical vs the locked screenshot. The screenshot stays as the historical reference but is not the active pass/fail gate.

### State checkpoint at the end of 2026-05-20

- F1 committed and pushed on canonical (`9483fd0`).
- F2a runner script written but not committed; lives uncommitted at `/Users/todd/GitHub/ecohydrology/ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/scripts/08_run_coupled_v3_smoke_canonical.py`.
- F2b first attempt errored on missing `synthetic_forcing.csv` (the lost-data discovery).
- Next concrete action: **Stage 01** regeneration. Then 04e -> 06c -> 03 -> 06b -> 05 -> 07. Then F2b retry against fresh inputs. Then fork-side reproduction. Then comparison.
