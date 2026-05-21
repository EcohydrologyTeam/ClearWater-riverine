# Phase G/H/I Design Specification

**Date:** 2026-05-21
**Author:** Phase F closeout audit + remediation plan
**Scope:** All follow-up work after Phase F (2026-05-21) across three repos:
- `ClearWater-riverine` on `steissberg-riverine-merged` (canonical, HEAD `b4b3368`)
- `ClearWater-modules-streaming` on `streaming` (worktree `/private/tmp/cw-modules-streaming`, HEAD `a6f3d30`)
- `ClearWater-data` on `steissberg-clearwater-data-chunked-reader` (HEAD `f8607e0`)

This document organizes 50+ audit findings plus the four Tier 3 items previously flagged into three sequenced phases: **G (BLOCKER fixes)**, **H (MAJOR correctness + parity items)**, and **I (MINOR/NIT cleanup + new feature wiring)**. Each phase has explicit goals, task-by-task fix sketches, validation criteria, dependencies, and effort estimates.

---

## 1. Background and audit summary

Phase F (2026-05-21) closed Tier 1 numerical parity (Salem T RMSE 0.56 °C vs streaming baseline 0.62 °C) and four of six Tier 2 robustness items (point_sources, decay_rate, diffusion-dispatch MVP, NaN/negative validation). The remaining four Tier 3 items previously identified — canonical zarr writer all-NaN bug, diffusion-dispatch HDF wiring, Internal-Cells dataset read, point-source sink handling — were known.

The deep code review uncovered three categories of additional defects:

### 1.1 BLOCKER findings (must fix before Phase G can proceed)

| ID | Repo | File:line | Defect |
|---|---|---|---|
| G-1 | riverine | utilities.py:213 | Missing `from pathlib import Path` import — `import clearwater_riverine` raises `NameError` on a clean install |
| G-2 | riverine | postproc_util.py:9 + others | Pre-existing circular import surfaced by `__init__.py` chain — `import clearwater_riverine` raises `AttributeError` |
| G-3 | modules-streaming | dox.py:834, n2.py:409, carbon.py:685 | Reaeration uses constructor-time scalar wind/velocity/flow — **silently produces wrong reaeration** when registry wind is time-varying |
| G-4 | data | zarr.py:78–82 | `ZarrDataStore` template uses uniform `pd.date_range` — non-uniform RAS stamps yield all-NaN output (the Bug A we already knew) |
| G-5 | data | zarr.py:161–163 | `drop_vars([c for c in coords if c != self.spatial_field])` compares string to list — always True — strips spatial coord on every chunk write |

Findings G-1 and G-2 mean canonical's *package import has never been exercised on a clean install*. Every Phase F test that "passed" did so because a streaming-fork install was shadowing canonical's source. This was missed because the Phase F regression suite was run inside the canonical worktree where the dev pixi env had imported canonical successfully at some earlier point.

### 1.2 MAJOR correctness findings

| ID | Repo | What |
|---|---|---|
| H-1 | riverine | point_sources arrays frozen on chunk-1's time axis in chunked mode → KeyError at first chunk boundary |
| H-2 | riverine | BC NaN validation runs BEFORE interpolation; docstring claims it catches interp-induced NaN but cannot |
| H-3 | riverine | `_calc_diffusion_array` CSV reader silently consumes first row as header |
| H-4 | riverine | Phase F regression suite tests shape/existence only; no numerical-correctness assertions for any new feature |
| H-5 | riverine | `_apply_continuity_correction` silently propagates NaN from `FLOW_ACROSS_FACE` into `ADVECTION_COEFFICIENT` |
| H-6 | riverine | `all_edges` continuity correction does not converge to `eps_converged=1e-12` on plan02 fixture (warning shipped, root cause unknown) |
| H-7 | modules-streaming | TSM `wind_input_height` log-law gate uses exact-equality `!= 2.0` — float-drift hazard |
| H-8 | modules-streaming | Reaeration internally rescales 2-m wind to 10-m unconditionally — height-convention disagreement with Temperature module's configurable `wind_input_height` |
| H-9 | modules-streaming | Process composition order matters for sibling-coupling (DOX reads Nitrogen, Phosphorus reads FloatingAlgae); silent one-substep lag if process order is wrong |
| H-10 | modules-streaming | Latent-heat-flux diagnostic naming ambiguity (signed vs magnitude) — invites downstream mass-balance closure bugs |
| H-11 | data | `ZarrDataStore.write` ignores `parameter_name` argument; mismatched `data.name` silently writes to wrong variable |
| H-12 | data | `DataArrayVariable.get_at_time` hardcodes the literal `'time'` for the guard while honoring `self.time_dimension` for the sel — silently returns the full array when `time_dimension != 'time'` |
| H-13 | data | `DataArrayVariable.get_at_time` uses exact-match `.sel` — off-by-microsecond raises `KeyError` |
| H-14 | data | `VariableRegistry.get` deprecation message references `get_data` method that does not exist |
| H-15 | data | `FloatVariable.set_at_time` silently mutates global scalar; `time` argument is discarded |

### 1.3 MINOR + NIT items + Tier 3 features

The 25+ MINOR/NIT findings range from missing-stacklevel warnings to typo'd type annotations to dead imports. Plus the four Tier 3 features previously known:

- **T3-A**: Diffusion-dispatch HDF wiring (register MANNINGS_N, FACE_VEL_X/Y, EDDY_VISCOSITY from io/hdf.py)
- **T3-B**: Internal-Cells dataset read (route through T2-A point_sources infrastructure)
- **T3-C**: Point-source sink handling on LHS diagonal
- **T3-D**: Canonical model_outputs.zarr writer time-grid fix (subsumed by G-4 in this spec)

These are wired into Phase I (the longest phase) so the foundation is solid before feature additions.

---

## 2. Phase G — BLOCKERS (must-fix-first, ~1 day)

### Goals

1. Restore canonical package import on a clean install.
2. Fix the silent reaeration-wind defect in modules-streaming.
3. Fix the two ZarrDataStore bugs that produce all-NaN / data-loss output.

### Tasks

#### G-1. Add missing `Path` import to canonical utilities.py

- **File:** `ClearWater-riverine/src/clearwater_riverine/utilities.py`
- **Severity:** BLOCKER
- **Fix:** Add `from pathlib import Path` at the top of the imports block.
- **Validation:** `python -c "import clearwater_riverine"` succeeds in a fresh venv with no prior install of the package.
- **Effort:** 5 minutes.

#### G-2. Resolve canonical circular import via `from __future__ import annotations`

- **File:** `ClearWater-riverine/src/clearwater_riverine/postproc_util.py`
- **Severity:** BLOCKER
- **Fix:** Add `from __future__ import annotations` at the top. This defers annotation evaluation and breaks the cycle: `clearwater_riverine/__init__.py` → `model.py` → `postproc_util.py` (no longer resolves `cwr.ClearwaterRiverine` at import time).
- **Validation:** `python -c "import clearwater_riverine"` succeeds without `AttributeError`; the existing pytest suite runs with the package imported (not shadowed). Re-run `tests/test_phase_f_regression.py` and confirm all 21 still pass against canonical's actual source.
- **Effort:** 5 minutes.

#### G-3. Wire registry-driven forcings into DOX/N2/Carbon reaeration

- **Files:** `ClearWater-modules-streaming/src/clearwater_modules_v3/processes/{dox,n2,carbon}.py`
- **Severity:** BLOCKER
- **Fix sketch:** In each process's `run(self, time, registry)` method (after the existing `water_temperature = registry.get_at_time("water_temperature", time)` pattern), add reads for `wind_speed`, `velocity`, `flow`, `topwidth`, `slope`, `shear_velocity`. Pass these values into the `_change_with_components` helpers instead of the constructor-frozen `self.<name>` scalars. Preserve the constructor-fallback pattern (`if name in registry: ... else: self.<name>`) that DOX already uses for `pressure_mb`, `ammonium`, and `salinity` (dox.py:705–713).
- **Validation:**
  1. Unit test: register a time-varying `wind_speed` DataArray with diurnal cycle, run a 24-hour DOX+TSM+NSM1 simulation, assert `atm_reaeration_rate` follows the wind cycle.
  2. Unit test: instantiate `Temperature(wind_input_height=10.0)` and `DOX()` with the same registry; verify both modules apply the same effective 10-m wind to their respective wind functions.
  3. End-to-end: re-run Phase F Santiam-Salem with the fix. Expected: T-bias residual closes from -0.22 °C toward zero (or surfaces a separate calibration issue that was previously masked).
- **Effort:** 4–6 hours (3 files, registry-fallback pattern, 3 new unit tests).
- **Out of scope for G-3 (defer to Phase H):** the height-convention disagreement between Temperature's configurable `wind_input_height` and reaeration's hard-coded `(10/2)**0.143` rescale. That is a design decision (H-7, H-8) that needs separate discussion.

#### G-4. Fix ZarrDataStore uniform-time-grid template

- **File:** `ClearWater-data/clearwater_data/io/zarr.py`
- **Severity:** BLOCKER
- **Fix sketch:**
  1. Add an optional `time_coord: pd.DatetimeIndex | None = None` kwarg to `ZarrDataStore.__init__` and `ChunkedZarrDataStore.__init__`. When supplied, use it directly as `self.time`. When not supplied, fall back to the existing `pd.date_range(start_date, end_date, freq=time_step)` for backward compatibility.
  2. Update `ClearWater-riverine/src/clearwater_riverine/model.py:__init_output_store` to pass the actual RAS time vector from `registry.get(VOLUME).time` as `time_coord`.
  3. Document in the docstring that `time_step` is used only when `time_coord` is None.
- **Validation:**
  1. Unit test: construct a `ZarrDataStore` with a non-uniform `pd.DatetimeIndex` (e.g., 59/60/61-minute jitter), write a chunk with the same stamps, read back, assert byte-equality.
  2. End-to-end: re-run Phase F Santiam-Salem with the fix. Expected: `model_outputs.zarr` is no longer all-NaN; canonical's per-constituent output validates without the `nsm1_history.nc` fallback path.
- **Effort:** 2–3 hours (substrate change + riverine wiring + tests).

#### G-5. Fix ZarrDataStore drop_vars list/string comparison

- **File:** `ClearWater-data/clearwater_data/io/zarr.py:161–163`
- **Severity:** BLOCKER
- **Fix sketch:**
  ```python
  keep = {"time", *self.spatial_field}  # spatial_field is already promoted to list at __init__
  data_clean = data.drop_vars([c for c in data.coords if c not in keep])
  ```
- **Validation:** Unit test that writes a chunk with extra coords on the spatial dim, reads back, and asserts the spatial coord survives.
- **Effort:** 30 minutes.

### Phase G validation gate

Before declaring Phase G complete, all of the following must pass:

1. Fresh-venv `pip install -e .` of all three repos, followed by `python -c "import clearwater_riverine; import clearwater_modules_v3; import clearwater_data"` succeeds.
2. `pytest tests/test_phase_f_regression.py` runs all 21 tests against canonical's actual source (not shadowed) with 21/21 pass.
3. End-to-end Phase F Santiam-Salem rerun: canonical's `model_outputs.zarr` contains finite per-constituent values; canonical's T bias closes toward streaming baseline (-0.30 °C target, allowable residual ≤ 0.2 °C).

### Phase G estimated effort: 1 day (8 hours)

---

## 3. Phase H — MAJOR correctness items (~3 days)

### Goals

1. Close the silent-correctness gaps surfaced by the audit.
2. Add numerical-correctness regression tests for every Phase F feature.
3. Resolve the modules-streaming height-convention disagreement and process-order race.
4. Make the canonical data substrate robust to time-resolution drift and naming conventions.

### Tasks

#### H-1. Refresh point_sources arrays on chunk-boundary

- **File:** `ClearWater-riverine/src/clearwater_riverine/model.py:__load_new_chunk`
- **Severity:** MAJOR
- **Fix sketch:**
  - Option A (cheaper): build point-source arrays against the full simulation time grid once at `Constituent.__init__`, indexed by `pd.date_range(start_datetime, end_datetime, freq=representative_dt)`. Reads still work in chunked mode because `get_at_time(name, t)` selects by stamp.
  - Option B (cleaner if non-uniform stamps): mirror the WET_MASK and ADVECTION_COEFFICIENT refresh pattern — re-load the CSV in `__load_new_chunk` and register fresh arrays for the new chunk's time window.
  - Decision: **Option B** for consistency with how WET_MASK and ADVECTION_COEFFICIENT are handled, and to avoid the assumption of uniform `representative_dt`.
- **Validation:** Two-chunk run with a point-source CSV that spans both chunks; assert the second chunk's transport sees the correct point-source contribution. This is the reproducer for the F3 finding.
- **Effort:** 2 hours.

#### H-2. Move BC NaN validation to after the .interp() step

- **File:** `ClearWater-riverine/src/clearwater_riverine/constituents.py:set_boundary_conditions`
- **Severity:** MAJOR
- **Fix sketch:** Add a second `_validate_constituent_values` call immediately after the `boundary.interp(time=target_time, method="linear")` step at line 290. Keep the existing pre-interpolation check (it catches malformed source CSVs); the post-interpolation check catches NaN introduced when the source time series does not span the simulation window. Update the inline comment to describe both checks honestly.
- **Validation:** Unit test that registers a BC time series ending 1 hour before `end_datetime`. Without the fix, `linear` extrapolation produces NaN at the trailing edge and the run silently dilutes. With the fix, the post-interp validation raises with a clear message naming the affected times.
- **Effort:** 1 hour.

#### H-3. `_calc_diffusion_array` CSV header handling

- **File:** `ClearWater-riverine/src/clearwater_riverine/utilities.py:_calc_diffusion_array`
- **Severity:** MAJOR
- **Fix sketch:** Decide whether the CSV requires a header. Recommendation: **require a header** (`cell_index,diffusion_coefficient`) and update the docstring to match. Then validate column names explicitly:
  ```python
  df = pd.read_csv(filepath)
  required = {"cell_index", "diffusion_coefficient"}
  missing = required - {c.lower() for c in df.columns}
  if missing:
      raise ValueError(f"Array diffusion CSV missing columns: {missing}")
  ```
- **Validation:** Round-trip test that writes a 3-row header-ful CSV, loads it via `_calc_diffusion_array`, and asserts the per-cell diffusion values match.
- **Effort:** 30 minutes.

#### H-4. Numerical-correctness regression tests for Phase F features

- **File:** `ClearWater-riverine/tests/test_phase_f_regression.py` (extend; do not replace)
- **Severity:** MAJOR
- **Fix sketch:** Add at minimum the following tests:
  1. **Continuity-correction residual reduction.** Construct a fixture (synthetic 4×4 mesh with a forced 5 % per-cell residual). Assert `max|r_i|` after `bc_only` is strictly below the initial residual on BC-adjacent cells; assert `max|r_i|` after `all_edges` is below `eps_converged` within `max_iter`. This locks the numerical claim of the port.
  2. **Decay-rate first-order check.** Quiescent (zero-flow) 1-cell mesh with `decay_rate = k > 0` and uniform IC; assert `c[t+1] ≈ c[t] / (1 + k*dt)` to within solver tolerance; compare against the analytical `c(t) = c_0 / (1 + k*dt)^n`.
  3. **Point-source mass-injection check.** On plan02 with a known point source `Flow_Rate=Q, Concentration=C` at cell 0, run one step and assert the realized mass increment at cell 0 matches `Q * C * dt` to within solver tolerance.
  4. **Newly-wet reconstruction on/off behavioral difference.** A fixture with at least one newly-wet cell where `reconstruct_newly_wet=True` lifts the c≈0 artifact and `=False` does not. Assert the difference is non-zero on that cell.
  5. **Chunked-mode point-sources roundtrip.** Two-chunk run; reproduces H-1 / F3.
  6. **NaN-in-FLOW_ACROSS_FACE defense.** Inject NaN at one edge of plan02's flow array and assert `register_advection_coefficient` either raises or warns. Currently silently propagates.
  7. **Internal-BC warning positive case.** Synthetic boundary-attributes DataFrame with `Type='Internal'`; assert the T2-E warning fires.
  8. **MMS-style implicit-Euler convergence.** Advection-diffusion analytical solution; refine in `dt` and assert L2 error scales as O(dt). Locks the convergence order against future Phase D/F changes.
- **Validation:** All new tests pass; coverage report shows lines in `utilities.py:_apply_continuity_correction`, `_apply_bc_only_correction`, `_apply_all_edges_correction`, `transport.py:decay-rate diagonal add`, and `linalg.py:_calculate_point_sources` are exercised.
- **Effort:** 1–1.5 days (8 new tests, may require building 1–2 small synthetic fixtures).

#### H-5. Defensive NaN guard in continuity_correction

- **File:** `ClearWater-riverine/src/clearwater_riverine/utilities.py:_apply_continuity_correction`
- **Severity:** MAJOR
- **Fix sketch:** At the entry to `_apply_continuity_correction` (immediately after the `nreal_count` and `V` extraction), add:
  ```python
  if not np.all(np.isfinite(adv_coeff)):
      n_nan = int(np.isnan(adv_coeff).sum())
      raise ValueError(
          f"FLOW_ACROSS_FACE / ADVECTION_COEFFICIENT contains {n_nan} NaN values "
          "at entry to continuity correction. The correction would propagate "
          "NaN into the LHS coefficient matrix and silently corrupt the WQ "
          "transport solve. Investigate the source HDF or the wet/dry "
          "amendments that may be writing NaN to flow_across_face."
      )
  ```
  Also replace the `print('here')` debug line in `linalg.py:511` with a `warnings.warn` or `raise ValueError`.
- **Validation:** Test 6 in H-4 above.
- **Effort:** 30 minutes.

#### H-6. Diagnose `all_edges` non-convergence on plan02

- **File:** `ClearWater-riverine/src/clearwater_riverine/utilities.py:_apply_all_edges_correction`
- **Severity:** MAJOR
- **Investigation plan:**
  1. Run plan02 with verbose iteration output and dump the residual at every refinement pass.
  2. Compute the condition number of `L = D @ D.T + ridge*I` on plan02. If `cond(L) > 1e10`, the ridge needs to be larger (currently `1e-14 * max(L.diag(), 1.0)`).
  3. Probe whether `max_iter=5` is sufficient on plan02 with a tighter ridge; if not, the convergence behavior of the iterative refinement on small meshes needs review.
- **Fix sketch (one of):**
  - Increase ridge to `1e-10 * max(L.diag(), 1.0)` and re-validate.
  - Add a one-pass post-projection refinement that re-applies `D @ c` and adjusts.
  - Tighten `eps_converged` only on large meshes; relax to round-off floor (e.g., `1e-9`) on small meshes.
- **Validation:** Convergence warning disappears on plan02; numerical-correctness test in H-4 (continuity-correction residual reduction) still passes.
- **Effort:** 4–6 hours (investigation + fix + test).

#### H-7. TSM `wind_input_height` exact-equality predicate

- **File:** `ClearWater-modules-streaming/src/clearwater_modules_v3/processes/temperature.py:1406`
- **Severity:** MAJOR (correctness-adjacent; user-impact: future drift hazard)
- **Fix sketch:** Replace `if self.wind_input_height != 2.0:` with `if abs(self.wind_input_height - 2.0) > 1e-12:`. Better still, remove the conditional entirely; at `wind_input_height = 2.0` exactly the log-law factor is `log(2/z0)/log(2/z0) = 1.0` and is correctly a no-op. The conditional is dead code optimization that introduces a float-equality hazard.
- **Validation:** Existing wind-function unit tests pass. New test: `Temperature(wind_input_height=2.0000001)` produces a log-law factor within 1e-7 of 1.0 (not exactly 1.0 — the conditional bypass is gone).
- **Effort:** 30 minutes.

#### H-8. Unify wind-height convention between Temperature and reaeration

- **Files:**
  - `ClearWater-modules-streaming/src/clearwater_modules_v3/utils/reaeration.py:kaw_20` (line ~167)
  - `ClearWater-modules-streaming/src/clearwater_modules_v3/processes/{dox,n2,carbon}.py` (callers)
- **Severity:** MAJOR (unit/convention mismatch)
- **Design decision required before implementation:** today, `Temperature` has a configurable `wind_input_height` and applies a log-law correction; `kaw_20` assumes 2-m wind input and internally rescales to 10-m via `(10/2)**0.143 ≈ 1.35`. A user setting `wind_input_height=10.0` for Temperature now applies the rescale on top of an already-10-m measurement, double-counting.
- **Proposed resolution:** Add `wind_input_height` to the reaeration utility surface; have callers pass it explicitly. `kaw_20` then applies the inverse correction to get back to 10-m if needed:
  ```python
  def kaw_20(wind_speed, *, wind_input_height: float = 2.0):
      if abs(wind_input_height - 10.0) > 1e-12:
          # Standard log-law to 10 m (z0=0.001 m for water surface)
          factor = math.log(10.0 / 0.001) / math.log(wind_input_height / 0.001)
          wind_speed = wind_speed * factor
      return ... # existing formula on 10-m wind
  ```
- **Validation:** With `wind_input_height=10.0`, `Temperature` and `DOX` see the same effective 10-m wind (regression test). With `wind_input_height=2.0` (the historical default), `Temperature` applies no log-law correction; `DOX` applies the existing `(10/2)**0.143` rescale.
- **Effort:** 3–4 hours (design, implementation, 3 callers, parametrized tests over (2 m, 10 m) × (height-aware, height-naive)).

#### H-9. Process-order race in modules-streaming

- **File:** `ClearWater-modules-streaming/src/clearwater_modules_v3/processes/__init__.py` and base `Process` class
- **Severity:** MAJOR
- **Fix sketch:** Add a class-level `upstream_processes: tuple[str, ...] = ()` declaration to the `Process` base class. Each Process subclass declares which sibling processes' rates it reads from (e.g., `DOX.upstream_processes = ('Nitrogen',)` because DOX reads `nitrification_flux_rate`). `Model.__init_model` then performs a topological-sort validation of the registered process tuple and raises if a reader is listed before its writer.
- **Validation:** Test that constructs `Model(processes=[DOX, Nitrogen])` raises with a message naming the dependency violation; `Model(processes=[Nitrogen, DOX])` succeeds.
- **Effort:** 4 hours (base class + per-Process annotations + Model validation + tests).

#### H-10. Latent-heat-flux diagnostic naming

- **File:** `ClearWater-modules-streaming/src/clearwater_modules_v3/processes/temperature.py:830–837`
- **Severity:** MAJOR (documentation-correctness)
- **Fix sketch:** Decide one of:
  - (a) Keep the diagnostic as the *signed* subtraction term; rename to `q_latent_subtraction_term` and document that it can be negative in condensation regime.
  - (b) Clamp the diagnostic to a positive magnitude and expose a separate `q_condensation` diagnostic for the negative branch.
- Recommendation: **(a)** for backward compatibility with existing diagnostic plotters; rename only the docstring to match the actual behavior.
- **Validation:** Test in the condensation regime (`e_air > e_sat`) that the diagnostic value flows correctly into `q_net = q_solar + q_sensible + q_sediment + q_longwave_down - q_longwave_up - q_latent`.
- **Effort:** 1 hour.

#### H-11. ZarrDataStore.write parameter_name handling

- **File:** `ClearWater-data/clearwater_data/io/zarr.py:107–109`
- **Severity:** MAJOR
- **Fix sketch:**
  ```python
  def write(self, data: ArrayLike, parameter_name: str) -> None:
      if data.name != parameter_name:
          data = data.rename(parameter_name)
      data.to_zarr(self.store_path, mode="a", consolidated=False, compute=True)
  ```
  Also remove the unused `prt = ...` debug assignment.
- **Validation:** Test that `store.write(da_named_wrong, parameter_name='c')` writes into the `'c'` slot, not creating a new `'wrong'` variable.
- **Effort:** 30 minutes.

#### H-12. DataArrayVariable.get_at_time time-dimension consistency

- **File:** `ClearWater-data/clearwater_data/variables/xarray.py:28–33`
- **Severity:** MAJOR
- **Fix sketch:**
  ```python
  def get_at_time(self, time: datetime) -> xr.DataArray:
      if self.time_dimension is not None and self.time_dimension in self.data_array.dims:
          return self.data_array.sel({self.time_dimension: time})
      return self.data_array
  ```
- **Validation:** Test parameterized over `time_dimension in {'time', 'stamp', None}` that the right time slice (or full array, when time-dim is None) is returned.
- **Effort:** 30 minutes.

#### H-13. DataArrayVariable.get_at_time tolerance kwarg

- **File:** `ClearWater-data/clearwater_data/variables/xarray.py` and `clearwater_data/variables/registry.py`
- **Severity:** MAJOR
- **Fix sketch:** Add an optional `tolerance: timedelta | None = None` kwarg to `Variable.get_at_time` (and thread through `VariableRegistry.get_at_time`). When supplied, use `method='nearest', tolerance=tolerance`. Document semantics: exact-match by default (preserves current behavior); user opts in to nearest-with-tolerance.
- **Validation:** Parametrized test over (exact match, off-by-microsecond, off-by-second) × (tolerance=None, tolerance=1s). Without tolerance: off-by-µs raises; with: resolves to nearest.
- **Effort:** 1.5 hours.

#### H-14. Fix VariableRegistry.get deprecation message

- **File:** `ClearWater-data/clearwater_data/variables/registry.py:55–59`
- **Severity:** MAJOR (UX correctness — users following the guidance hit AttributeError)
- **Fix sketch:** Either (a) rename `Variable.get` to `Variable.get_data` substrate-wide to match the message (large change; coordinate with all callers), or (b) update the message to refer to the actual method:
  ```python
  warnings.warn(
      "VariableRegistry.get is deprecated; use get_variable(key).get() instead.",
      DeprecationWarning,
      stacklevel=2,
  )
  ```
- Recommendation: **(b)** for short-term; schedule (a) as part of a substrate v0.4 release with a deprecation cycle.
- **Validation:** Following the deprecation guidance does not raise `AttributeError`.
- **Effort:** 15 minutes for (b); ~4 hours for (a) substrate-wide.

#### H-15. FloatVariable.set_at_time silent overwrite

- **File:** `ClearWater-data/clearwater_data/variables/float.py:35–40`
- **Severity:** MAJOR
- **Fix sketch:** Decide the contract:
  - (a) Raise `NotImplementedError("FloatVariable does not support per-time set; use set(value) to update the scalar globally")` if `time` is non-None.
  - (b) Accept silently and document loudly.
- Recommendation: **(a)** — silent global mutation behind a per-time API is the kind of latent bug that bites at production scale.
- **Validation:** Test that `fv.set_at_time(some_time, value)` raises with the documented message; `fv.set(value)` continues to work.
- **Effort:** 30 minutes.

### Phase H validation gate

1. All numerical-correctness tests added in H-4 pass.
2. End-to-end: re-run Phase F Santiam-Salem. Expected: T-bias residual closes further (from -0.22 °C toward 0 once height-convention is unified per H-8). RMSE remains ≤ 0.62 °C (streaming baseline) or improves.
3. New Phase G+H regression test that runs both canonical and a fresh streaming-shadow-free install of canonical against the same RAS HDF and asserts identical numerical output to within 1e-6.

### Phase H estimated effort: 3 days (24 hours)

---

## 4. Phase I — MINOR/NIT cleanup + Tier 3 feature wiring (~5 days)

### Goals

1. Land the four originally-flagged Tier 3 features: diffusion-dispatch HDF wiring, Internal-Cells dataset read, point-source sink handling, and (now subsumed by G-4) zarr writer fix.
2. Clean up the 25+ MINOR/NIT findings.
3. Add infrastructure that prevents future Phase F-class defects.

### Tasks

#### I-1. Diffusion-dispatch HDF wiring (originally Tier 3-A)

- **Files:** `ClearWater-riverine/src/clearwater_riverine/io/hdf.py`, `model.py`, `io/config.py`
- **Severity:** Feature (was blocked by Phase F MVP deferral)
- **Fix sketch:**
  1. Extend `RASHDFDataSource.__set_internal_paths` to add paths for `MANNINGS_N`, `FACE_VEL_X`, `FACE_VEL_Y`, `EDDY_VISCOSITY`, `CELL_EDDY_VISCOSITY_X/Y`.
  2. Add these to `self.temporal_variables` (or `self.static_variables` for `MANNINGS_N`) with proper space-dimension wiring.
  3. Modify `__read_static_variables` / `__read_temporal_variables` to read these conditionally — only when the HDF dataset exists. Emit a warning if the user has selected a non-constant diffusion method but the required dataset is missing.
  4. Extend the config schema to accept a structured `diffusion_coefficient`:
     ```yaml
     diffusion_coefficient:
       method: elder         # one of: constant, elder, eddy_viscosity, array
       alpha: 0.6            # method-specific
     ```
  5. Wire the method+params through `model.py:__init_model` to register `diffusion_method` and any method-specific scalars on the registry, where `calculate_coeff_to_diffusion_term`'s dispatcher already reads them.
- **Validation:**
  1. Unit test: load an HDF that contains `Cell Manning's n` → MANNINGS_N is in registry.
  2. End-to-end test: configure `method: elder` on an HDF that has the required variables; run; assert per-cell diffusion varies with velocity and depth.
  3. End-to-end test: configure `method: elder` on an HDF that lacks MANNINGS_N; assert clear error at init time (not at first transport step).
- **Effort:** 1.5 days.

#### I-2. Internal-Cells dataset read + point_sources routing (originally Tier 3-B)

- **Files:** `ClearWater-riverine/src/clearwater_riverine/io/hdf.py`, `constituents.py`, `linalg.py`
- **Severity:** Feature
- **Fix sketch:**
  1. Extend `__define_boundary_hydrodynamics` to read `Geometry/Boundary Condition Lines/Internal Cells` (shape `(N, 4)`, dtype `[(BC Line ID, Cell Index, Station Start, Station End)]`).
  2. Join with `Attributes` on `BC Line ID` to get BC name and time series.
  3. For each Internal-Cells row, create a synthetic point-source-style entry: `Cell_Index, Datetime, Flow_Rate, Concentration` from the BC line's time-series. Route through the existing T2-A `_load_point_sources` infrastructure.
  4. Replace the T2-E warning with a positive-case behavior; keep the warning only when neither External Faces nor Internal Cells is present (true error case).
- **Validation:**
  1. End-to-end test on the source Santiam-Salem HDF (un-subset): Internal-Cells BCs route through, the downstream WQ field matches the subset-extractor path within roundoff.
  2. Regression: T2-E warning still silent on subset HDF.
- **Effort:** 1 day.

#### I-3. Point-source sink handling on LHS diagonal (originally Tier 3-C)

- **Files:** `ClearWater-riverine/src/clearwater_riverine/transport.py`, `constituents.py`
- **Severity:** Feature
- **Fix sketch:**
  1. In `TransportEngine.run` per-constituent loop, immediately before the `decay_rate` diagonal modification, read the constituent's point-source flows at `current_time + time_step`. For sink cells (`Flow_Rate < 0`), add `|Flow_Rate|` to the LHS diagonal.
  2. Build the diagonal modification into the same per-constituent `A_solve` copy as the decay-rate addition to avoid two copies.
  3. Update T2-A's `_load_point_sources` to remove the "sinks not yet supported" warning when sink-handling is wired.
- **Validation:**
  1. Unit test on a 1-cell quiescent mesh with `Flow_Rate=-Q` and known initial mass: assert mass removal rate matches `Q * c[t+1]` per step.
  2. Regression: existing T2-A point_sources tests still pass for source cells.
- **Effort:** 4 hours.

#### I-4. MINOR/NIT cleanup batch

Lower-priority items from the three audits. Each is small individually; batched for one PR.

- **canonical**: F11 (point_sources interpolate-by-time), F12 (dt=0 guard), F13 (linalg `print('here')` → warn), F14 (drop unused `numba` import), F15 (decay_rate avoid CSR copy per step), F16 (unify dt derivation), F17 (cite canonical validation run in internal_bc_audit.md), F18 (centralize continuity-correction mode validation), F19 (document `has_point_sources` as public surface), F20 (note T1-G has no test by design).
- **modules-streaming**: F4 (arrhenius temperature clamp warning), F5 (NaN propagation comment in arrhenius_correction), F8 (preserve wind_c validation), F9 (decouple `examples` from package `__init__`), F10 (q_solar diagnostic asymmetry note).
- **data**: N1 (chunk window out-of-range guard), N2 (subset_time inverted/descending guard), N3 (CSVDataSource missing-index error message), N4 (CSV value_field arr.rename), N5 (CSV missing-pair `on_missing` kwarg), N6 (ZarrDataStore typo-tolerant kwargs), N7 (DataArrayVariable.set_at_time 1-D handling), NIT1–NIT5 typo/cleanup.

- **Effort:** 1 day (batched, ~30 files).

#### I-5. Substrate v0.4 deprecation cycle (data)

- **Files:** `ClearWater-data/clearwater_data/variables/registry.py`, `variables/base.py`, all callers
- **Severity:** Substrate-level cleanup
- **Fix sketch:** Coordinated rename of `Variable.get` → `Variable.get_data` to match the deprecation message. Bump substrate version to 0.4.0. Update all canonical/modules-streaming call sites to `registry.get_variable(key).get_data()`.
- **Validation:** All three repos' test suites pass with the rename. Deprecation warning fires correctly with `stacklevel=2`.
- **Effort:** 4 hours.

#### I-6. CI infrastructure

- **Files:** `.github/workflows/*` in each repo
- **Severity:** Infrastructure
- **Fix sketch:**
  1. Add a GitHub Actions workflow per repo that does `pip install -e .` in a fresh venv, runs `python -c "import <package>"` to catch G-1/G-2-class defects, then runs the test suite.
  2. Add a cross-repo integration job that installs all three repos and runs the canonical Phase F regression suite end-to-end against a small fixture.
- **Validation:** CI green on a clean PR. Workflow catches a re-introduction of G-1 or G-2 with a clear failure message.
- **Effort:** 4 hours.

### Phase I validation gate

1. All MINOR/NIT findings have either been fixed or moved to a "deferred — not blocking" tracking issue with explicit justification.
2. The four Tier 3 features (I-1, I-2, I-3, and the G-4-equivalent zarr fix) have end-to-end tests that pass.
3. CI catches G-1 and G-2-class defects on a clean install.
4. Re-run Phase F Santiam-Salem validation. Expected: full per-constituent parity with streaming baseline (all 6 constituents bit-identical or within 1e-6); T-bias and RMSE within 0.05 °C of streaming locked baseline.

### Phase I estimated effort: 5 days (40 hours)

---

## 5. Schedule and sequencing

### Dependencies

- **Phase G is a hard prerequisite for Phase H and I.** Until G-1 and G-2 are fixed, no test in canonical actually runs against canonical's source.
- **G-4 (zarr writer) blocks H-4 test 5 (chunked-mode point_sources roundtrip).**
- **G-3 (reaeration wind) should land before any re-validation against the Santiam-Salem baseline.** Otherwise the wind-driven gas exchange remains frozen and downstream metrics are misleading.
- **H-8 (wind-height convention) depends on G-3.**
- **I-2 (Internal-Cells) depends on T2-A point_sources being solid; H-1 (chunked-mode point_sources refresh) should land first.**

### Recommended sequence

| Phase | Calendar | Goals |
|---|---|---|
| **G** | Day 1 | Restore canonical import; fix reaeration; fix zarr writer; fix drop_vars |
| **H** | Days 2–4 | Correctness gaps; numerical-correctness test suite; height-convention; process-order race; data substrate consistency |
| **I** | Days 5–9 | Tier 3 features (diffusion HDF wiring, Internal-Cells, point-source sinks); MINOR/NIT cleanup; substrate v0.4; CI |

### Total estimated effort: ~9 working days (~72 hours of focused work)

### Off-ramp checkpoints

After Phase G: canonical is importable, no all-NaN outputs, reaeration follows wind. **This is a viable shipping state if scope is constrained.**

After Phase H: numerical correctness tests in place; all silent-failure surfaces closed; height-convention unified. **This is the recommended shipping state for "robust and flexible for thousands of applications."**

After Phase I: full feature parity with streaming reference; all Tier 3 features wired; substrate cleanup landed. **This is the long-term target state.**

---

## 6. Per-task task IDs for tracking

The task IDs in this spec (G-1 through G-5, H-1 through H-15, I-1 through I-6) are stable identifiers. When implementing, reference the task ID in commit messages and PR titles for traceability:

```
Phase G-1: add missing Path import to utilities.py
Phase H-5: defensive NaN guard in continuity_correction
Phase I-2: Internal-Cells dataset read routes through point_sources
```

Each task's validation criteria should be explicit in the PR description, and the per-task effort estimates in this spec should be checked against actual time spent (input for future phase planning).

---

## 7. Out of scope for this spec

The following items were identified during the audit but are explicitly out of scope:

1. **Substrate-level `is_intensive` exposure on `Variable`.** The streaming fork added a per-constituent `is_intensive` flag that landed in canonical's Constituent class. Whether the *substrate* `Variable` should also carry this concept is a design question deferred.
2. **Migration of canonical to use ChunkedZarrDataStore for non-chunked runs.** The current branching between `ZarrDataStore` and `ChunkedZarrDataStore` at `model.py:954–963` is functional but duplicates the time-grid construction logic. A future refactor could unify.
3. **TSM calibration improvements (wind_a/wind_b/wind_c).** The Edinger wind-function defaults are v1/Fortran-parity inherited; the residual T-bias after Phase G/H may surface that the calibration itself is the next limitation. A calibration study is its own scope.
4. **Postproc / plotting parity with streaming fork.** Streaming has methods on `ClearwaterRiverine` for plotting/animation that canonical exposes via `RiverinePlotter`. The API styles differ but the functionality is present.

These items can be opened as separate tracked issues / future phase candidates.
