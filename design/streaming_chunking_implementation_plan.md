# Streaming / Chunking Reconciliation — Implementation Plan

**Status:** Approved design; implementation in progress
**Date:** 2026-05-18
**Author:** Todd Steissberg
**Companion (analysis / what & why):** `design/streaming_reconciliation_clearwater_data.md`
**This document (how & sequence):** the implementation plan derived from that analysis plus the design decisions resolved on 2026-05-18.

This file is the actionable plan. The companion analysis doc establishes *what* the
fork vs. main-branch streaming approaches are and *why* "re-base, not replace."
This doc records the *resolved design decisions*, the *standing rules*, and the
*ordered implementation sequence* with its validation gate.

---

## 1. Scope & priorities

- **Focus:** the `ClearWater-riverine` + `ClearWater-data` streaming-chunking design
  — RAS HDF5 *reading* and ClearWater constituent *read/write* — to reduce memory
  pressure while preserving computational speed.
- **Priority consumer = NSM/TSM** (the completed v3 in `ClearWater-modules-streaming`).
  Hard constraint: do **not** break the `VariableRegistry` + `ChunkedZarrDataStore`
  contract v3 NSM/TSM already consume.
- **ESM = flexible, secondary, pluggable** — a hydraulics-only consumer at a
  documented seam; never the `ClearwaterRiverine` chunk driver. `refactor_strategy.md`
  (in the ESM-streaming repo) is ESM-Workstream-B input only, not a core driver.
- **v3 ↔ `ClearWater-modules` merge = future / out of scope.** This design only
  stays compatible with v3's existing contract.
- **Flexibility principle:** every I/O path stays behind the `clearwater_data`
  `DataSource`/`VariableRegistry` seam so the strategy is swappable.

---

## 2. Resolved design decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | RAS input = windowed temporal reads via an extended `RASHDFDataSource` + re-based static-geometry `mesh_cache` + a bounded live time-window, all behind the `VariableRegistry`/`DataSource` seam | Best memory↔speed balance for long NSM/TSM coupled runs; matches companion §6; avoids both the resident-mesh blowup and the naïve per-window re-read I/O cost |
| **D2** | B1 = an **additive** `ChunkedZarrDataSource` / `read_chunk` in `clearwater_data`, with round-trip `write_chunk → read_chunk` tests | Fills an unimplemented Protocol stub; doesn't touch `write_chunk`/eager `read`, so v3 is unaffected unless it opts in; one shared blessed reader for riverine + v3 + ESM |
| **D3** | Output stays on the blessed `ChunkedZarrDataStore` (Zarr v3, `region="auto"`); **B2** spatial-chunk bug fixed with a shape-assert test | Forced by the NSM/TSM contract priority; B2 is a correctness defect, not a choice |
| **B3/B4** | Uniform-cadence design (the blessed store's `time_step`) + a loud precondition guard (error if RAS stamps are not a uniform grid, or `chunk_size` is not an integer multiple of the step) + keep the fork's scalar/array-`dt` and tolerance-boundary handling as cheap robustness. **No** non-uniform machinery in `clearwater_data`. | **Verified:** Willamette / Corvallis-Santiam-Albany RAS hydraulics are uniform (daily baseline → uniform 15-min rebuild → hourly USGS parent; all regular grids). The historical problem there was cadence being *uniformly too coarse*, not irregular. Full non-uniform machinery would be over-engineering for the real targets. |
| **D4** | Resume/checkpoint substrate unified on Zarr v3, read back via the B1 reader; `npz`/JSON sidecars retained | D2's v3 reader removes the only reason the fork needed a separate Zarr-v2 artifact; single blessed substrate. **Caveat:** rolling-2-slot + flux-continuity invariants must be re-validated on v3 (v3 `region`/consolidated semantics differ from v2 `append_dim`) |
| **D5** | The net-new layer (in-memory release + process-per-chunk checkpoint/resume + cross-boundary flux/mass continuity) stays in `clearwater_riverine`, re-based on the blessed reader/store; `clearwater_data` stays generic (B1+B2 only) | This layer encodes transport-solver semantics (rolling-2-slot, WQ mass continuity), not generic data-layer concerns; keeping it out of the shared substrate protects the v3 contract and contains scope. The release *pattern* is designed to be extractable later if v3 ever needs it. |
| **D6** | ESM consumes hydraulics-only via the shared `RASHDFDataSource` windows (or a hydraulic Zarr) at its daily cadence through its own `SharedDatasetView`/`TemporalAggregator` + a **new duration/exceedance reducer**; never the `ClearwaterRiverine` chunk driver | Companion §5/§6 + the ESM-secondary priority. The duration/exceedance reducer is net-new work on the **ESM-streaming side**, out of scope here but recorded as the seam's requirement |

### Note on the companion analysis doc

`streaming_reconciliation_clearwater_data.md` frames B3/B4 as the *code's*
fragility to non-uniform cadence and, in §3/§7, reads as though the production
data may be irregular. **That premise was checked and corrected here:** the
Willamette/Albany production cadence is uniform. The companion doc's §9 open
question — that `design/refactor_strategy.md` "was not present" — is also
resolved: it lives in `ClearWater-modules-phase2-ESM-streaming/design/` (it is
ESM-Workstream-B-specific). The companion doc was a source-inspection-only
analysis and did not run the suite, so it does not record the canonical
LHS-assembly crash discovered and fixed here (see F0).

---

## 3. Standing rules

- Every `git commit` and every `git push` requires a fresh, explicit, per-action
  user "yes." No Claude/Anthropic/AI mention or `Co-Authored-By` trailer in any
  commit message, PR title, or body.
- **Any `ClearWater-data` change goes on a dedicated branch off `main` — never a
  commit to `clearwater_data` main.** Merge-back is a separate, explicit PR/review
  step. (`clearwater_data` is the shared substrate v3 NSM/TSM consume.)
- Riverine work happens on `steissberg-riverine-merged` (branched from canonical
  `ClearWater-riverine` `main` @ `9bd4470`, PR #135 / `clearwater_data` base).
- The reconciliation is a **curated re-application**, not a `git merge` of the
  streaming fork (different lineage; a merge would re-pollute canonical with the
  legacy architecture).
- There are **two complementary port maps**, covering disjoint slices of the fork:
  the **numerical-correctness** map (dry→wet, wet→dry mass leak, signed advection,
  multi-BC ghost, `is_intensive`, the Step-4 subsystem) and the **streaming/chunking**
  map (companion §10). Neither supersedes the other; both feed this reconciliation.

---

## 4. Implementation sequence

Each phase is validated against the calibrated mass-balance gate (Phase B).

### F0 — Canonical LHS crash fix — **DONE**

Committed `0d7aff9` on `steissberg-riverine-merged` (not pushed). `linalg.py`
`flow_in_indices = np.where(...)` was missing the `[0]` its three sibling
`np.where(...)[0]` calls have, so `len(flow_in_indices)` returned 1 (tuple
length) instead of the inflow-edge count, undersizing `length_of_values` in
`__init_matrix_values` and overflowing `__fill` on plans with internal inflow
edges (plan01/05/06/07/08). One-token fix; behaviorally sound
(`a[(arr,)] == a[arr]` for 1-D); also corrects the always-true
`len(flow_in_indices) > 0` guard in `__fill_advection_values`. This is a
canonical-only refactor regression and a prerequisite for any suite run.

### Phase B — Calibrate the validation gate

`tests/test_final_mass.py` has been rewritten onto the config-driven /
`VariableRegistry` API (uncommitted). Calibrate it so it is *discriminating*:
realistic per-plan conditions and tolerances tight enough that the tidal
multi-boundary plans (plan06/07/08) **show** the wet→dry / multi-BC mass leak
the numerical ports fix, while the steady/single-boundary plans (01/02/04/05)
pass. Method: measure each plan's actual closure error on canonical first, then
set per-plan tolerances / `xfail`-with-reason from the data (the fork's
bound-based `_mass_bal_residual_bound*` helpers do not exist in canonical
`postproc_util`; do not port them in Phase B). Commit the harness rewrite as one
coherent unit. **This gate validates every phase below.**

### Phase A — `clearwater_data` branch (B1 + B2 only)

On a new branch off `clearwater_data` main (never main):
1. **B2** — fix the spatial-chunk-extent bug in `ChunkedZarrDataStore._init_zarr_store`
   (use the per-field point count, not `len(spatial_field_values)`); add a test
   asserting chunk shape `== (chunk_length, n_cells)`.
2. **B1** — implement an additive `ChunkedZarrDataSource` / `read_chunk` returning
   a lazily-sliced `xr.open_zarr(...).sel(time=slice(...))` window without
   materializing the whole array; round-trip `write_chunk → read_chunk` test.
3. Uniform-cadence precondition guard helper (loud error if stamps are not a
   uniform grid or `chunk_size` is not an integer multiple of the step).
   Backward-compatible throughout; v3 untouched unless it opts in.

### Phase C — Re-base the riverine streaming layer (`steissberg-riverine-merged`)

1. Swap the fork `io/hdf.py` window reader → extended `RASHDFDataSource`
   (windowed temporal reads); keep the fork `mesh_cache.py` static-geometry
   cache layered on top (D1). Everything behind the `VariableRegistry`/`DataSource`
   seam.
2. Swap the hand-rolled Zarr-v2 flush → `ChunkedZarrDataStore` (Zarr v3,
   `region="auto"`) for constituent output (D3).
3. Re-base the in-memory release + checkpoint/resume onto the v3 store; resume
   reads via the B1 reader (D4); the layer stays in `clearwater_riverine` (D5).
   **Preserve and re-validate** the rolling-2-slot solver-continuity invariant
   and cross-boundary flux/mass continuity.
   - **Decomposed (2026-05-19) into C3a + C3b, each guard-gated.** Canonical's
     one-chunk-resident model already subsumes the fork's whole-window
     NaN-`release` machinery, so this is a *capability re-base*, not a
     verbatim port of `_release_to_stream`/`checkpoint`.
   - **C3a DONE:** cross-chunk mass-flux continuity. The global balance only
     needs time-integrated boundary sums + start/end domain snapshots, so a
     **lean fixed-size accumulator** (Option B) folds each chunk's
     contribution at `__finalize_chunk` (interior chunks drop the shared
     overlap slot; the final chunk keeps it; per-transition mass flux
     partitions exactly). `_calculate_mass_flux` register → `overwrite=True`
     (fixes the 6th-defect crash). C2 loud guard lifted. Non-chunked
     `calculate_global_mass_balance` path byte-identical (new optional
     `chunk_accumulator` only). Validated: chunked closure ≡ non-chunked to
     ~7 sig figs, all 7 plans; suite 21 pass / 6 skip.
   - **C3b NEXT:** checkpoint/resume on the resolved v3 substrate
     (`write_chunk` + `.npz` + JSON; resume via B1 `read_chunk`).
4. B4 riverine-side: tolerance-based chunk-boundary detection (`>=` next
   boundary) + scalar-vs-array `dt` handling in `__increment_timestep`.
   Validate each step against the Phase-B gate (no closure regression).
   - **C2-discovered (2026-05-19), C4 to fix:** `__init_chunks` uses
     `pd.date_range(start, end, freq=chunk_size)[1:-1]`. When
     `(end - start)` is not an exact integer multiple of `chunk_size`,
     `date_range`'s last element is `< end`, so `[1:-1]` drops a
     *legitimate interior* chunk boundary — the final chunk then spans
     ~2× `chunk_size`, exceeding the memory ceiling `chunk_size` is meant
     to bound. Exact-grid runs are unaffected; the C2 chunked oracle
     deliberately uses even splits to isolate this from the v3-write check.

   Note (C2 done): step 2 needed no fork code — §10 RE-BASE, canonical
   already on the blessed `ChunkedZarrDataStore.write_chunk(region="auto")`
   v3 path. C2 instead added the missing chunked-path test oracle (the
   Phase-B guard was non-chunked only), found+loud-guarded the chunked
   `mass_flux` re-registration crash (6th defect; continuity is step 3's),
   and proved the v3 chunked write reproduces the non-chunked field.

### Phase D — Numerical-correctness ports (separate workstream)

The PORT-1..N solver fixes from the numerical-correctness port map (dry→wet,
wet→dry mass leak, signed advection, multi-BC ghost, `is_intensive`, the Step-4
subsystem), each validated against the calibrated Phase-B gate. Independent of
Phase C; can follow or interleave once Phase B exists. PORT-1/PORT-2 are
expected to flip plan06/07/08 from the Phase-B expected-fail state to pass.

---

## 5. Out of scope (tracked elsewhere)

- The ESM duration/exceedance reducer and wiring `SharedDatasetView` /
  `TemporalAggregator` into ESM's `increment_timestep` — ESM-streaming side.
- The v3 ↔ `ClearWater-modules` merge — a future step.
- The numerical-correctness port map detail — a parallel workstream sharing only
  the Phase-B gate.

---

## 6. Dependencies & risks

- Phase C step 3 (resume/read-back) depends on Phase A B1.
- Phase D validation depends on Phase B.
- **Risk:** re-basing release/checkpoint onto Zarr v3 must preserve the
  rolling-2-slot + flux-continuity invariants; v3 `region`/consolidated
  semantics differ from the fork's v2 `append_dim` — re-validate explicitly.
- **Risk:** the blessed per-window HDF read trades the fork's resident-mesh
  memory for repeated I/O on long runs; `mesh_cache.py` mitigates only static
  geometry. Benchmark Phase C against the speed acceptance before finalizing.
- **Acceptance target:** the 500,000-cell, 92-day Willamette/Albany run completes
  within the 16 GB workstation budget without OOM, and no slower than the
  pre-reconciliation baseline.
