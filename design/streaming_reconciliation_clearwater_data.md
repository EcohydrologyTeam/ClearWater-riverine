# Streaming / Chunking Reconciliation: hand-rolled fork vs. `clearwater_data` + `clearwater-riverine` main

**Status:** Analysis / decision input for the streaming-reconciliation plan
**Date:** 2026-05-18
**Author:** Todd Steissberg
**Scope:** Whether and how the hand-rolled chunking/streaming methodology in
`ClearWater-Riverine-streaming` can be replaced by the Zarr-chunking approach
established on the **main** branches of `ClearWater-data` and `ClearWater-riverine`.
A secondary section covers the ESM consumer, which has a different timestep and
process set than TSM/NSM.

---

## 0. Provenance (what was compared)

| Repo | Branch | HEAD | Role |
|---|---|---|---|
| `ClearWater-data` | `main` | `81689d4` | Generic xarray+dask+Zarr interchange layer |
| `ClearWater-riverine` | `main` | `9bd4470` (Merge PR #135 `refactor-clearwater-data`) | "Blessed" data-access path consuming `clearwater_data` |
| `ClearWater-Riverine-streaming` | `dry-cell-fix` | `df6650f` | Hand-rolled streaming/chunking fork |
| `ClearWater-modules-streaming` | `nsm2-features/step4` | `eeb7591` | v3 NSM2/TSM peers (reference: already on `clearwater_data`) |
| `ClearWater-modules-phase2-ESM-streaming` | `streaming` | `85cae10` | ESM consumer (different timestep/processes) |

The chunking-relevant files on `clearwater-riverine` `main` (`model.py`,
`io/hdf.py`, `constituents.py`) were verified byte-identical to the working
branch `steissberg-riverine-merged`, so all findings below apply to `main`.

---

## 1. Executive summary

**Corrected premise.** The reconciliation was triggered by the belief that
LimnoTech had already completed the HEC-RAS chunked-streaming work in
`ClearWater-data`, making the fork's hand-rolled version redundant. That is only
partially true:

- `ClearWater-data` is a **generic xarray + dask + Zarr interchange layer**.
  It contains **no HEC-RAS, HDF5, UGRID, or mesh-topology code**, **no chunked
  reader**, and does time-only chunking with at least one spatial-chunk bug.
- `ClearWater-riverine` `main` (PR #135) is the actual "blessed" path. It
  *consumes* `clearwater_data`, but uses the chunked Zarr store **for model
  output only**. Hydraulic *input* is re-read from `.p##.hdf` per time-window
  via `RASHDFDataSource` (h5py, eager). **There is no persisted, blessed
  chunked-Zarr hydraulic store anywhere** to drop in.

**Verdict: re-base, do not replace.** The main branches and the fork solve
**partly-overlapping, partly-disjoint** problems. The overlap (genuine
duplicate work) is the data-access plumbing: the HDF→xarray windowed reader and
the Zarr output writer. The disjoint part (fork-only, net-new, *not* in
`clearwater_data` or `clearwater-riverine` main) is in-memory release,
process-per-chunk checkpoint/resume, and cross-boundary flux/mass continuity —
the capabilities the fork was actually built for.

The reconciliation is therefore: **delete the duplicated plumbing and adopt the
blessed reader + Zarr store; keep the fork's release/checkpoint/continuity layer
re-based on top of it; fix four `clearwater_data` blockers first.**

---

## 2. The fundamental architectural difference

These are **not two implementations of the same thing**.

### Main branches = "stateless re-read" chunking

- Hydraulics re-read from `.p##.hdf` per window:
  `RASHDFDataSource.read_chunk()` (`ClearWater-riverine/src/clearwater_riverine/io/hdf.py:348-379`),
  h5py eager numpy slice (`io/hdf.py:80-101`, slice at `:93`).
- Only **one chunk window** of the 4 temporal hydraulic variables is resident:
  `__load_new_chunk()` unregisters and re-registers them each rollover
  (`src/clearwater_riverine/model.py:389-411`); topology/static stay resident
  for the whole run.
- Chunk driver is a clean first-class abstraction:
  `chunk_size: Optional[timedelta]` (`model.py:101`, parsed `:123-124`),
  `__chunked_mode` (`:139`), `__init_chunks` →
  `__chunk_ends = pd.date_range(start, end, freq=chunk_size)[1:-1]`
  (`:380-386`), `update()` → `__transport_chunked()` (`:414-418`):
  `if __current_time in __chunk_ends: __finalize_chunk(); __load_new_chunk()`.
- `clearwater_data.io.zarr.ChunkedZarrDataStore` is used **only as the output
  sink** for computed constituents (`model.py:356-377`, `__finalize_chunk`
  `:421-436`, `ChunkedZarrDataStore.write_chunk` with `region="auto"`
  `ClearWater-data/clearwater_data/io/zarr.py:109-133`).
- **No checkpoint/resume and no in-memory release** — it never holds more than
  one window, so it does not need them (verified: no
  `checkpoint`/`from_checkpoint`/`release`/`append_dim` in `main` `src`).

### Fork = "resident-mesh + flush/release + checkpoint" chunking

- Whole simulation-window mesh allocated up front via a one-shot
  `datetime_range` selection (`io/hdf.py` `_parse_dates`), not an iterating
  input loop. h5py `dataset[()]` materializes the full window then slices
  (`io/hdf.py:121-124`). NumPy-backed mesh (the design memo explicitly rejects
  a dask-backed mesh: `design/cw_riverine_streaming_in_memory_release.md:44-50`).
- Within-window chunking = Zarr time-slab **flush every `streaming_interval`
  steps** (default 100) — `_flush_to_stream` (`transport.py:1691`) writes
  `mesh.isel(time=slice(start,end))` to Zarr **v2** (`mode='w'`, then
  `mode='a', append_dim='time'`, `transport.py:1738-1741`) — followed by
  **in-place NaN release** of flushed slots, `_release_to_stream`
  (`transport.py:1754-1828`), keeping a rolling 2-slot live window for solver
  continuity. Memory ceiling ≈ `(streaming_interval + 2) × (n_face + n_edge) ×
  Σ dtype_bytes`.
- Orchestrator-level **process-per-chunk checkpoint/resume**: `checkpoint()`
  (`transport.py:1893`), `from_checkpoint()` (`transport.py:1981`); mesh = Zarr
  v2 consolidated, fluxes = `npz`, metadata = JSON. Built to survive macOS
  jetsam on multi-day 587k-cell runs and to make long runs resumable —
  problems the main branches do not address.

---

## 3. Side-by-side comparison

| Dimension | Hand-rolled fork (`dry-cell-fix`) | Main branches (`clearwater-riverine` `9bd4470` + `clearwater-data` `81689d4`) |
|---|---|---|
| Input HDF read | h5py `dataset[()]` → full window, slice (`io/hdf.py:121-124`); one-shot `datetime_range` | h5py per-window `read_chunk` re-read (`io/hdf.py:348-379`); iterating chunk loop |
| Chunk abstraction | implicit: `streaming_interval` (steps) flush threshold | first-class: `chunk_size: timedelta` driver (`model.py:101,139,380-418`) |
| Resident memory | whole window mesh, then NaN-release flushed slots | only **one** chunk window of 4 temporal vars; topology resident whole run |
| Zarr role | streaming **output + resume substrate**; Zarr **v2**, `append_dim='time'` | constituent **output only**; Zarr **v3**, `region="auto"` |
| Chunked Zarr reader | own `read_streamed` lazy `xr.open_zarr` slab (`transport.py:1866`) | **none** — `read_chunk` is an unimplemented Protocol/ABC stub (`clearwater_data/io/base.py:19,78`); `ZarrDataSource.read` eagerly `.compute()`s the whole variable |
| Cross-boundary state | full live mesh + `fluxes.npz` + per-constituent state via checkpoint; rolling-2-slot invariant | **constituent IC only** (`reset_initial_conditions`, `model.py:404-408`); no flux/mass accumulator |
| Checkpoint / resume | **yes** (`checkpoint`/`from_checkpoint`) | **none anywhere in src** |
| Static geometry cache | `io/mesh_cache.py` gzipped pickle, HDF-hash-keyed | **none** (re-read from HDF each run) |
| Irregular RAS cadence | handled in transport math via per-step `dt` array | **fragile**: `__increment_timestep` (`model.py:216`) breaks on array `dt`; chunk boundary is exact-timestamp equality (`model.py:415`) → `chunk_size` must be an integer multiple of a uniform RAS step |

---

## 4. Duplicated vs. net-new

### 4.1 Genuinely duplicated (the original mistake — remove from the fork)

1. **HDF→xarray windowed reader.** Fork `io/hdf.py` `_hdf_to_xarray`
   ≡ main `RASHDFDataSource.read_chunk`. Both h5py, eager, in-memory. Main's
   is better-factored (implements the `clearwater_data`
   `DataSource`/`ChunkedDataSource` protocol).
2. **Zarr output of computed fields.** Fork's hand-rolled time-slab flush
   (Zarr v2, manual `append_dim`) ≡ main's `ChunkedZarrDataStore.write_chunk`
   (Zarr v3, `region="auto"`). Main's is the standardized version, and is
   already what the v3 NSM2/TSM peers in `ClearWater-modules-streaming`
   consume (`src/clearwater_modules_v3/model.py:59,419,434`).

### 4.2 Fork-only and **not** duplicated (do not discard)

The main branches have no equivalent, and `clearwater_data` does not provide:

- **In-memory NaN release** bounding RSS on a resident multi-day mesh.
- **Process-per-chunk checkpoint/resume** (macOS-jetsam survival +
  resumability — the motivating reason the fork exists).
- **Cross-boundary flux/mass continuity** beyond a single constituent IC.
- **Static-geometry disk cache** (a net improvement over main re-reading
  topology every run).
- **Robust irregular-RAS-cadence handling** — here the fork is *more* correct
  than main, whose `__increment_timestep` and exact-timestamp chunk-boundary
  logic require a uniform RAS step.

---

## 5. ESM consumer (different timestep / processes)

ESM is the easiest of the downstream consumers to reconcile because it has
**zero dependence on the riverine transport solver** — it consumes only
hydraulic geometry/forcing.

- **Narrow contract.** ESM needs 3 required per-cell `(nface,)` arrays —
  `water_surface_elev`, `depth`, `velocity` — plus static geometry
  (`face_x`, `face_y`, `cell_area`) and optional T/N/P/sediment/ice
  (`ClearWater-modules-phase2-ESM-streaming/src/esm/model.py:250-293`,
  `HydraulicData`).
- **Must NOT use `ClearwaterRiverine`'s chunk driver.** That driver is welded
  to constituent transport (requires a `constituents:` config and a transport
  solve). The only cleanly reusable unit for ESM is `RASHDFDataSource` (+ a
  Zarr store) — not `ClearwaterRiverine`.
- **Different timestep is the real constraint.** ESM runs **daily**
  (`esm/model.py:330` `dt_days: float = 1.0`); RAS/transport run sub-daily.
  ESM does no internal aggregation. Process semantics differ by field:
  - growth / GDD / nutrient uptake → daily **mean** (correct as-is)
  - **scour mortality → daily *max* velocity** (`esm/processes/mortality.py:34-83`)
  - **drowning / drying / recession → sub-daily WSE *peak + duration***
    (`esm/processes/mortality.py:84-255`,
    `esm/processes/germination.py:116-135`) — cannot be reconstructed from a
    daily mean.
- ESM already has the right reduction layer **dormant in-tree**:
  `src/esm/coupling/temporal_aggregator.py` (`TemporalAggregator` with a
  per-field reducer table — mean for T/N/P, **max for velocity/depth**) plus
  `SharedDatasetView.get_field_window`. **Not wired into `increment_timestep`**
  (`esm/model.py:393-396,2151-2154`).
- **Gap independent of the store choice:** no reducer computes
  *duration/exceedance* (hours-above-threshold) for physically-correct
  drowning/drying. Must be added regardless of which store is used.
- **Out of scope for this decision:** sediment/bed-change comes from a separate
  RAS sediment product (`synthetic_sediment.nc`), never from `transport.mesh`
  or the hydraulic store.

---

## 6. Recommended target architecture

```
RAS .p##.hdf
   │
   ▼
clearwater_riverine.io.hdf.RASHDFDataSource     ← single blessed HDF reader
   │  (implements clearwater_data DataSource / ChunkedDataSource protocol)
   ├───────────────────────────────┐
   ▼                               ▼
ClearwaterRiverine (transport)     ESM (hydraulics-only, daily)
  → ChunkedZarrDataStore (Zarr v3)   → RASHDFDataSource windows OR hydraulic Zarr
    constituent OUTPUT               → SharedDatasetView + TemporalAggregator
  + KEEP (re-based) fork layer:        (sub-daily → daily: mean / max /
    - in-memory NaN release             NEW duration reducer)
    - process-per-chunk checkpoint     → esm.HydraulicData (contract unchanged)
    - cross-boundary flux continuity
   │
   ▼
clearwater_data.VariableRegistry   ← shared data model (v3 NSM2/TSM already use)
```

Concretely:

1. **De-duplicate the reader.** Replace the fork's `io/hdf.py` window reader
   and hand-rolled Zarr-v2 output with `clearwater_riverine.io.hdf.RASHDFDataSource`
   + `clearwater_data.io.zarr.ChunkedZarrDataStore` (Zarr v3) + the
   `DataSource` protocol and `VariableRegistry` data model. This aligns
   Riverine-streaming, modules-streaming, and ESM-streaming on the contract the
   v3 NSM2/TSM peers already consume.
2. **Keep the fork's release + checkpoint/resume + flux continuity**, re-based
   on top of the blessed reader as a layer the main branches do not have.
3. **For ESM:** wire the dormant `SharedDatasetView` + `TemporalAggregator`
   into `increment_timestep`, add a duration/exceedance reducer, and feed it
   `RASHDFDataSource` windows or a hydraulic Zarr store. ESM never touches
   `ClearwaterRiverine`.

---

## 7. Blockers checklist — `clearwater_data` (must fix before adoption)

These block adoption of the blessed Zarr path regardless of scope. File
references are `ClearWater-data` `main` @ `81689d4` unless noted.

- [ ] **B1 — No chunked Zarr *reader*.** `ChunkedDataSource.read_chunk` /
      `ChunkedDataProvider.read_chunk` are declared only as Protocol/ABC stubs
      (`clearwater_data/io/base.py:19`, `:78`) and never implemented in
      `io/zarr.py` (only `write_chunk` exists, `zarr.py:109`).
      `ZarrDataSource.read` eagerly `.compute()`s the **entire** variable
      (`zarr.py:15-21`). **Action:** implement a `ChunkedZarrDataSource` (or
      `ZarrDataSource.read_chunk`) returning a lazily-sliced
      `xr.open_zarr(...).sel(time=slice(...))` window without materializing the
      whole array.

- [ ] **B2 — Spatial-chunk bug.** `ChunkedZarrDataStore._init_zarr_store`
      uses `len(self.spatial_field_values)` for the spatial chunk extent
      (`zarr.py:92`); after normalization `spatial_field_values` is a *list of
      arrays*, so this counts spatial *fields* (≈1), not spatial *points* —
      inconsistent with the coordinate shape built at `zarr.py:57` (which
      correctly uses `len(value)` per field). Chunk shape is wrong unless there
      is exactly one cell. **Action:** use the per-field point count; add a
      test asserting chunk shape == `(chunk_length, n_cells)`.

- [ ] **B3 — Fixed uniform timestep only.** The store builder uses a single
      `pd.date_range(start, end, freq=time_step)` with one
      `time_step: timedelta` (`zarr.py:47-60`); `ChunkedZarrDataStore`
      computes `chunk_length = int(chunk_size / time_step)` (`zarr.py:90`),
      requiring `chunk_size` to be an integer multiple of `time_step`.
      Irregular HEC-RAS output cadence is not representable. **Action:** either
      support an explicit non-uniform `time` coordinate, or document and
      enforce the uniform-cadence precondition (with a clear error) and provide
      a resample-on-ingest path.

- [ ] **B4 — Exact-timestamp chunk-boundary fragility (riverine `main`).**
      In `ClearWater-riverine/src/clearwater_riverine/model.py`:
      `__increment_timestep` advances by `CHANGE_IN_TIME` and breaks when
      `dt` is a per-step array (`model.py:216`); chunk rollover is
      exact-timestamp equality `if self.__current_time in self.__chunk_ends`
      (`model.py:415`) against `pd.date_range(..., freq=chunk_size)[1:-1]`
      (`model.py:380-386`). A daily chunk over sub-daily RAS output that does
      not evenly divide the chunk can miss the boundary and never roll over.
      **Action:** make chunk-boundary detection tolerance-based
      (`>=` next boundary) and handle scalar vs. array `dt` in
      `__increment_timestep`; add a non-uniform-cadence regression test.

### Secondary follow-ups (not hard blockers)

- [ ] Zarr writes are always `consolidated=False`, `zarr_format=3` in
      `clearwater_data` vs. the fork's `zarr_format=2`,
      `consolidated=True` for the resume mesh — confirm reader compatibility
      and pick one convention for the resume substrate.
- [ ] `clearwater_data/io/__init__.py` is empty, so concrete classes have no
      short import path — add exports to stabilize the public API the fork and
      ESM will depend on.
- [ ] `clearwater_data` has no tests/examples for the chunked path — add
      round-trip write_chunk → read_chunk coverage as part of B1.

---

## 8. Migration path (suggested ordering)

1. Land B1–B4 in `clearwater_data` / `clearwater-riverine` with tests.
2. Swap the fork's `io/hdf.py` reader for `RASHDFDataSource`; keep the fork's
   `mesh_cache.py` static cache on top of it.
3. Swap the fork's hand-rolled Zarr-v2 flush for `ChunkedZarrDataStore`
   (Zarr v3); re-target `checkpoint()`/`from_checkpoint()` and the
   release logic onto the new store.
4. Verify the migrated fork against the existing mass-balance closure suite
   (`tests/test_final_mass.py` — note: a refactor-API rewrite of this suite is
   currently uncommitted on `steissberg-riverine-merged`).
5. ESM: wire `SharedDatasetView` + `TemporalAggregator` into
   `increment_timestep`; add the duration/exceedance reducer; validate against
   a Santiam-Salem smoke run.

For the file-by-file decisions backing steps 2–3 (what to port, re-base, skip,
or evaluate as the fork code is incrementally merged onto this `#135` base),
see **§10**.

---

## 9. Risks & open questions

- **Performance of per-window HDF re-read.** The blessed reader re-opens and
  re-slices the HDF every chunk. On long runs this trades the fork's
  resident-mesh memory cost for repeated I/O. Benchmark before committing;
  the fork's `mesh_cache.py` mitigates only static geometry, not the temporal
  re-read.
- **Resume substrate format.** Fork checkpoints use Zarr v2 +
  `npz` + JSON; the blessed store is Zarr v3. Decide whether the resume mesh
  also moves to v3 or stays a separate v2 artifact.
- **Cross-boundary flux continuity** is fork-specific and has no upstream
  counterpart — its correctness must be re-validated after re-basing on the
  blessed reader (the rolling-2-slot invariant must be preserved).
- The full multi-workstream reconciliation plan is being designed in a
  separate working session; referenced fork memos
  `design/refactor_strategy.md` and `design/albany_90day_orchestrator_refactor.md`
  were **not present** in the repos at analysis time and could not be
  cross-checked.

---

## 10. Per-component port decision table (fork → `#135` base)

Operationalizes §4 for the incremental merge of `ClearWater-Riverine-streaming`
(`dry-cell-fix` @ `df6650f`) onto this branch (`steissberg-riverine-merged`,
the PR #135 / `clearwater_data` base).

**Legend**

| Decision | Meaning |
|---|---|
| **SKIP** | Do not port. The `#135` base already provides the blessed equivalent; porting re-introduces the duplicate work. |
| **RE-BASE** | Keep the behavior/intent, reimplement on top of the `#135` reader/store (`RASHDFDataSource` / `ChunkedZarrDataStore` / `VariableRegistry`). |
| **PORT** | Net-new capability absent from the `#135` base; bring over largely as-is (may need a substrate retarget). |
| **EVALUATE** | Requires a direct diff against the `#135` equivalent before deciding; cannot be settled from analysis alone. |

> Fork file:line references are from `dry-cell-fix` @ `df6650f` as read during
> analysis; confirm against the working tree before editing.

| Fork component (file : symbol) | Role | Decision | Rationale / blocker | `#135` target |
|---|---|---|---|---|
| `io/hdf.py` : `HDFReader`, `_hdf_to_xarray` (`:112-130`), `_parse_dates` (`:170-211`), `define_hydrodynamics_timevarying` (`:391-490+`) | hand-rolled HDF→xarray window reader | **SKIP** | Core duplicated plumbing (§4.1). Caveat: main's `RASHDFDataSource.temporal_variables` is a fixed 4-entry dict — extra temporal vars (cell depth/velocity) require *extending* `RASHDFDataSource`, not porting the fork reader. | `clearwater_riverine.io.hdf.RASHDFDataSource` (`io/hdf.py:114`) + `.read_chunk` (`:348-379`) |
| `io/inputs.py` : loader registry | fork-only input loader registry | **EVALUATE** | Not in main. Decide whether it is subsumed by `clearwater_data` `DataSource` / `VariableRegistry`; port only the residual gap. | `clearwater_data.io.base` Protocols + `VariableRegistry` |
| `io/mesh_cache.py` : `read_cache`/`write_cache` (`:144-178`), key (`:65-132`) | static-geometry gzip-pickle disk cache | **PORT** | Net-new (§4.2); main re-reads topology every run. Cache key already version-stamped — re-validate key after re-base. | layered on `RASHDFDataSource.mesh` |
| `io/config.py`, `io/outputs.py` | config / output helpers | **EVALUATE** | Diff against main's config/output paths; expect partial re-base. | main `io/config.py`, `model.py` output store |
| `transport.py` : `__init__` streaming kwargs + state init (`:73-93`, `:380-399`) — `streaming_output`, `streaming_interval`, `release_after_flush`, `_stream_path`, `_released_time_range` | streaming config plumbing | **RE-BASE** | Keep the knobs; retarget storage to the blessed store. | `ChunkedZarrDataStore` ctor (`clearwater_data/io/zarr.py:81`) |
| `transport.py` : `update()` flush trigger (`:1004-1006`) | every-N-steps flush trigger | **RE-BASE** | Keep the trigger cadence; the action it fires changes. | `__finalize_chunk` analogue (`model.py:421-436`) |
| `transport.py` : `_flush_to_stream()` (`:1691`), Zarr-v2 `append_dim` (`:1738-1741`) | hand-rolled streaming output flush | **RE-BASE** | Duplicate of main's Zarr output. Gated by **B2** (chunk-shape bug) + **B3** (uniform-cadence assumption). | `ChunkedZarrDataStore.write_chunk` (`zarr.py:109`, Zarr v3, `region="auto"`) |
| `transport.py` : `_release_to_stream()` (`:1754-1828`), rolling-2-slot NaN release | bounded-memory in-place release | **PORT** | Net-new (§4.1); absent from main/`clearwater_data`. Preserve the rolling-2-slot solver-continuity invariant exactly. | n/a (new layer on the blessed store) |
| `transport.py` : `read_streamed()` (`:1866`), `available_time_range` (`:1852`) | lazy read-back of released slots | **RE-BASE — BLOCKED on B1** | Needs a chunked Zarr *reader*; `clearwater_data` has only `write_chunk`, no `read_chunk` impl. Cannot land until **B1**. | new `ChunkedZarrDataSource` (per **B1**) |
| `transport.py` : `checkpoint()` (`:1893`), `from_checkpoint()` (`:1981`) | process-per-chunk checkpoint/resume | **PORT** | Net-new (§4.2). `fluxes.npz` + `metadata.json` port as-is; mesh substrate (Zarr v2 consolidated) — decide v2 vs `#135` v3 (§7 secondary follow-up). | n/a (new layer) |
| `transport.py` : cross-boundary flux/mass continuity (`fluxes.npz` + rolling-2-slot) | mass continuity across chunk boundaries | **PORT** | Net-new; main carries constituent IC only (§4.2, §9). **Re-validate closure after re-base.** | n/a (new layer) |
| `transport.py` : `finalize()` (`:1598`) | finalize / save | **EVALUATE** | Reconcile with main's `finalize` / `__finalize_chunk` (`model.py:421-436`). | main `model.py` finalize path |
| `cli.py` : `--checkpoint-dir`, `--checkpoint-interval` (`:21-24`, loop `:66-73`) | CLI surface for checkpointing | **PORT** | Net-new CLI for the kept checkpoint feature; reconcile with main's CLI if present. | main `cli.py` |

**Out of scope for this branch.** The Layer-2 orchestrator (process-per-chunk
driver, `_orchestrator_helpers.py`, `08_run_coupled_*`, `DailyAggregator`)
lives in the **ESM-streaming repo**, not `ClearWater-riverine`. It is tracked
in the ESM-streaming side of the reconciliation, not here.

**Tally:** 1 SKIP (the duplicated reader), 4 RE-BASE (1 blocked on B1),
4 PORT, 3 EVALUATE. Net: the only thing deleted outright is the duplicated HDF
reader; the streaming/checkpoint/continuity layer is preserved and re-based.

---

## Appendix: method

Findings derived from direct source inspection of the five repos/branches in
§0 on 2026-05-18. All file:line references were read or grep-verified against
the stated commits. The `clearwater-riverine` `main` chunking files were
confirmed identical to `steissberg-riverine-merged`. No code was modified to
produce this analysis.
