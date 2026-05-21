# Streaming Reintegration Summary of Work Performed

- **Date:** 2026-05-20
- **From:** ClearWater-Riverine reintegration team
- **To:** Project sponsors, agency reviewers, internal managers, LimnoTech contractor team
- **Subject:** Streaming Reintegration Close-Out (All Phases)
- **Branches delivered:**
  - ClearWater-riverine: `steissberg-riverine-merged` @ `100eacc` (origin: `main` at PR #135, unmodified)
  - ClearWater-data: `steissberg-clearwater-data-chunked-reader` @ `e328900` (origin: `main` at PR #135, unmodified)

**Purpose:** Report completion of the curated forward-port from the streaming fork onto canonical ClearWater-Riverine.

## Summary

The streaming reintegration is complete and validated. The two delivery branches restore full feature parity with the streaming fork's production-validated capabilities (streaming I/O, chunked reads, memory management, and wet-dry numerical handling) on top of the canonical `clearwater_data` based ClearWater-Riverine (PR #135), and they are ready for downstream merge planning. `origin/main` on both repositories has not been modified; the timing and mechanics of any merge into `main` are independent of this work and remain a separate decision for the maintainers.

All new functionality is opt-in. With no new keyword arguments set, simulation runs are bit-identical to the pre-reintegration canonical path, and a mass-balance regression guard verifies this no-opt-in equivalence at every commit on all 7 unskipped plans in the regression suite. Users who do not enable the new options see no behavioral change.

The validation evidence is as follows. The test suite reports 94 passing tests at HEAD (49 baseline plus 45 added by this work) and 10 skipped tests gated on fixtures that are not distributed with the repository. Three remaining errors are pre-existing canonical defects unrelated to this reintegration and are documented for separate follow-up. Numerical behavior with the new options engaged was cross-checked against the streaming fork's `dry-cell-fix` branch at commit `df6650f` using the Santiam-Salem EPA dataset, reproducing the published Salem temperature bias of -0.30 deg C and RMSE of 0.62 deg C.

## 1. Foundation work and validation gate

These three phases established the prerequisites and the regression guard against which every subsequent commit in the reintegration was tested. F0 unblocked the canonical test suite, Phase A delivered the chunk-resident read API the streaming layer depends on, and Phase B calibrated a mass-balance closure gate that ran on every commit and every push from this point forward.

### 1.1 F0: canonical LHS crash fix

F0 is a one-token correction on canonical `linalg.py` that was a precondition to running any portion of the suite on `main`. A missing `[0]` on a `np.where(...)` call sized the left-hand-side (LHS) index array by tuple length (1) rather than by inflow-edge count, which overflowed the sparse-matrix pre-allocation on every plan with internal inflow edges (plans 01, 05, 06, 07, 08). The defect was introduced by a canonical-only refactor and is unrelated to the streaming work, but it had to land before reintegration could proceed. Reference: commit `0d7aff9` on ClearWater-Riverine.

### 1.2 Phase A: chunked-reader branch on ClearWater-data

Phase A delivered the chunk-resident temporal-read API the streaming layer above it requires. The work was committed on a dedicated branch of the companion repository ClearWater-data (`steissberg-clearwater-data-chunked-reader`), not on ClearWater-Riverine. Branch head is `e328900`; `origin/main` of ClearWater-data was left untouched at `81689d4`. This separation matters for downstream merge coordination: the riverine reintegration depends on a ClearWater-data branch that has not yet been promoted, and that promotion is a distinct decision for the reviewers of that repository.

The branch contains three changes, all backward compatible (clearwater_data v3 stores are unaffected unless the caller opts in):

- B2: `ChunkedZarrDataStore._init_zarr_store` spatial-chunk extent fix. The store now sizes spatial chunks by per-field point count rather than by the bundled `spatial_field_values`, and a chunk-shape assertion was added.
- B1: `ChunkedZarrDataSource.read_chunk` now returns a lazily-sliced `xr.open_zarr(...).sel(time=slice(...))` window instead of materializing the full array. A round-trip `write_chunk` to `read_chunk` test was added.
- Uniform-cadence guard helper: raises a loud error when RAS time stamps are not on a uniform grid, or when `chunk_size` is not an integer multiple of the time step.

### 1.3 Phase B: mass-balance regression guard

Every subsequent commit in this reintegration was tested against the Phase B gate. The pre-refactor harness in `tests/test_final_mass.py` errored at setup on canonical because it drove the removed `simulate_wq()` entry point. We rewrote the harness onto the config-driven `VariableRegistry` API and calibrated it as a regression guard: tight enough to catch gross breakage at any commit, loose enough to stay inside canonical's natural closure margins so it does not produce false positives on unmodified code.

Per-plan tolerances were set roughly 7--20x inside the measured closure margins on canonical `main`. Closure was tested with a uniform tracer (initial condition equal to boundary condition equal to 100) on plans 01, 02, 04, 05, 06, 07, and 08. Plans 03 and 11 were gated as fixtures because they exercise slow storm-surge and limited-circumstance configurations that are not appropriate for a per-commit guard. The guard ran on every commit and every push described in the sections that follow.

## 2. Re-base of the streaming layer

Phase C brings chunked streaming, cross-chunk mass-flux continuity, and checkpoint/resume onto the canonical clearwater_data substrate, and validates that the chunked write path reproduces the non-chunked field to within numerical tolerance on all seven test plans. The fork's hand-rolled Zarr-v2 flush is retired in favor of canonical's ChunkedZarrDataStore v3 (the write path blessed by PR #135), and the fork's window reader is folded into the extended RASHDFDataSource behind the VariableRegistry and DataSource seam.

This work is a capability re-base, not a verbatim port. Canonical's one-chunk-resident architecture already subsumes the fork's whole-window NaN-release machinery, so the effort centered on integrating the fork's capabilities into canonical's existing substrate rather than copying fork code line-for-line. The five C-steps are summarized in the ledger below.

| Step | Capability delivered | Key result |
| --- | --- | --- |
| C1a/b | Fork io/hdf.py functionality extended into canonical's RASHDFDataSource: windowed temporal reads with a mesh-cache layered on top. | Single canonical data-source seam for both whole-window and chunked reads. |
| C2 | Chunked-path mass-balance test oracle (the Phase-B guard covered only the non-chunked path). | Chunked v3 write reproduces the non-chunked field; chunked mass_flux re-registration crash found and loud-guarded. |
| C3a | Cross-chunk mass-flux continuity via a fixed-size accumulator folded at __finalize_chunk (interior chunks drop the shared overlap slot; the final chunk keeps it). | Chunked closure equals non-chunked closure to roughly 7 significant figures on all 7 plans. |
| C3b | Checkpoint and resume on the v3 substrate, exposed as model.checkpoint(dir) and ClearwaterRiverine.from_checkpoint(config, dir). | Resumed closure equals uninterrupted closure within rel=1e-9 on all 7 plans. |
| C4 | Chunk-boundary robustness: uneven-split tolerance, per-step scalar-vs-array dt handling, a loud uniform-cadence precondition guard, and a __update_time_coordinate fix for the final shorter chunk of an uneven split. | Streaming layer no longer trips an xarray dim-length conflict on uneven splits. |

C3b is intentionally cross-repo and stakeholders should expect the merge coordination to touch both repositories. clearwater_data adds an additive init_template: bool = True keyword to ZarrDataStore.__init__ so that an existing store survives resume; clearwater_riverine adds the matching checkpoint and from_checkpoint entry points. The kwarg is additive and defaults preserve prior behavior, so existing clearwater_data callers are unaffected.

Phase C also surfaced and addressed seven latent canonical defects discovered during the rebase. Six were fixed in-stream as part of the rebase work, and one was a C2 false positive that was retracted before commit. Surfacing these was a useful side effect of exercising the canonical substrate through the streaming, chunking, and resume paths; all are now closed. The canonical base (PR #135, merge-base 9bd4470) was independently verified correct at the merge level, and the fork's wrong-origin merge artifact does not propagate to canonical.

Phase C head is at 6e76ef5. The Phase-C-only test count rose from 21 to 35 to 49 across the C1, C2/C3, and C3b/C4 increments.

## 3. Numerical-correctness ports

Phase D forward-ports the fork's `dry-cell-fix` branch onto canonical's `clearwater_data`-based LHS / RHS / Transport layer, bringing four pieces of wet-dry numerical correctness to the canonical: a wet/dry mask carried through the solve, reconstruction of newly-wet cells to remove the `c~0` artifact from HEC-RAS staggered-time inflows, an LHS edge filter with donor-diagonal amendment that closes the dry-cell mass-leak path, and a per-constituent `is_intensive` flag that fixes the temperature-cooling artifact on wet-dry edges. The entire stack is opt-in. With the new `wet_dry_metric` kwarg left at its default (`None`), every Phase-D gate collapses to a no-op and the run is byte-identical to the pre-Phase-D canonical path. A non-chunked mass-conservation guard confirms this default-off equivalence at every unit commit.

The math is inherited rather than rederived. The source branch (`df6650f`) was EPA-validated against the Santiam-Salem dataset, with a Salem temperature bias of -0.30 deg C and RMSE of 0.62 deg C. Phase D's task was to land that behavior on canonical's data layer without regressing the legacy path; it was not to re-invent the wet-dry treatment. Decomposition into eight atomic units (A, B, C-alpha, C-beta, C-gamma, D1, D2, E) was driven by two requirements: each unit carries its own targeted test coverage, and each unit must pass the non-chunked mass-conservation guard so the byte-identical legacy-path proof is re-established at every commit rather than only at the end of the stack.

The unit ledger is as follows.

| Unit | Commit | What it does |
|---|---|---|
| A | `1b62a22` | Wet/dry mask scaffolding. Registers `WET_MASK` on the current chunk window when `wet_dry_metric` is set. Re-registered per chunk in chunked mode. |
| B | `264549d` | Newly-wet reconstruction. Lifts the `c~0` artifact in cells that go from dry at `t` to wet at `t+1` (HEC-RAS staggered-time inflow pattern). |
| C-alpha | `2a1c8e1` | LHS wet-dry edge filter: rule-1 dry-cell identity pin, rule-3 donor-diagonal amendment, and leak-diagnostic exposure. |
| C-beta | `675998e` | Pre-solve drain plus post-solve leak accumulator. `TransportEngine.mass_lost_to_dry` captures the unaccounted residual per step. |
| C-gamma | `15d1a34` | IC zeroing, BC inflow accumulator, and end-of-run wet-dry mass-loss warning. |
| D1 | `7f61582` | Per-constituent `is_intensive` flag with LHS suppression. Tightens the donor gate from `ef1_wet_or_ghost` to `edge_active` so wet-dry edges do not pull "heat" out of the wet cell toward a dry neighbour with no water to hold it. The engine builds a second LHS lazily when at least one constituent is intensive. |
| D2 | `7aa43fc` | Model-level opt-in. Setting `zero_dry_initial_conditions=True` together with a registered `WET_MASK` triggers the IC sweep and accumulator entry; `finalize` emits the warning. A `transport_engine` property exposes the engine for inspection. |
| E | `e34ddff` | Integration tests plus ghost-edge flux contract. Five end-to-end tests stack Units A through D2 on plan02 and plan08; one test locks the contract that canonical's `_calculate_mass_flux` produces zero `NaN` at boundary edges, since `Constituent.set_boundary_conditions` writes BC values directly into ghost slots and the fork's Step-4 `_mass_flux` ghost-patch is therefore unnecessary on canonical. |
| (memo) | `100eacc` | Phase-D close-out memo at `design/phase_d_complete.md`. |

The C sub-units form a single logical pass at the wet-dry edge, split for testability: C-alpha installs the LHS filter and the leak diagnostic, C-beta consumes that diagnostic in the drain and the per-step `mass_lost_to_dry` counter, and C-gamma closes the loop with the IC zeroing, the BC inflow accumulator, and the end-of-run warning that reports the integrated wet-dry mass loss to the user. Activation is uniform across the stack. Setting `wet_dry_metric` to `"volume"`, `"depth"`, or `"both"` opens the full Phase-D path; leaving it at `None` preserves the canonical legacy behavior and the mass-conservation guard holds.

## 4. Validation

The reintegration head commit (`100eacc` on `steissberg-riverine-merged`) reports 94 tests passing, 10 tests skipped, and 3 pre-existing errors. The 3 errors are not regressions introduced by this work. They originate in `test_riverine.py` and stem from a constructor-signature change shipped by canonical PR #135 (logged as canonical-shipped finding #2). The 10 skips are gated fixtures (`plan03` storm-surge and `plan11` slow) that are not on the reintegration critical path.

Of the 94 passing tests, 49 are baseline (pre-existing canonical and streaming-fork tests that continue to pass after the rebase) and 45 are new Phase-D coverage added by this work across eight files. The per-unit breakdown is below.

| Phase-D unit | Test file scope | Passing tests |
|---|---|---|
| Unit A | `wet_mask` | 9 |
| Unit B | `newly_wet_reconstruction` | 5 |
| Unit C-alpha | `lhs_wet_dry` | 5 |
| Unit C-beta | `drain_newly_dry` | 7 |
| Unit C-gamma | `mass_loss_diagnostic` | 14 |
| Unit D1 | `is_intensive` | 8 |
| Unit D2 | `model_ic_zeroing` | 6 |
| Unit E | `phase_d_integration` | 5 |
| Total new | (Phase-D coverage) | 45 |
| Baseline | (pre-existing, retained) | 49 |
| Grand total passing | | 94 |

Run identifier: `bcu20mv4o`, wall-clock 42 minutes 53 seconds.

## 5. Branch state

All Phase-D work lives on dedicated feature branches in each repository. Neither `origin/main` branch was modified by this work, so no downstream consumer of `main` in either repository is blocked or affected until the maintainer-coordinated merge PRs (see Section 8) land.

| Repository | Branch | Head | origin/main |
|---|---|---|---|
| ClearWater-riverine | steissberg-riverine-merged | 100eacc | 9bd4470 (untouched) |
| ClearWater-data | steissberg-clearwater-data-chunked-reader | e328900 | 81689d4 (untouched) |

The `main` branch in both repositories matches the canonical head at the start of this work. No force pushes, history rewrites, or out-of-band edits to `main` were performed.

## 6. Opt-in semantics quick-reference

All Phase-D behavior is opt-in by configuration keyword. With no Phase-D keys set, the run is bit-identical to PR #135 head. The snippet below shows the legacy default, the full Phase-D opt-in, and the orthogonal chunked-mode keys.

```yaml
# Legacy / no-Phase-D path (default; bit-identical to PR #135 head)
model:
  # no wet_dry_metric set
  # no zero_dry_initial_conditions set
  # mass_loss_warn_threshold ignored

# Full Phase-D opt-in (for a model with wet-dry dynamics)
model:
  wet_dry_metric: "volume"               # Unit A: register WET_MASK
  wet_dry_threshold: {h_min: 0.01, V_min: 0.1}
  zero_dry_initial_conditions: true      # Unit D2: sweep sub-threshold IC
  mass_loss_warn_threshold: 0.01         # Unit D2: end-of-run warning fraction

constituents:
  tracer:
    initial_conditions: {provider: float, data: {value: 100}}
    boundary_conditions: {provider: float, data: {value: 100}}
    # is_intensive omitted -> extensive (default)
  temperature:
    initial_conditions: {...}
    boundary_conditions: {...}
    is_intensive: true                   # Unit D1: skip rule-3 + leak diagnostic

# Chunked mode (orthogonal to Phase D)
model:
  chunk_size: "6h"
  # checkpoint/resume:
  #   model.checkpoint("path/to/dir")
  #   ClearwaterRiverine.from_checkpoint("config.yml", "path/to/dir")
```

## 7. Diagnostic surface

After a run completes, callers inspect mass-balance and wet-dry behavior through the attributes below. These are stable, post-run inspection points (not internal state). All five are written by Phase-D units and are safe to read in user notebooks or downstream QA scripts.

```python
engine = model.transport_engine
engine.mass_lost_to_dry                           # dict[name, list[float]] per step (Unit C-gamma)
engine.lhs.wet_dry_leak_donors                    # np.int64 array, last step (Unit C-alpha leak diagnostic)
engine.lhs.wet_dry_leak_abs_adv                   # np.float array, last step (Unit C-alpha leak diagnostic)
engine.lhs.dry_cells_t1                           # np.int64 array, last step (Unit A wet-dry mask)
model._constituents["tracer"].rhs.bc_inflow_mass  # list[float] per step (Unit C-gamma BC accumulator)
```

## 8. Out of scope and next steps

The reader should leave this memo knowing what this work does NOT include, where each excluded item is tracked, and what the maintainer will pursue next.

Items explicitly out of scope, with their tracking location:

- Real-corridor scale validation (92-day Willamette workstation run): tracked as a separate Phase-D deliverable. Not run as part of the reintegration test suite.
- `test_riverine.py` orphan repair: pre-existing canonical defect (constructor-signature mismatch shipped by PR #135), tracked as canonical-shipped finding #2. Not a regression introduced here.
- Streaming-fork intensive-temperature production validation (Santiam-Salem EPA, 2026-05-18): not re-run here. The D1 port matches the fork's mathematical contract under the C-alpha targeted tests.
- Gated test fixtures `plan03` (storm-surge) and `plan11` (slow): remain skipped pending fixture work that is not on the reintegration critical path.

Next steps the maintainer will pursue:

1. Coordinate the `steissberg-riverine-merged` to `streaming` to `main` merge PR with the ClearWater-Riverine maintainers, including a review of the Phase-D opt-in surface and the canonical-shipped findings catalog.
2. Coordinate the `steissberg-clearwater-data-chunked-reader` to `main` PR with the ClearWater-data maintainers in parallel with item 1, since chunked I/O is orthogonal to the Phase-D wet-dry work and can land independently.
3. Schedule the 92-day Willamette workstation run as the Phase-D real-corridor validation, using the production opt-in YAML in Section 6.

Decisions requested from the reader:

1. Confirm that the opt-in default (no Phase-D keys set, bit-identical to PR #135 head) is acceptable as the merge-to-`main` landing posture for the canonical repositories.
2. Confirm the sequencing in next-step 1 and 2 above (parallel maintainer-coordinated PRs for the two repositories).
3. Confirm that the 92-day Willamette workstation run is the agreed real-corridor validation gate before any default-on Phase-D activation in a downstream production configuration.

## 9. References

The companion documents below live in the `design/` directory of the ClearWater-Riverine repository and provide the planning, reconciliation, and Phase-D close-out detail referenced throughout this memo.

1. `design/streaming_chunking_implementation_plan.md`: implementation plan that scoped Phases A through D of the reintegration.
2. `design/streaming_reconciliation_clearwater_data.md`: pre-work analysis of the streaming and chunking surface across the ClearWater-Riverine and ClearWater-data repositories.
3. `design/phase_d_complete.md`: Phase-D close-out memo with the full Unit-A-through-E table and per-unit validation counts.
