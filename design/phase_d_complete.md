# Phase D: Numerical-Correctness Port — Close-Out

**Date:** 2026-05-20
**Branch:** `steissberg-riverine-merged`
**Head:** `e34ddff`
**Status:** Complete

## What Phase D delivered

Phase D ports the streaming fork's `dry-cell-fix` branch (`df6650f`,
EPA-validated against Santiam-Salem with Salem T bias −0.30 °C /
RMSE 0.62 °C) onto the canonical clearwater_data-based
`ClearWater-riverine`. Each unit lands as an atomic commit on
`steissberg-riverine-merged`; together they restore numerical
correctness on wet-dry edges without touching the legacy
no-`WET_MASK` code path.

The whole stack is opt-in via Unit A's `wet_dry_metric` kwarg
(default `None`). When `wet_dry_metric` is left at its default, every
gate in Phase D collapses to a no-op and the run is byte-identical
to the pre-Phase-D path; the conservation guard suite confirms this
at every unit. Setting `wet_dry_metric` to `"volume"`, `"depth"`, or
`"both"` opens the full stack.

## Unit-by-unit ledger

| Unit | Commit | What it does |
|---|---|---|
| **A** | `1b62a22` | Wet/dry mask scaffolding. Registers `WET_MASK` on the current chunk window when `wet_dry_metric` is set. Re-registered per chunk in chunked mode. |
| **B** | `264549d` | Newly-wet reconstruction. `reconstruct_newly_wet` lifts the `c≈0` artifact in cells that go from dry at t to wet at t+1 (HEC-RAS staggered-time inflow pattern). Two-pass for wetting fronts; "only lift, never lower" preserves solver-positive values. |
| **C-α** | `2a1c8e1` | LHS wet-dry edge filter. Adds rule-1 (dry-cell identity pin so the row is non-singular) and rule-3 (donor-diagonal contribution on wet-dry edges so the implicit solve sinks the wet cell's outflow mass). Emits `wet_dry_leak_donors` / `wet_dry_leak_abs_adv` / `dry_cells_t1` for downstream consumers. |
| **C-β** | `675998e` | Pre-solve drain + post-solve leak accumulator. `drain_newly_dry` apportions a dying cell's `V·c` to wet face-neighbours via `f·c_donor`; the unaccounted residual lands in `TransportEngine.mass_lost_to_dry`. The C-α leak diagnostic is integrated post-solve into the same accumulator. |
| **C-γ** | `15d1a34` | IC zeroing + BC inflow accumulator + end-of-run warning. `zero_dry_initial_conditions` returns the per-constituent IC mass loaded into sub-threshold cells; `RHS.bc_inflow_mass` accumulates per-step ghost-cell inflow mass; `emit_mass_loss_warning` compares the per-constituent total against `mass_loss_warn_threshold × BC inflow`. All three helpers are free functions so the caller decides when to invoke. |
| **D1** | `7f61582` | Per-constituent `is_intensive` flag + LHS suppression + engine cache. Intensive scalars (e.g. water temperature) tighten the donor gate from `ef1_wet_or_ghost` to `edge_active` so wet-dry edges no longer contribute to the donor's diagonal — restores pre-rule-3 behaviour for the temperature case, where pulling "heat" out of the wet cell toward a dry neighbour is non-physical. `TransportEngine` builds a second LHS / matrix lazily when at least one constituent is flagged intensive. |
| **D2** | `7aa43fc` | Model-level opt-in. `ClearwaterRiverine.__init__` invokes `_zero_dry_initial_conditions_fn` (aliased to dodge the kwarg name collision) after engine construction when both `zero_dry_initial_conditions=True` and `WET_MASK` is in the registry. `finalize` calls `emit_mass_loss_warning`. New `transport_engine` property exposes the engine for inspection. |
| **E** | `e34ddff` | Integration tests + ghost-edge flux contract. Five end-to-end tests stack Units A–D2 together on plan02 + plan08; one locks the contract that canonical's `_calculate_mass_flux` has zero NaN at boundary edges (the fork's Step-4 `_mass_flux` ghost-patch is unnecessary on canonical because `Constituent.set_boundary_conditions` writes BC values directly into `registry.get(name)`'s ghost slots). |

## Final test tally

Full suite at `e34ddff` (run `bcu20mv4o`, 42:53):

```
94 passed, 10 skipped, 3 errors in 42:53 (0:42:53)
```

- **94 passed** = 49 baseline + 45 new Phase-D tests across eight files
- **10 skipped** = `plan03` / `plan11` gated fixtures (storm-surge / slow)
- **3 errors** = pre-existing `test_riverine.py::{test_datetime_range, test_riverine_initialize, test_riverine_update}` orphans from PR #135's constructor-signature change (canonical-shipped finding #2; not Phase-D regressions)

Per-unit pre-push validations (all green):

| Unit | Targeted | Non-chunked guard |
|---|---|---|
| C-α | 5/5 (1.86 s) | 7/2 (full suite `b1ijkd8wv`: 54/10) |
| C-β | 7/7 (1.90 s) | 7/2 (5:31) |
| C-γ | 14/14 (2.44 s) | 7/2 (7:03) |
| D1 | 8/8 (2.42 s) | 7/2 (6:31) |
| D2 | 6/6 (1.26 s) | 7/2 (4:18) |
| E | 5/5 (79.25 s) | (rolled into final full suite) |

## Branch state

- **`ClearWater-riverine`**
  `origin/steissberg-riverine-merged @ e34ddff`
  `origin/main` untouched at `9bd4470` (PR #135 base)
- **`ClearWater-data`**
  `origin/steissberg-clearwater-data-chunked-reader @ e328900`
  `origin/main` untouched at `81689d4`
  Unchanged since Phase-C C3b (init_template flag for checkpoint/resume).

## Opt-in semantics (quick reference)

```yaml
# legacy / no-Phase-D path (default; bit-identical to pre-Unit-A):
model:
  # no wet_dry_metric set
  # no zero_dry_initial_conditions set
  # mass_loss_warn_threshold ignored (no losses to warn about)

# full Phase-D opt-in (e.g. for a model with wet-dry dynamics):
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
```

## Diagnostic surface

After a run, callers can inspect (no API churn; all already exposed):

```python
engine = model.transport_engine                 # D2: property
engine.mass_lost_to_dry                         # dict[name, list[float]] per step
engine.lhs.wet_dry_leak_donors                  # np.int64 array (last step)
engine.lhs.wet_dry_leak_abs_adv                 # np.float array (last step)
engine.lhs.dry_cells_t1                         # np.int64 array (last step)
model._constituents["tracer"].rhs.bc_inflow_mass  # list[float] per step
```

## What is explicitly NOT in scope here

- Real-corridor scale validation (e.g. 92-day Willamette workstation
  run) — separate Phase-D deliverable, not yet executed on canonical.
- Merge of `steissberg-riverine-merged` → `streaming` → `main` —
  outside this branch's scope; coordinated with the Riverine
  maintainers separately.
- Fork's gated-fixture work (`test_riverine.py` orphan repair) —
  pre-existing canonical defect, tracked under canonical-shipped
  finding #2.
- Streaming fork's intensive temperature production validation
  (Santiam-Salem EPA, 2026-05-18) does not re-run here; the D1 port
  matches the fork's mathematical contract under the C-α tests.
