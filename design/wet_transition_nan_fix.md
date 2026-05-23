# Wet-transition NaN propagation: diagnosis and fix design

**Status:** Proposed (2026-05-22)
**Branch target:** `steissberg-riverine-merged`
**Author:** investigation from Santiam-Salem dye-tracer regression

## TL;DR

Cells that wet during a run end up with `c[t+1] = NaN` and never recover,
even when a continuous point source is actively pumping mass into the
cell after it wets. The mass that the wet cell should receive from
advection of upstream wet cells (and/or from its own point source) is
silently lost. In the Santiam-Salem dye tracer with 3 point sources, the
entire wetted domain ends up with `dye = 0` at `t_end` because the wetting
front carries NaN through every cell it reaches.

The NaN originates not in the LHS matrix, not in `spsolve`, and not in
the load or point-source RHS terms. It originates in the outflow
ghost-cell contribution: `_ghost_cell(flowing_in=False)` reads ghost
constituent values at `current_time + time_step` from the registry, and
those ghosts are `NaN` whenever a user's BC CSV does not define a value
for that BC line. The NaN then flows: ghost concentration multiplier →
`diffusion_face × NaN = NaN` → RHS at the interior cell adjacent to the
outflow ghost → `spsolve` propagates NaN through the sparse system to
every reachable cell.

A secondary bug: `reconstruct_newly_wet` cannot lift NaN cells because
its "only lift, never lower" guard is written as `candidate > x_arr[i]`,
which is `False` when `x_arr[i]` is NaN. Even if the primary RHS NaN
were fixed, NaN cells would not heal through the Phase-D Unit B
mechanism.

## Symptoms (observed)

Santiam-Salem dye tracer, 15-day window 2008-09-01 → 2008-09-16, 3
point sources at cells 125016, 148424, 89621, continuous injection at
0.1 m³/s and 20000 mg/L per source.

Per-cell trace at cell 125016 (Upstream point-source cell):

| t   | V (m³)   | WET_MASK | dye (mg/L)  |
|-----|---------:|:--------:|------------:|
| 0   | 4.77e-5  | False    | 20000 (IC)  |
| 1   | 4.77e-5  | False    | 2.6e-4      |
| 5   | 4.77e-5  | False    | 8.2e-36     |
| 12  | 4.77e-5  | False    | 5.8e-91     |
| 18  | 4.77e-5  | False    | 3.2e-138    |
| **19** | **107.9** | **True**  | **NaN**    |
| 20+ | 108–146  | True     | NaN forever |

The cell wets between steps 18 and 19 (V jumps 5e-5 → 108 m³). At the
wet-transition step, the solve produces NaN. Once NaN, every subsequent
step has `c[t]` NaN in the cell, the LHS rule-1 pin doesn't fire (cell
is wet at t+1), and `reconstruct_newly_wet` won't lift NaN. NaN sticks
forever.

## Mechanism (confirmed by instrumentation)

Wrapping `scipy.sparse.linalg.spsolve` at the wet-transition step shows:

```
[spsolve call #19] cell 125016:
  A[125016, :]: 3 nonzero entries, all finite
    A[125016, 124633] = -0.0841
    A[125016, 125015] = -0.0668
    A[125016, 125016] = +0.1509
  RHS[125016] = 4.24e-146 (finite)
  Globally: NaN in A.data = 0 (of 175822)
            NaN in RHS    = 189
            NaN in x      = 4783
```

So:

1. `A` is clean (no NaN entries).
2. `RHS` contains 189 NaN entries at specific interior cells.
3. `spsolve` propagates those 189 RHS NaNs through the sparse coupling
   to 4783 cells in the solution vector.
4. Cell 125016 itself has a finite RHS but is downstream of NaN cells
   in the sparse coupling, so its `x[125016]` comes out NaN.

Component-wise RHS audit at the same step (also instrumented):

```
load:            n_NaN=0
ghost_cells_in:  n_NaN=0
ghost_cells_out: n_NaN=189   <-- source
point_sources:   n_NaN=0
RHS (sum):       n_NaN=189
```

The 189 NaN cells correspond to interior real cells with an edge to an
outflow ghost cell whose dye value at `t+1` is NaN in the registry.

## Root cause: `_ghost_cell(flowing_in=False)`

`src/clearwater_riverine/linalg.py:824-828`:

```python
concentration_multipliers = np.zeros(registry.get_variable(NFACE).get_data())
concentration_multipliers[internal_cell_index] = registry.get_at_time(
    constituent_name,
    current_time + time_step
)[external_cell_index]
```

`concentration_multipliers` is initialized to zero (the safe default for
unset ghosts), but the very next line overwrites entries at
`internal_cell_index` with whatever the registry holds for the matching
`external_cell_index`. If a ghost cell is along a BC line that the user's
BC CSV does not enumerate (e.g., a downstream outflow BC), the registry
holds NaN there at every timestep. The NaN replaces the safe 0 and then
multiplies `diffusion_face`, contaminating the outflow ghost contribution
to that interior cell's RHS.

This is asymmetric with the inflow branch, which reads the same ghost
values but at edges where the BC CSV is required to provide a value
(inflow boundaries are user-facing; users naturally set BC values
there). The outflow path quietly assumes the registry will be filled,
but the canonical BC reader leaves unspecified ghosts as NaN.

The fork API hid this because its ghost initialization filled all ghosts
to 0 (or some other default) before any BC overlay. Canonical's strict
"only fill what the user said to fill" stance was the right call, but
its ghost-cell consumer paths (this one in particular) did not get
updated to expect NaN.

## Secondary bug: `reconstruct_newly_wet` cannot lift NaN

`src/clearwater_riverine/transport.py:192-195`:

```python
if candidate is not None and candidate > x_arr[i]:
    x_arr[i] = candidate
    reconstructed[i] = True
    gather_conc[i] = candidate
```

The "only lift, never lower" rule was correct for preserving a positive
solution the implicit solve already computed. It is *incorrect* against
NaN: `candidate > NaN` is `False` for any candidate, so even with a
valid upstream-flow-weighted candidate (e.g., 100 mg/L from a wet
upstream donor), the NaN cell is never lifted.

Once a cell is NaN, all subsequent steps:

- have `c[t]` NaN (registry stores NaN forward through
  `set_at_time(constituent, t+1, x_full)` line 762-767),
- compute `RHS_load = V[t] * fillna(0) / dt` so the load is fine,
- but never get `reconstruct_newly_wet` to overwrite NaN with a real
  value because of the `>` comparison.

## Proposed fix

Three patches, ranked by importance. Patches 1 and 2 are required;
Patch 3 is defensive.

### Patch 1 (required, root cause): sanitize outflow ghost concentrations

File: `src/clearwater_riverine/linalg.py`, function `_ghost_cell`,
around line 824-828.

```python
ghost_concs = np.asarray(
    registry.get_at_time(constituent_name, current_time + time_step)
)[external_cell_index]
# Outflow / unspecified ghosts default to 0: a zero-gradient ghost for
# diffusion-only contributions on outflow edges, and a zero
# concentration multiplier on inflow edges where the user did not
# define a BC value (the prior behaviour of the fork API). Without
# this, NaN at any unspecified ghost propagates through diffusion
# into the interior cell's RHS, then through spsolve to every
# reachable cell, silently nuking the entire downstream constituent
# field.
ghost_concs = np.where(np.isfinite(ghost_concs), ghost_concs, 0.0)
concentration_multipliers[internal_cell_index] = ghost_concs
```

Rationale: treating an unspecified ghost as zero is the
forward-compatible interpretation. Users who actually want a non-zero
outflow BC continue to specify it in the BC CSV; users who only specify
inflow BCs get the natural zero-gradient outflow behavior. This matches
fork semantics for the common case and removes the NaN-injection path
entirely.

Alternative considered: fix this at the BC reader by forward-filling
ghost values to 0 at registration time. Rejected because (a) it makes
the registry-level invariant "ghosts are always finite for every
registered constituent", which is stronger than necessary and risks
masking real BC-definition bugs in user configs; (b) `_ghost_cell` is
already the natural consumer-side filter point; (c) the change is
contained to two lines.

### Patch 2 (required, defensive): make `reconstruct_newly_wet` NaN-aware

File: `src/clearwater_riverine/transport.py`, function
`reconstruct_newly_wet`, around line 192.

```python
# "Only lift, never lower" rule -- BUT treat NaN as -inf so a
# qualifying candidate always replaces a NaN entry. The pre-fix
# comparison `candidate > x_arr[i]` evaluates to False when
# `x_arr[i]` is NaN, leaving the cell stuck at NaN forever.
should_lift = candidate is not None and (
    not np.isfinite(x_arr[i]) or candidate > x_arr[i]
)
if should_lift:
    x_arr[i] = candidate
    reconstructed[i] = True
    gather_conc[i] = candidate
```

Even after Patch 1 removes the primary NaN source, downstream code or
future BC sources may legitimately produce NaN in newly-wet cells (e.g.,
sub-threshold-but-finite volume cells where the load coefficient
underflows). Phase-D Unit B's whole purpose is to lift such cases.
Treating NaN as the floor (which it semantically is for "we have no
information about this cell yet") restores that purpose without
weakening the "only lift, never lower" rule for finite values.

### Patch 3 (defensive): warn loudly on RHS NaN

File: `src/clearwater_riverine/linalg.py`, in `RHS.update_values`
right after `self.values[:] = self.__calculate_rhs(...)`.

```python
n_nan = int(np.isnan(self.values).sum())
if n_nan > 0:
    import warnings as _w
    _w.warn(
        f"RHS assembly produced {n_nan} NaN values for constituent "
        f"{constituent_name!r} at {current_time}. The implicit solve "
        "will propagate NaN to every cell coupled to these via the "
        "sparse matrix. Component breakdown: "
        f"load={int(np.isnan(self._calculate_load(registry, current_time, time_step)).sum())}, "
        f"ghost_in/out: trace via _ghost_cell. "
        "Likely cause: unspecified ghost BC values (see "
        "design/wet_transition_nan_fix.md).",
        UserWarning,
        stacklevel=2,
    )
```

The existing LHS NaN warning (around line 515) is a useful precedent.
The RHS path has the same blast radius — one NaN cell silently nukes
the whole downstream field — and currently has no diagnostic at all.

## Regression test design

Add a test in `tests/test_wet_transition_point_source.py` that:

1. Builds a minimal HEC-RAS HDF with a known wetting-front geometry: a
   thalweg cell that wets at `t=t_wet`, and a downstream cell that wets
   at `t=t_wet + dt`.
2. Configures a single constituent `dye` with:
   - IC = 0 everywhere
   - BC CSV with **only** an Upstream BC (no downstream BC entry,
     reproducing the user-facing config that triggers the bug)
   - point_sources CSV with a continuous source at the thalweg cell
     (`Flow_Rate=0.1, Concentration=100` from `t=0` to `t=end`)
3. Runs canonical with `wet_dry_metric="both"`, `reconstruct_newly_wet=True`.
4. Asserts at `t=t_wet + N` for `N in [1, 5, 10]`:
   - `dye[thalweg_cell]` is finite and `> 0` (point source delivered mass)
   - `dye[downstream_cell]` is finite at `t=t_wet + dt + N`,
     monotonically rising from 0 toward the upstream value (advection
     transported mass downstream)
5. Asserts mass balance: `sum_cells(V * dye) ≈ flow * conc * dt *
   (t_end - t_wet)` within a few percent.

The pre-Patch-1 behavior fails all three assertions. The post-fix
behavior must pass them.

## Migration / backward-compat notes

- **Patch 1 default behavior changes for users who relied on canonical
  NaN-propagation as a "loud failure" signal.** No such users exist in
  practice; the failure is silent (a downstream constituent field of
  zeros looks identical to no transport), so this is a safe default.
- **Patch 2 default behavior changes for newly-wet cells with NaN
  in `x_full`.** Pre-fix, those cells stayed at NaN. Post-fix, they
  are lifted by `reconstruct_newly_wet` whenever a wet upstream
  neighbor or ghost provides a finite candidate. This is the
  documented intent of Unit B; pre-fix this never fired in practice
  because the comparison was wrong.
- **Patch 3 is additive (warning only), no behavior change**.
- **The fork API is unaffected** — the relevant code path is
  canonical-only (gated by `WET_MASK in registry` in the LHS, and the
  ghost path is shared but the fork pre-filled ghosts to 0).

## Open questions

- Should outflow BC ghosts default to "zero-gradient" (current
  proposal: ghost = 0 → diffusion contribution proportional to
  interior c) or "true zero-gradient" (ghost = interior c at the
  edge, no net diffusion contribution)? The proposed Patch 1 fix
  treats it as ghost = 0. For zero-gradient outflow semantics
  matching the typical wet-domain finite-volume convention,
  `ghost_concs = interior_c[internal_cell_index]` would be more
  correct physically. Worth a separate discussion / phase doc; the
  current proposal preserves the historical fork default which was
  almost certainly ghost = 0.

- Phase-D Unit B was disabled-by-default at one point
  (`reconstruct_newly_wet=False`) on the streaming reference baseline.
  After Patch 2, is it safe to re-enable by default? The original
  rationale for `False` was performance on dry-start RAS HDFs where
  newly-wet cells overwhelmingly lacked a qualifying upstream neighbor
  anyway. After Patch 1 fixes the upstream NaN issue, Unit B should
  actually fire on real wet upstream neighbors — performance evaluation
  needed.

## Evidence files

- Symptom trace: `case_studies/santiam_salem/output/dye_tracer_3src_15day_canonical/transport.nc`
  (sidecar from `19_dye_tracer_canonical.py`; `dye[t=-1]` is all zeros)
- Reference (fork-API) output:
  `case_studies/santiam_salem/output/dye_tracer_3src_15day/transport.zarr`
  (dye field is non-zero and propagates as expected)
- Driver config: `case_studies/santiam_salem/output/dye_tracer_3src_15day_canonical/dye_riverine_config.yml`
- Driver script: `case_studies/santiam_salem/scripts/19_dye_tracer_canonical.py`
