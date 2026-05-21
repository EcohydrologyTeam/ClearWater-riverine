# Internal-Type Boundary Condition Audit (Phase F T2-E)

**Date:** 2026-05-21
**Outcome:** Latent defect identified on both ClearWater-riverine (canonical) and ClearWater-Riverine-streaming. Not blocking Phase F Santiam-Salem validation because the subset extractor flattens the affected BC representation. Warning added; full fix tracked here for future work.

## Background

HEC-RAS 2D models can declare boundary condition (BC) lines in two flavors:

- **External BCs**: Polylines drawn at the perimeter of the 2D mesh. Mass / flow enters through cell faces on the perimeter. The HDF stores the per-face data in `Geometry/Boundary Condition Lines/External Faces`.
- **Internal BCs**: Polylines drawn through the interior of the 2D mesh (e.g., for a tributary that joins a main channel mid-domain, or for a forcing line that does not coincide with the mesh perimeter). The HDF stores the per-cell data in `Geometry/Boundary Condition Lines/Internal Cells`.

Both flavors carry the same time series of flows / stages / concentrations. The difference is geometrical: which mesh cells the BC affects, and through which faces or cells the mass enters.

## Audit finding

Both canonical and the streaming repo read **only** `External Faces` from the HDF:

```python
# both repos, identical pattern:
external_faces = pd.DataFrame(
    infile[self.paths['boundary_condition_external_faces']][()]
)
```

Neither repo reads `Internal Cells`. When a RAS HDF declares an Internal-type BC line:

- The BC's flow time series enters the HEC-RAS hydrodynamics correctly (RAS handles Internal BCs internally during the 2D solve).
- The HEC-RAS HDF writes the resulting face flows / volumes everywhere they actually occur, including at Internal-BC cells.
- But when canonical (or streaming) reads the HDF to set up the WQ transport, it does NOT extract the Internal-Cells dataset; mass injected at Internal BCs is silently absent from the ghost-cell BC injection.

For a model with an Internal BC carrying a non-trivial constituent concentration, this would produce a downstream cold spot / dilution that does not reflect the user's BC concentration time series.

## Why Phase F validation is unaffected

The Santiam-Salem source HDF (`Santiam_Salem.p01.hdf`) DOES contain Internal BCs:

| Name | Type | Length (m) |
|---|---|---|
| Upstream | **Internal** | 200.8 |
| Santiam | **Internal** | 106.9 |
| Downstream | External | 918.3 |

But the subset extractor (Stage 04e/06c in the `ClearWater-modules-phase2-ESM-streaming/case_studies/santiam_salem/scripts/` chain) produces a different HDF (`santiam_salem_subset_2008-09_hourly.p01.hdf`) where the BC structure has been transformed:

| Name | Type | Notes |
|---|---|---|
| Upstream | `Flow Hydrograph` | External faces only; 100 perimeter faces synthesized |
| Santiam | `Flow Hydrograph` | (Downstream BC dropped because outside the clip) |

The subset extractor maps the Internal BCs onto **External-face representations** of the clipped mesh, and the `Internal Cells` dataset is not written to the subset HDF. Both canonical and streaming then correctly read all the BC data via `External Faces`.

Phase F Salem T validation (canonical bias -0.52 °C, RMSE 0.56 °C versus streaming locked baseline -0.30 °C, RMSE 0.62 °C) is therefore unaffected by this issue.

## When the defect bites

A user running canonical (or streaming) on the SOURCE HDF directly — i.e., without going through the subset extractor — would miss the Upstream and Santiam BC mass entirely. The downstream WQ field would show a steep dilution / undershoot that is not real.

More broadly, ANY RAS HDF with at least one Internal-type BC line will silently drop mass at those BC locations.

## Defensive warning (committed)

`io/hdf.py:__define_boundary_hydrodynamics` now inspects the `Geometry/Boundary Condition Lines/Attributes` table's `Type` field. If any row reads `Internal` (case-insensitive), a `UserWarning` is emitted listing the affected BC names and explaining the limitation. The warning fires on every model construction from an un-subset HDF with Internal BCs.

## Fix plan (future Tier 3 work)

To fully resolve, the HDF reader needs to:

1. Read `Geometry/Boundary Condition Lines/Internal Cells` (shape `(N, 4)`, dtype `[(BC Line ID, Cell Index, Station Start, Station End)]`).
2. For each row, allocate a virtual ghost cell at the `Cell Index` location (or use a different mass-injection mechanism that doesn't require ghost cells).
3. Join with `Attributes` on `BC Line ID` to get the BC's name and time series.
4. Wire the resulting per-cell BC mass into the existing ghost-cell BC injection path in `Constituent.set_boundary_conditions`.

Subtlety: Internal-BC cells are real cells (not perimeter ghosts), so the upstream/downstream face-direction conventions don't apply the same way. The mass injection is a volumetric source at the cell, not a face flow. The RHS contribution is therefore similar to the `point_sources` path added in Phase F T2-A — `Flow_Rate × Concentration` added to the RHS load for that cell at each step. Reusing the T2-A infrastructure (with the BC flow as the source Flow_Rate and the BC concentration as the source Concentration) is probably the cleanest implementation.

## Recommendation

- For real-world canonical applications: continue running through the subset-extractor pattern that flattens Internal BCs into External-face representations. The subset script is the right place for this transformation (each RAS deck's mesh-and-BC topology varies, so a generic in-reader handler is harder than a per-deck subset script).
- For future canonical work targeting full-mesh RAS HDFs: implement the fix above, ideally by extending the T2-A point_sources mechanism to accept BC time series as a source.
- For now: the warning in `io/hdf.py` is sufficient to surface the issue to users.
