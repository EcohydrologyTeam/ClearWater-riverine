"""Phase-D Unit C-alpha: LHS wet-dry edge filter, rule-1 dry-cell pinning,
and rule-3 donor-diagonal amendment.

Each test builds a minimal hand-rolled registry and runs
``LHS.update_values`` directly, then inspects the resulting sparse-matrix
triples to verify the documented behavior.

The opt-out path (no ``WET_MASK`` in the registry) is exercised by the
existing 49-test guard suite at every commit; the tests here cover the
opt-in path the canonical code did not previously support.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.linalg import LHS
from clearwater_riverine.variables import (
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    EDGE_FACE_CONNECTIVITY,
    EDGES_FACE1,
    EDGES_FACE2,
    FLOW_ACROSS_FACE,
    NEDGE,
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    WET_MASK,
)


T0 = datetime(2026, 1, 1, 0, 0, 0)
T1 = T0 + timedelta(seconds=60)


def _make_time_var(values_t, values_t1, space_dim):
    arr = np.stack([np.asarray(values_t), np.asarray(values_t1)], axis=0)
    return DataArrayVariable(
        xr.DataArray(
            arr,
            dims=("time", space_dim),
            coords={"time": [T0, T1], space_dim: np.arange(arr.shape[1])},
        ),
        space_dimension=space_dim,
    )


def _make_lhs_registry(
    *,
    nreal,
    nghost,
    edges,
    flow_t,
    volume_t,
    volume_t1,
    wet_t=None,
    wet_t1=None,
    diff_coef_t=None,
):
    """Build a minimal registry suitable for ``LHS(registry).update_values``."""
    nface = nreal + nghost
    ef = np.asarray(edges, dtype=int)
    nedge = len(edges)
    registry = VariableRegistry()

    # Real model registers nreal as a numpy integer (from .max() + 1),
    # which Python treats as an integer in arithmetic so pre-allocation
    # downstream stays int-typed. Match that here.
    registry.register(NUMBER_OF_REAL_CELLS, FloatVariable(int(nreal)))
    registry.register(CHANGE_IN_TIME, FloatVariable(60.0))

    # EDGE_FACE_CONNECTIVITY is (nedge, 2); EDGES_FACE1/2 are the columns
    # exposed separately as (nedge,) arrays.
    ef_da = xr.DataArray(
        ef, dims=("nedge", "2"), coords={"nedge": np.arange(nedge)}
    )
    registry.register(
        EDGE_FACE_CONNECTIVITY,
        DataArrayVariable(ef_da, space_dimension="nedge"),
    )
    registry.register(
        EDGES_FACE1,
        DataArrayVariable(
            xr.DataArray(ef[:, 0], dims=("nedge",), coords={"nedge": np.arange(nedge)}),
            space_dimension="nedge",
        ),
    )
    registry.register(
        EDGES_FACE2,
        DataArrayVariable(
            xr.DataArray(ef[:, 1], dims=("nedge",), coords={"nedge": np.arange(nedge)}),
            space_dimension="nedge",
        ),
    )

    registry.register(
        VOLUME, _make_time_var(volume_t, volume_t1, "nface"),
    )
    registry.register(
        FLOW_ACROSS_FACE,
        _make_time_var(flow_t, flow_t, "nedge"),
    )
    diff = (
        np.zeros(nedge) if diff_coef_t is None
        else np.asarray(diff_coef_t)
    )
    registry.register(
        COEFFICIENT_TO_DIFFUSION_TERM,
        _make_time_var(diff, diff, "nedge"),
    )
    if wet_t is not None and wet_t1 is not None:
        registry.register(
            WET_MASK,
            _make_time_var(wet_t, wet_t1, "nface"),
        )
    return registry


def _assemble(lhs):
    """Build the CSR matrix from an updated LHS."""
    n = int(lhs.real_cell_count)
    return csr_matrix(
        (lhs.coefficients, (lhs.rows, lhs.columns)), shape=(n, n)
    ).toarray()


# --- tests ---------------------------------------------------------------


def test_legacy_path_runs_without_wet_mask():
    """Sanity: legacy path (no WET_MASK) still computes a non-singular
    matrix when there are no dry cells. Existing canonical behaviour."""
    registry = _make_lhs_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[0.0],
        volume_t=[10.0, 10.0],
        volume_t1=[10.0, 10.0],
    )
    lhs = LHS(registry)
    lhs.update_values(registry, T0, timedelta(seconds=60))
    A = _assemble(lhs)
    # Diagonals come from the load (V/dt = 10/60 ~ 0.167) and any
    # diffusion (zero here). Off-diagonals stay zero with zero flow.
    assert A[0, 0] > 0
    assert A[1, 1] > 0
    # Diagnostic instance attrs are empty on the legacy path.
    assert len(lhs.wet_dry_leak_donors) == 0
    assert len(lhs.wet_dry_leak_abs_adv) == 0
    assert len(lhs.dry_cells_t1) == 0


def test_persistently_dry_cell_is_pinned_to_identity():
    """Rule 1: a real cell that is dry at t+1 gets its diagonal pinned
    (+1.0 contribution) when WET_MASK is in the registry."""
    registry = _make_lhs_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[0.0],
        volume_t=[10.0, 0.0],
        volume_t1=[10.0, 0.0],
        wet_t=[True, False],
        wet_t1=[True, False],
    )
    lhs = LHS(registry)
    lhs.update_values(registry, T0, timedelta(seconds=60))
    # Dry cell 1 should be in dry_cells_t1 and its diagonal pinned.
    assert 1 in lhs.dry_cells_t1
    A = _assemble(lhs)
    # The pin contributes 1.0 to A[1,1]; load and diffusion add zero
    # (volume==0, no diffusion). Expect A[1,1] approximately 1.0.
    assert A[1, 1] == pytest.approx(1.0, abs=1e-12)


def test_wet_to_dry_donor_diagonal_amendment():
    """Rule 3 amendment: an edge with adv > 0 whose donor (face1) is wet
    at t+1 contributes +|adv| to the donor's diagonal even when the
    recipient (face2) is dry. The recipient's row gets NO off-diagonal
    coupling -- it stays pinned to identity by rule 1."""
    # Two real cells, edge 0 -> 1; cell 0 stays wet, cell 1 goes dry at
    # t+1. Positive flow of magnitude 3.0 means mass would leave cell 0
    # toward (now-dry) cell 1.
    registry = _make_lhs_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[3.0],
        volume_t=[10.0, 10.0],
        volume_t1=[10.0, 0.0],
        wet_t=[True, True],
        wet_t1=[True, False],
    )
    lhs = LHS(registry)
    lhs.update_values(registry, T0, timedelta(seconds=60))
    A = _assemble(lhs)
    # Donor (cell 0) diagonal includes the load + the +3.0 outflow
    # contribution from the rule-3 amendment. Load = V/dt = 10/60 ~
    # 0.167. Expected A[0,0] approximately 3.167.
    assert A[0, 0] == pytest.approx(10.0 / 60.0 + 3.0, abs=1e-9)
    # Recipient (cell 1) row stays clean: no off-diagonal coupling
    # from the wet-dry edge. Just the rule-1 identity pin on diagonal.
    assert A[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert A[1, 1] == pytest.approx(1.0, abs=1e-12)


def test_wet_dry_leak_diagnostic_populated():
    """The wet-dry leak diagnostic instance attrs are populated with
    (donor, |adv|) pairs that C-beta will use to compute
    mass_lost_to_dry post-solve."""
    registry = _make_lhs_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[3.0],
        volume_t=[10.0, 10.0],
        volume_t1=[10.0, 0.0],
        wet_t=[True, True],
        wet_t1=[True, False],
    )
    lhs = LHS(registry)
    lhs.update_values(registry, T0, timedelta(seconds=60))
    # One wet-dry edge: donor is cell 0, |adv| = 3.0.
    assert list(lhs.wet_dry_leak_donors) == [0]
    np.testing.assert_array_almost_equal(lhs.wet_dry_leak_abs_adv, [3.0])


def test_wet_to_wet_edge_unchanged_by_amendment():
    """Both endpoints wet at t+1 -> edge is fully active; advection
    contributes both donor diagonal AND off-diagonal coupling. The
    amendment changes nothing for wet-to-wet edges."""
    registry = _make_lhs_registry(
        nreal=2, nghost=0,
        edges=[(0, 1)],
        flow_t=[3.0],
        volume_t=[10.0, 10.0],
        volume_t1=[10.0, 10.0],
        wet_t=[True, True],
        wet_t1=[True, True],
    )
    lhs = LHS(registry)
    lhs.update_values(registry, T0, timedelta(seconds=60))
    A = _assemble(lhs)
    # Donor diagonal: load + |adv|
    assert A[0, 0] == pytest.approx(10.0 / 60.0 + 3.0, abs=1e-9)
    # Off-diagonal coupling on the wet-to-wet edge: -|adv|
    assert A[1, 0] == pytest.approx(-3.0, abs=1e-9)
    # Recipient diagonal carries only its load (no donor-diag for
    # negative flow here since adv is positive).
    assert A[1, 1] == pytest.approx(10.0 / 60.0, abs=1e-9)
    # No wet-dry edges -> no leak diagnostic entries.
    assert len(lhs.wet_dry_leak_donors) == 0
