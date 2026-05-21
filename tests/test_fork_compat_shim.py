"""Fork-compat shim tests (Phase F enabler).

The shim exposes the canonical VariableRegistry as the
fork-style ``model.mesh`` API the Phase-2 ESM streaming
orchestrator (``08_run_coupled_v3_smoke.py``) was written against,
plus extends ``model.update`` and ``model.finalize`` to accept the
fork's optional kwargs without changing default behavior.

The tests use a minimal hand-rolled registry rather than a full model
construction so they focus on the shim contract in isolation. The
no-regression check for the model-level wiring (default no-arg
``update()`` and ``finalize()``) is the existing 94-test suite, which
runs alongside this file in CI.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from clearwater_data.variables import VariableRegistry
from clearwater_data.variables.xarray import DataArrayVariable
from clearwater_data.variables.float import FloatVariable

from clearwater_riverine.fork_compat import (
    MeshView,
    apply_update_concentration,
)
from clearwater_riverine.variables import (
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)


T0 = datetime(2026, 1, 1, 0, 0, 0)
DT_SEC = 60.0
T1 = T0 + timedelta(seconds=DT_SEC)


def _make_time_var(values_t, values_t1, *, space_dim):
    arr = np.stack([np.asarray(values_t), np.asarray(values_t1)], axis=0)
    return DataArrayVariable(
        xr.DataArray(
            arr.astype(float),
            dims=("time", space_dim),
            coords={"time": [T0, T1], space_dim: np.arange(arr.shape[1])},
        ),
        space_dimension=space_dim,
    )


def _make_registry(nreal=3, nghost=1):
    """Minimal registry covering the fork-compat surface."""
    nface = nreal + nghost
    registry = VariableRegistry()
    registry.register(NUMBER_OF_REAL_CELLS, FloatVariable(int(nreal)))
    registry.register(NFACE, FloatVariable(nface))
    # VOLUME is the reference variable for time / nface dim sizes.
    registry.register(
        VOLUME,
        _make_time_var([10.0] * nface, [10.0] * nface, space_dim="nface"),
    )
    # An extensive constituent (the fork orchestrator's typical case).
    registry.register(
        "water_temp_C",
        _make_time_var([17.35] * nface, [17.35] * nface, space_dim="nface"),
    )
    # A static topology variable to exercise the static-read path.
    registry.register(
        "edges_face1",
        DataArrayVariable(
            xr.DataArray(
                np.array([0, 1, 2], dtype=np.int64),
                dims=("nedge",),
                coords={"nedge": np.arange(3)},
            ),
            space_dimension="nedge",
        ),
    )
    return registry


# --- MeshView read surface ------------------------------------------------


def test_getitem_returns_registry_dataarray():
    """``mesh[name]`` returns the underlying DataArray, not a copy."""
    registry = _make_registry()
    mesh = MeshView(registry)
    da = mesh["water_temp_C"]
    assert isinstance(da, xr.DataArray)
    # Identity check: same object the registry holds.
    assert da is registry.get_variable("water_temp_C").get()


def test_getitem_time_returns_time_coord():
    """``mesh['time']`` returns the time coord (a DataArray)."""
    registry = _make_registry()
    mesh = MeshView(registry)
    t_coord = mesh["time"]
    assert isinstance(t_coord, xr.DataArray)
    # The first entry of the time coord is T0.
    assert np.datetime64(T0) == t_coord.values[0]


def test_contains_proxies_registry():
    """``name in mesh`` matches ``name in registry``."""
    registry = _make_registry()
    mesh = MeshView(registry)
    assert "water_temp_C" in mesh
    assert "edges_face1" in mesh
    assert "definitely_not_a_real_var" not in mesh


def test_data_vars_supports_membership():
    """``name in mesh.data_vars`` proxies to ``__contains__``."""
    registry = _make_registry()
    mesh = MeshView(registry)
    assert "water_temp_C" in mesh.data_vars
    assert "missing" not in mesh.data_vars


def test_coords_supports_membership():
    """``name in mesh.coords`` proxies to ``__contains__``."""
    registry = _make_registry()
    mesh = MeshView(registry)
    # The fork orchestrator only checks names via mesh.coords; the
    # shim treats data_vars and coords identically against the
    # registry's flat namespace.
    assert "edges_face1" in mesh.coords


def test_sizes_time_and_nface():
    """``mesh.sizes['time']`` and ``mesh.sizes['nface']`` return ints."""
    registry = _make_registry(nreal=3, nghost=1)
    mesh = MeshView(registry)
    assert mesh.sizes["time"] == 2
    assert mesh.sizes["nface"] == 4


def test_sizes_rejects_unknown_dim():
    """Unknown dim names raise KeyError to surface unexpected accesses."""
    registry = _make_registry()
    mesh = MeshView(registry)
    with pytest.raises(KeyError, match="MeshView.sizes only supports"):
        mesh.sizes["nedge"]


def test_nreal_is_integer():
    """``mesh.nreal`` returns int (the fork's ``mesh.nreal`` is int)."""
    registry = _make_registry(nreal=3)
    mesh = MeshView(registry)
    assert mesh.nreal == 3
    assert isinstance(mesh.nreal, int)


def test_time_coord_indexable():
    """``mesh.time[idx]`` is indexable like the fork's mesh.time."""
    registry = _make_registry()
    mesh = MeshView(registry)
    # Fork pattern: transport.mesh.time[hdf_idx]
    assert mesh.time[0].values == np.datetime64(T0)
    assert mesh.time[1].values == np.datetime64(T1)


def test_nface_coord_values():
    """``mesh.nface.values`` is indexable like the fork's pattern."""
    registry = _make_registry(nreal=3, nghost=1)
    mesh = MeshView(registry)
    # Fork pattern: transport.mesh.nface.values[0:nface_real]
    np.testing.assert_array_equal(mesh.nface.values, np.arange(4))


# --- MeshView write-through ------------------------------------------------


def test_loc_write_propagates_to_registry():
    """``mesh[name].loc[...] = arr`` mutates the registry in place.

    This is the exact write pattern from the fork orchestrator at
    line 1669 of ``08_run_coupled_v3_smoke.py``.
    """
    registry = _make_registry(nreal=3, nghost=1)
    mesh = MeshView(registry)
    new_T = np.array([20.0, 21.0, 22.0])
    mesh["water_temp_C"].loc[
        {"time": mesh.time[1], "nface": mesh.nface.values[0:3]}
    ] = new_T
    # Read back from the registry (not via the shim) and confirm the
    # write landed.
    stored = registry.get_variable("water_temp_C").get()
    written = stored.isel(time=1, nface=slice(0, 3)).values
    np.testing.assert_array_equal(written, new_T)
    # The ghost cell (index 3) should be untouched.
    assert stored.isel(time=1, nface=3).values == 17.35


def test_isel_returns_consistent_values():
    """``mesh[name].isel(time=t, nface=real)`` returns the registry data."""
    registry = _make_registry(nreal=3, nghost=1)
    mesh = MeshView(registry)
    real = np.arange(3)
    ic = mesh["water_temp_C"].isel(time=0, nface=real).values
    np.testing.assert_array_equal(ic, [17.35, 17.35, 17.35])


# --- apply_update_concentration -------------------------------------------


def test_apply_update_concentration_noop_on_none():
    """No-op when the dict is None or empty."""
    registry = _make_registry()
    apply_update_concentration(registry, T0, 4, None)
    apply_update_concentration(registry, T0, 4, {})
    # Registry value unchanged.
    da = registry.get_variable("water_temp_C").get()
    assert da.isel(time=0, nface=0).values == 17.35


def test_apply_update_concentration_writes_first_nreal_plus_ghost():
    """Override writes to the first nreal+1 slots at current_time."""
    registry = _make_registry(nreal=3, nghost=1)
    nreal_plus_ghost = 4
    override = np.array([25.0, 26.0, 27.0, 28.0])
    apply_update_concentration(
        registry, T0, nreal_plus_ghost,
        {"water_temp_C": override},
    )
    da = registry.get_variable("water_temp_C").get()
    np.testing.assert_array_equal(
        da.isel(time=0).values,
        override,
    )
    # The T1 slot is untouched.
    np.testing.assert_array_equal(
        da.isel(time=1).values,
        np.full(4, 17.35),
    )


def test_apply_update_concentration_unwraps_dataarray():
    """Override values may be DataArrays; the helper reads ``.values``."""
    registry = _make_registry(nreal=3, nghost=1)
    override = xr.DataArray(
        np.array([30.0, 31.0, 32.0, 33.0]),
        dims=("nface",),
    )
    apply_update_concentration(
        registry, T0, 4,
        {"water_temp_C": override},
    )
    da = registry.get_variable("water_temp_C").get()
    np.testing.assert_array_equal(
        da.isel(time=0).values,
        [30.0, 31.0, 32.0, 33.0],
    )


def test_apply_update_concentration_multiple_constituents():
    """Multiple keys in the dict each write to their named variable."""
    registry = _make_registry(nreal=3, nghost=1)
    # Add a second variable so the test covers more than one route.
    registry.register(
        "DOX",
        _make_time_var([9.4] * 4, [9.4] * 4, space_dim="nface"),
    )
    apply_update_concentration(
        registry, T0, 4,
        {
            "water_temp_C": np.array([20.0, 20.0, 20.0, 20.0]),
            "DOX": np.array([8.0, 8.0, 8.0, 8.0]),
        },
    )
    np.testing.assert_array_equal(
        registry.get_variable("water_temp_C").get().isel(time=0).values,
        [20.0, 20.0, 20.0, 20.0],
    )
    np.testing.assert_array_equal(
        registry.get_variable("DOX").get().isel(time=0).values,
        [8.0, 8.0, 8.0, 8.0],
    )
