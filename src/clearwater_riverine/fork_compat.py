"""Fork-compat shim layer.

The streaming fork (`ClearWater-Riverine-streaming`) exposes the transport
state as an xarray Dataset on ``model.mesh``. The canonical model
(`ClearWater-riverine`, PR #135 lineage) exposes the same state as a
``VariableRegistry`` on ``model.registry``. Code originally written
against the fork's API, including the Phase-2 ESM streaming case-study
orchestrator (`08_run_coupled_v3_smoke.py`), uses the fork-style
``model.mesh[...]`` access pattern for reads and writes.

This module supplies a ``MeshView`` that wraps a ``VariableRegistry``
and presents the subset of the xarray Dataset API the fork-side
orchestrator uses. The view does not copy data: the DataArrays it
returns are the same objects the registry stores, so ``mesh[X].loc[...]
= arr`` writes mutate the registry in place. Companion changes to
``ClearwaterRiverine.update`` and ``ClearwaterRiverine.finalize`` accept
the fork's optional kwargs (`update_concentration`, `save`,
`output_filepath`) without changing the no-arg defaults.

Scope: covers the exact access surface inventoried in
``design/willamette_validation_plan.md`` (F1 step 1). New access
patterns added later should extend this view explicitly rather than be
deduced by `__getattr__` magic; the shim's purpose is a small, audited
surface, not a full xarray-Dataset emulator.

Adopted 2026-05-20 on `steissberg-riverine-merged` as the Phase F
enabler for the Santiam-Salem validation reproduction (compared
against fork run `v3_smoke_15day_wind10m_final_mumax_1_3` baseline).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from clearwater_data.variables import VariableRegistry

from clearwater_riverine.variables import (
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)


__all__ = ["MeshView"]


def _get(registry: VariableRegistry, name: str) -> xr.DataArray:
    """Return the underlying mutable DataArray for ``name``.

    Goes through ``get_variable(...).get()`` rather than ``registry.get(...)``
    to avoid the deprecation warning the latter emits. The returned
    DataArray is the registry's storage, not a copy; mutations via
    ``.loc[...] = arr`` propagate.
    """
    return registry.get_variable(name).get()


class MeshView:
    """Read-mostly view of a VariableRegistry shaped like ``model.mesh``.

    Exposes the exact access patterns inventoried in F1 step 1 of the
    Willamette validation plan: keyed reads, membership tests on
    ``data_vars`` and ``coords``, ``sizes`` for ``time`` and ``nface``,
    direct ``time`` and ``nface`` coordinate arrays, and ``nreal``.
    Writes through ``mesh[name].loc[...] = arr`` work natively because
    the returned DataArray is the registry's storage.
    """

    def __init__(self, registry: VariableRegistry):
        self._registry = registry

    # --- keyed access ----------------------------------------------------

    def __getitem__(self, name: str) -> xr.DataArray:
        if name == "time":
            # ``mesh["time"]`` historically returns the time coord as a
            # DataArray. Pull it from VOLUME which always has a time
            # dim by construction.
            return _get(self._registry, VOLUME).time
        return _get(self._registry, name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    # --- xarray-Dataset-like introspection -------------------------------

    @property
    def data_vars(self):
        # The fork orchestrator only uses ``data_vars`` for ``name in
        # mesh.data_vars`` membership tests. Returning ``self`` is
        # sufficient because ``__contains__`` already proxies to the
        # registry. The same view also satisfies ``mesh.coords``.
        return self

    @property
    def coords(self):
        return self

    @property
    def sizes(self):
        return _SizesView(self._registry)

    @property
    def nreal(self) -> int:
        """Number of real (non-ghost) cells.

        Fork's ``mesh.nreal`` is an integer. Canonical stores it as a
        FloatVariable (the ``register(NUMBER_OF_REAL_CELLS, ...)`` line
        in ``model.__init_model``); the value is whole-number, the type
        is float for FloatVariable's contract. Cast to int here.
        """
        return int(self._registry.get_variable(NUMBER_OF_REAL_CELLS).get())

    @property
    def time(self):
        return _get(self._registry, VOLUME).time

    @property
    def nface(self):
        return _get(self._registry, VOLUME).nface


class _SizesView:
    """View supporting ``mesh.sizes['time']`` and ``mesh.sizes['nface']``.

    Backs ``mesh.sizes`` with a registry-derived dict. The fork
    orchestrator only reads these two keys; other keys raise KeyError
    to surface unexpected accesses early.
    """

    def __init__(self, registry: VariableRegistry):
        self._registry = registry

    def __getitem__(self, dim: str) -> int:
        if dim == "time":
            return int(_get(self._registry, VOLUME).sizes["time"])
        if dim == "nface":
            return int(self._registry.get_variable(NFACE).get())
        raise KeyError(
            f"MeshView.sizes only supports 'time' and 'nface'; got {dim!r}. "
            "If a new fork-compat access pattern is needed, extend the shim "
            "in fork_compat.py."
        )

    def __contains__(self, dim: str) -> bool:
        return dim in ("time", "nface")


def apply_update_concentration(
    registry: VariableRegistry,
    current_time,
    nreal_plus_ghost_count: int,
    update_concentration: Optional[dict],
) -> None:
    """Fork-compat helper: apply ``update_concentration`` overrides
    in place at ``current_time``.

    Mirrors the fork's behaviour: for each constituent in the dict,
    write the override values into the first ``nreal+1`` (real cells +
    ghost slot) at the current-time slot. The transport solver's IC at
    the next step reads from this slot, so the override propagates
    into the solve without changes to the transport engine.

    No-op when ``update_concentration`` is None or empty. Values may be
    plain arrays or xarray DataArrays (the latter unwrapped via
    ``.values``).
    """
    if not update_concentration:
        return
    write_count = int(nreal_plus_ghost_count)
    for name, value in update_concentration.items():
        if hasattr(value, "values"):
            arr = np.asarray(value.values)
        else:
            arr = np.asarray(value)
        da = _get(registry, name)
        # Direct slice assignment on the time-resident sub-array.
        # ``da.sel(time=current_time)`` returns a view; ``values[...]``
        # mutates the underlying buffer.
        da.sel(time=current_time).values[:write_count] = arr[:write_count]
