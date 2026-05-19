"""Disk-based cache for the static portion of a Clearwater Riverine mesh.

The HEC-RAS HDF source for a real corridor (e.g., Albany, 587k cells)
contains thousands of small geometry datasets (face/cell coordinates,
edge-face connectivity, polygon vertices, volume-elevation lookup
tables). Walking those on every model init costs 30-60 minutes of
wall-clock per run, dominated (>95% by sample profiling) by tiny
``H5Dread`` calls. The geometry is fully deterministic from the source
HDF, so we cache it on disk after the first build and rehydrate on
subsequent runs.

Only STATIC mesh outputs are cached. Time-varying hydrodynamic arrays
(``edge_velocity``, ``water_surface_elev``, ``volume``, ``face_flow``,
eddy viscosities, gate flows, etc.) are still read fresh from the HDF
on every run because they are I/O-cheap (sequential slabs) compared to
the per-cell static loops.

Cache key inputs:
    - HDF source identity: file size + mtime + a 64-KiB head/tail hash.
      We deliberately avoid hashing multi-GB files; size+mtime catches
      every realistic edit and the head/tail hash catches the degenerate
      "two files of the same size and mtime" collision.
    - ``clearwater_riverine.__version__`` -- bumping the package
      version invalidates all caches.
    - ``CACHE_SCHEMA_VERSION`` -- bump in code if the cache payload
      structure changes.
    - Mesh-affecting constructor args (currently none beyond the HDF
      itself; ``datetime_range`` and ``diffusion_coefficient_input`` do
      not affect static geometry, so they are intentionally excluded
      from the key).

Format: gzipped pickle. The static payload contains an
``xarray.Dataset`` plus a handful of ``pandas.DataFrame`` attributes
(boundary data, volume/area-elevation lookups). Pickle is the only
format that round-trips this mix without a custom encoder; gzip keeps
the on-disk size reasonable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


CACHE_SCHEMA_VERSION = 1
"""Bump when the static payload structure changes (data var names,
attr names, dtypes, etc.). All on-disk caches with a different schema
version are treated as stale and rebuilt.
"""

CACHE_DIR_NAME = ".cwr_mesh_cache"
"""Default sibling directory next to the source HDF."""

_HEAD_TAIL_BYTES = 64 * 1024
"""How many bytes to hash from the start and end of the HDF when
deriving the cache key. Hashing the full multi-GB file would defeat
the whole point of the cache."""


@dataclass
class CacheKeyInputs:
    """All inputs that feed the cache key hash.

    Anything that, if changed, should invalidate the cache must live
    here. Keep it small and JSON-serializable.
    """
    hdf_size: int
    hdf_mtime_ns: int
    hdf_head_tail_sha1: str
    cwr_version: str
    schema_version: int
    extra: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "hdf_size": self.hdf_size,
                "hdf_mtime_ns": self.hdf_mtime_ns,
                "hdf_head_tail_sha1": self.hdf_head_tail_sha1,
                "cwr_version": self.cwr_version,
                "schema_version": self.schema_version,
                "extra": self.extra,
            },
            sort_keys=True,
        )


def _hash_head_tail(hdf_path: Path) -> str:
    """SHA-1 over the first and last ``_HEAD_TAIL_BYTES`` of the file.

    For files smaller than ``2 * _HEAD_TAIL_BYTES`` we hash the whole
    file (the head and tail will overlap, but the result is still a
    deterministic function of the file contents).
    """
    h = hashlib.sha1()
    size = hdf_path.stat().st_size
    with hdf_path.open("rb") as f:
        head = f.read(_HEAD_TAIL_BYTES)
        h.update(head)
        if size > _HEAD_TAIL_BYTES:
            tail_offset = max(_HEAD_TAIL_BYTES, size - _HEAD_TAIL_BYTES)
            f.seek(tail_offset)
            tail = f.read(_HEAD_TAIL_BYTES)
            h.update(tail)
    return h.hexdigest()


def build_cache_key_inputs(
    hdf_path: Path,
    cwr_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> CacheKeyInputs:
    """Construct the full set of inputs that determine the cache key."""
    stat = hdf_path.stat()
    return CacheKeyInputs(
        hdf_size=stat.st_size,
        hdf_mtime_ns=stat.st_mtime_ns,
        hdf_head_tail_sha1=_hash_head_tail(hdf_path),
        cwr_version=cwr_version,
        schema_version=CACHE_SCHEMA_VERSION,
        extra=dict(extra or {}),
    )


def cache_key_hash(inputs: CacheKeyInputs) -> str:
    """Stable 16-char hex digest of the cache key inputs."""
    return hashlib.sha1(inputs.to_json().encode("utf-8")).hexdigest()[:16]


def default_cache_dir(hdf_path: Path) -> Path:
    """Sibling directory next to the source HDF where caches live."""
    return hdf_path.parent / CACHE_DIR_NAME


def cache_file_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pkl.gz"


def read_cache(path: Path) -> Optional[Dict[str, Any]]:
    """Load a cached payload, or return None on any failure.

    Failures here (corrupt pickle, partial write, version drift) must
    NEVER surface to the caller as a successful load -- they have to
    fall through to the rebuild path. We log nothing here; the caller
    decides whether to print a stale-cache message.
    """
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return payload


def write_cache(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic-ish gzipped pickle write.

    We write to a sibling temp file then rename, so a crash mid-write
    cannot leave a half-written cache that would be loaded on the next
    run. ``Path.replace`` is atomic on POSIX and best-effort on
    Windows -- both are fine for our purposes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
