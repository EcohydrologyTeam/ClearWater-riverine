"""Per-constituent ``two_way_coupling`` flag: config round-trip and the
``ClearwaterRiverine.constituent_coupling()`` accessor.

The flag is consumed by an external kinetics model (e.g. the
ClearWater-modules ``Riverine`` bridge), not by the transport solver. It
controls whether a coupled kinetics model's writes feed back into
transport (``True``, default, shared buffer) or are discarded
(``False``, isolated snapshot). These tests assert the flag parses from
the constituent config with the correct default and is exposed on the
model.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import pytest
import yaml

import clearwater_riverine as cwr

_HERE = Path(__file__).resolve().parent
_PLAN = (
    _HERE / "data" / "simple_test_cases" / "plan02_2x1"
    / "clearWaterTestCases.p02.hdf"
)
_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/"
    "Unsteady Time Series/Time Date Stamp"
)


pytestmark = pytest.mark.skipif(
    not _PLAN.exists(), reason="plan02 fixture missing"
)


def _time_bounds(hdf_path: Path):
    with h5py.File(hdf_path, "r") as f:
        raw = f[_TIME_PATH][()]
    stamps = pd.to_datetime(
        pd.Series(raw).str.decode("utf8"), format="%d%b%Y %H:%M:%S"
    )
    return stamps.iloc[0], stamps.iloc[-1]


def _const(two_way=None):
    block = {
        "initial_conditions": {"provider": "float", "data": {"value": 1.0}},
        "boundary_conditions": {"provider": "float", "data": {"value": 1.0}},
    }
    if two_way is not None:
        block["two_way_coupling"] = two_way
    return block


def _build(tmp_path, constituents):
    start, end = _time_bounds(_PLAN)
    cfg = {
        "model": {
            "simulation_directory": str(tmp_path),
            "hydrodynamic_input": str(_PLAN),
            "start_datetime": str(start),
            "end_datetime": str(end),
            "diffusion_coefficient": 0.01,
            "output_variables": [],
            "mass_flux_calculation": False,
        },
        "constituents": constituents,
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cwr.ClearwaterRiverine(config_filepath=str(cfg_path))


def test_two_way_coupling_defaults_true(tmp_path):
    model = _build(tmp_path, {"tracer": _const()})
    assert model.constituent_coupling() == {"tracer": True}


def test_two_way_coupling_explicit_false(tmp_path):
    model = _build(tmp_path, {"tracer": _const(two_way=False)})
    assert model.constituent_coupling() == {"tracer": False}


def test_two_way_coupling_mixed(tmp_path):
    model = _build(
        tmp_path,
        {
            "Ap": _const(),                 # default -> True
            "NH4": _const(two_way=False),   # explicit one-way
            "NO3": _const(two_way=True),    # explicit two-way
        },
    )
    assert model.constituent_coupling() == {
        "Ap": True,
        "NH4": False,
        "NO3": True,
    }
