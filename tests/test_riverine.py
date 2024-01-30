import pytest
import clearwater_riverine as cwr
import xarray as xr

from typing import (
    Tuple
)

@pytest.fixture
def data_dir() -> str:
    return './tests/data/simple_test_cases'

@pytest.fixture
def diffusion_coefficient() -> float:
    return 0.001

@pytest.fixture
def sim02(data_dir) -> str:
    return data_dir + r'/plan02_2x1/'

@pytest.fixture
def plan02(sim02) -> str:
    return sim02 + 'clearWaterTestCases.p02.hdf'

@pytest.fixture
def initial_condition_path(sim02) -> str:
    return sim02 + 'cwr_initial_conditions_p02.csv'

@pytest.fixture
def boundary_condition_path(sim02) -> str:
    return sim02 + 'cwr_boundary_conditions_p02.csv'

@pytest.fixture
def date_range_index() -> Tuple[int, int]:
    return (5, 8)

@pytest.fixture
def date_range_string() -> Tuple[int, int]:
    return ('01-01-2023 12:00:00', '01-01-2023 12:10:00') 

@pytest.fixture
def cwr_instance(
    plan02,
    diffusion_coefficient
) -> cwr.ClearwaterRiverine:
    """Return an instance of a ClearwaterRiverine class"""
    return cwr.ClearwaterRiverine(
        ras_file_path=plan02,
        diffusion_coefficient_input=diffusion_coefficient
    )

@pytest.fixture
def clearwater_riverine_index_instance(
    plan02,
    diffusion_coefficient,
    date_range_index,
) -> cwr.ClearwaterRiverine:
    """Return an instance of a ClearwaterRiverine class"""
    return cwr.ClearwaterRiverine(
        ras_file_path=plan02,
        diffusion_coefficient_input=diffusion_coefficient,
        datetime_range=date_range_index
    )

@pytest.fixture
def clearwater_riverine_string_instance(
    plan02,
    diffusion_coefficient,
    date_range_string,
) -> cwr.ClearwaterRiverine:
    """Return an instance of a ClearwaterRiverine class"""
    return cwr.ClearwaterRiverine(
        ras_file_path=plan02,
        diffusion_coefficient_input=diffusion_coefficient,
        datetime_range=date_range_string,
    )

def test_datetime_range(
    cwr_instance,
    clearwater_riverine_index_instance,
    clearwater_riverine_string_instance,
):
    """Test all datetime_range options."""
    assert len(cwr_instance.mesh.time) == 25
    assert len(clearwater_riverine_index_instance.mesh.time) == 4
    assert len(clearwater_riverine_string_instance.mesh.time) == 3


def test_riverine_initialize(
        cwr_instance,
        initial_condition_path,
        boundary_condition_path
    ):
    """Test initialize method."""
    cwr_instance.initialize(
        initial_condition_path=initial_condition_path,
        boundary_condition_path=boundary_condition_path,
        units='degC'
    )
    concentration = cwr_instance.mesh.concentration
    assert concentration.isel(time=0,nface=0).values == 100
    upstream_boundary = concentration.isel(time=0, nface=4)
    downstream_boundary = concentration.isel(time=0, nface=6)

    assert upstream_boundary.values == 100
    assert downstream_boundary.values == 100
    assert concentration.attrs['Units'] == 'degC'

def test_riverine_update(
        cwr_instance,
        initial_condition_path,
        boundary_condition_path
    ):
    """Test clearwater riverine update method"""
    
    cwr_instance.initialize(
        initial_condition_path=initial_condition_path,
        boundary_condition_path=boundary_condition_path,
        units='degC'
    )
    assert cwr_instance.time_step == 0
    assert cwr_instance.mesh.concentration.isel(time=1, nface=0).values == 0

    cwr_instance.update()
    assert cwr_instance.mesh.concentration.isel(time=1, nface=0).values != 0
    assert cwr_instance.time_step == 1
    assert cwr_instance.mesh.concentration.isel(time=1, nface=4).values == 100
    assert cwr_instance.mesh.concentration.isel(time=1, nface=6).values == 100