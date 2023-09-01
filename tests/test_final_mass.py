from clearwater_riverine.postproc_util import _run_simulation
from clearwater_riverine.postproc_util import _mass_bal_global
from clearwater_riverine.postproc_util import _mass_bal_global_100_Ans
from clearwater_riverine.postproc_util import _mass_bal_val
import pytest


#NOTE: Relative paths below are referenced from the root directory of the repo
#      i.e. please run pytest only at the ./ClearWater-riverine directory
#      level or pytest will give the "FileNotFoundError" message.

@pytest.fixture
def data_dir() -> str:
    return './tests/data/simple_test_cases'


@pytest.fixture
def sim02(data_dir) -> str:
    return data_dir + r'/plan02_2x1'

@pytest.fixture
def plan02(sim02):
    fpath = sim02 + r'/clearWaterTestCases.p02.hdf'
    diff_coef = 0.1
    intl_cond = sim02 + r'/cwr_initial_conditions_p02.csv'
    bndry_cond = sim02 + r'/cwr_boundary_conditions_p02.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan02(plan02):
    massBal_df = _mass_bal_global(plan02)
    massBal_df_100 = _mass_bal_global_100_Ans(plan02)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


@pytest.fixture
def sim03(data_dir) -> str:
    return data_dir + r'/plan03_2x1'

@pytest.fixture
def plan03(sim03):
    fpath = sim03 + r'/clearWaterTestCases.p03.hdf'
    diff_coef = 0.1
    intl_cond = sim03 + r'/cwr_initial_conditions_p03.csv'
    bndry_cond = sim03 + r'/cwr_boundary_conditions_p03.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

@pytest.mark.skip(reason="#this test is only needed in limited testing circumstances and takes longer to run")
def test_mass_end_plan03(plan03):
    massBal_df = _mass_bal_global(plan03)
    massBal_df_100 = _mass_bal_global_100_Ans(plan03)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer



@pytest.fixture
def sim01(data_dir) -> str:
    return data_dir + r'/plan01_10x5'

@pytest.fixture
def plan01(sim01):
    fpath = sim01 + r'/clearWaterTestCases.p01.hdf'
    diff_coef = 0.1
    intl_cond = sim01 + r'/cwr_initial_conditions_p01.csv'
    bndry_cond = sim01 + r'/cwr_boundary_conditions_p01.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan01(plan01):
    massBal_df = _mass_bal_global(plan01)
    massBal_df_100 = _mass_bal_global_100_Ans(plan01)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


