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
    diff_coef = 0.01
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
    diff_coef = 0.01
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
    diff_coef = 0.01
    intl_cond = sim01 + r'/cwr_initial_conditions_p01.csv'
    bndry_cond = sim01 + r'/cwr_boundary_conditions_p01.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan01(plan01):
    massBal_df = _mass_bal_global(plan01)
    massBal_df_100 = _mass_bal_global_100_Ans(plan01)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


@pytest.fixture
def sim04(data_dir) -> str:
    return data_dir + r'/plan04_10x5_fullBndry'

@pytest.fixture
def plan04(sim04):
    fpath = sim04 + r'/clearWaterTestCases.p04.hdf'
    diff_coef = 0.01
    intl_cond = sim04 + r'/cwr_initial_conditions_p04.csv'
    bndry_cond = sim04 + r'/cwr_boundary_conditions_p04.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan04(plan04):
    massBal_df = _mass_bal_global(plan04)
    massBal_df_100 = _mass_bal_global_100_Ans(plan04)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


@pytest.fixture
def sim05(data_dir) -> str:
    return data_dir + r'/plan05_10x5_tidal_fullBndry'

@pytest.fixture
def plan05(sim05):
    fpath = sim05 + r'/clearWaterTestCases.p05.hdf'
    diff_coef = 0.01
    intl_cond = sim05 + r'/cwr_initial_conditions_p05.csv'
    bndry_cond = sim05 + r'/cwr_boundary_conditions_p05.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan05(plan05):
    massBal_df = _mass_bal_global(plan05)
    massBal_df_100 = _mass_bal_global_100_Ans(plan05)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


@pytest.fixture
def sim06(data_dir) -> str:
    return data_dir + r'/plan06_10x5_tidal_multiBndry'

@pytest.fixture
def plan06(sim06):
    fpath = sim06 + r'/clearWaterTestCases.p06.hdf'
    diff_coef = 0.01
    intl_cond = sim06 + r'/cwr_initial_conditions_p06.csv'
    bndry_cond = sim06 + r'/cwr_boundary_conditions_p06.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan06(plan06):
    massBal_df = _mass_bal_global(plan06)
    massBal_df_100 = _mass_bal_global_100_Ans(plan06)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


@pytest.fixture
def sim07(data_dir) -> str:
    return data_dir + r'/plan07_10x5_tidal_multiBndry_isle'

@pytest.fixture
def plan07(sim07):
    fpath = sim07 + r'/clearWaterTestCases.p07.hdf'
    diff_coef = 0.01
    intl_cond = sim07 + r'/cwr_initial_conditions_p07.csv'
    bndry_cond = sim07 + r'/cwr_boundary_conditions_p07.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan07(plan07):
    massBal_df = _mass_bal_global(plan07)
    massBal_df_100 = _mass_bal_global_100_Ans(plan07)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer


@pytest.fixture
def sim08(data_dir) -> str:
    return data_dir + r'/plan08_10x5Rf_tidal_multiBndry_isle'

@pytest.fixture
def plan08(sim08):
    fpath = sim08 + r'/clearWaterTestCases.p08.hdf'
    diff_coef = 0.01
    intl_cond = sim08 + r'/cwr_initial_conditions_p08.csv'
    bndry_cond = sim08 + r'/cwr_boundary_conditions_p08.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan08(plan08):
    massBal_df = _mass_bal_global(plan08)
    massBal_df_100 = _mass_bal_global_100_Ans(plan08)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer



#The following tests are for the "Sumwere Creek" example model
@pytest.fixture
def data_dir2() -> str:
    return './tests/data/sumwere_test_cases'

@pytest.fixture
def sim11(data_dir2) -> str:
    return data_dir2 + r'/plan11_stormSurge'

@pytest.fixture
def plan11(sim11):
    fpath = sim11 + r'/clearWaterTestCases.p11.hdf'
    diff_coef = 0.01
    intl_cond = sim11 + r'/cwr_initial_conditions_p11.csv'
    bndry_cond = sim11 + r'/cwr_boundary_conditions_p11.csv'
    return _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)

def test_mass_end_plan11(plan11):
    massBal_df = _mass_bal_global(plan11)
    massBal_df_100 = _mass_bal_global_100_Ans(plan11)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer
