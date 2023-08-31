from clearwater_riverine.postproc_util import _run_simulation
from clearwater_riverine.postproc_util import _mass_bal_global
from clearwater_riverine.postproc_util import _mass_bal_global_100_Ans
from clearwater_riverine.postproc_util import _mass_bal_val


def test_mass_end_plan02():
    """NOTE: Relative paths below are referenced from the root directory of the repo
             i.e. please run pytest from only at the ./ClearWater-riverine directory
             level or pytest will give the "FileNotFoundError" message.
    """
    fpath = './examples/data/simple_test_cases/plan02_2x1/clearWaterTestCases.p02.hdf'
    diff_coef = 0.1
    intl_cond = './examples/data/simple_test_cases/plan02_2x1/cwr_initial_conditions_p02.csv'
    bndry_cond = './examples/data/simple_test_cases/plan02_2x1/cwr_boundary_conditions_p02.csv'
    plan02 = _run_simulation(fpath, diff_coef, intl_cond, bndry_cond)
    massBal_df = _mass_bal_global(plan02)
    massBal_df_100 = _mass_bal_global_100_Ans(plan02)
    answer = _mass_bal_val(massBal_df_100, 'Mass_end')
    assert _mass_bal_val(massBal_df, 'Mass_end') == answer 