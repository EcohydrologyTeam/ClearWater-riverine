#from pathlib import Path

import numpy as np
import pandas as pd

import clearwater_riverine as cwr
from clearwater_riverine import variables

#import pytest



#def _run_simulation(ras_hdf, diff_coef, intl_cnd, bndry) -> crw.ClearwaterRiverine:
def _run_simulation(ras_hdf, diff_coef, intl_cnd, bndry):
    """Returns a Clearwater Riverine Simulation object that has water quality results"""
    fpath = ras_hdf
    simulation = cwr.ClearwaterRiverine(fpath, diff_coef, verbose=False)
    simulation.initial_conditions(intl_cnd)
    simulation.boundary_conditions(bndry)
    simulation.simulate_wq()
    return simulation


def _mass_bal_global(simulation) -> pd.DataFrame:
    """Returns entire domain and overall simulation period mass balance values"""
    
    #Find Mass at the start of simulation
    nreal_index = simulation.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] + 1
    vol_start = simulation.mesh.volume[0][0:nreal_index]
    conc_start = simulation.mesh.concentration[0][0:nreal_index]
    mass_start = vol_start * conc_start
    mass_start_sum = mass_start.sum()
    mass_start_sum_val = mass_start_sum.values
    mass_start_sum_val_np = np.array([mass_start_sum_val])

    #Find Mass at the end of simulation
    t_max_index = len(simulation.mesh.time) - 2
    vol_end = simulation.mesh.volume[t_max_index][0:nreal_index]
    conc_end = simulation.mesh.concentration[t_max_index][0:nreal_index]
    mass_end = vol_end * conc_end
    mass_end_sum = mass_end.sum()
    mass_end_sum_val = mass_end_sum.values
    mass_end_sum_val_np = np.array([mass_end_sum_val])
    
    #Construct dataframe to be returned
    d = {'Mass_start':mass_start_sum_val_np, 'Mass_end':mass_end_sum_val_np}
    df = pd.DataFrame(data=d)
    return df




def _mass_bal_global_100_Ans(simulation) -> pd.DataFrame:
    """Returns entire domain and overall simulation period mass balance values
       assuming intial conditions are 100 mg/L everywhere and any boundary
       conditions inputs are also 100mg/L
    """
    
    #Find Mass at the start of simulation
    nreal_index = simulation.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] + 1
    vol_start = simulation.mesh.volume[0][0:nreal_index]
    conc_start = simulation.mesh.concentration[0][0:nreal_index]
    conc_start_100 = conc_start.copy(deep=True)
    conc_start_100 = conc_start_100.where(conc_start_100==100, other=100)
    mass_start = vol_start * conc_start_100
    mass_start_sum = mass_start.sum()
    mass_start_sum_val = mass_start_sum.values
    mass_start_sum_val_np = np.array([mass_start_sum_val])

    #Find Mass at the end of simulation
    t_max_index = len(simulation.mesh.time) - 2
    vol_end = simulation.mesh.volume[t_max_index][0:nreal_index]
    conc_end = simulation.mesh.concentration[t_max_index][0:nreal_index]
    conc_end_100 = conc_end.copy(deep=True)
    conc_end_100 = conc_end_100.where(conc_end_100==100, other=100)
    mass_end = vol_end * conc_end_100
    mass_end_sum = mass_end.sum()
    mass_end_sum_val = mass_end_sum.values
    mass_end_sum_val_np = np.array([mass_end_sum_val])
    
    #Construct dataframe to be returned
    d = {'Mass_start':mass_start_sum_val_np, 'Mass_end':mass_end_sum_val_np}
    df = pd.DataFrame(data=d)
    return df


def _mass_bal_val(df, col) -> float:
    mass = df[col].values[0]
    return mass