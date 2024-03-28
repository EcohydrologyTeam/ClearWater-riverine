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
    vol_start_sum = vol_start.sum()
    mass_start_sum = mass_start.sum()
    vol_start_sum_val = vol_start_sum.values
    mass_start_sum_val = mass_start_sum.values
    vol_start_sum_val_np = np.array([vol_start_sum_val])
    mass_start_sum_val_np = np.array([mass_start_sum_val])

    #Find Mass at the end of simulation
    t_max_index = len(simulation.mesh.time) - 1
    vol_end = simulation.mesh.volume[t_max_index][0:nreal_index]
    conc_end = simulation.mesh.concentration[t_max_index][0:nreal_index]
    mass_end = vol_end * conc_end
    vol_end_sum = vol_end.sum()
    mass_end_sum = mass_end.sum()
    vol_end_sum_val = vol_end_sum.values
    mass_end_sum_val = mass_end_sum.values
    vol_end_sum_val_np = np.array([vol_end_sum_val])
    mass_end_sum_val_np = np.array([mass_end_sum_val])
    
    #Construct dataframe to be returned
    d = {'Vol_start': vol_start_sum_val_np,
         'Mass_start': mass_start_sum_val_np,
         'Vol_end': vol_end_sum_val_np,
         'Mass_end': mass_end_sum_val_np}
    df = pd.DataFrame(data=d)
    
    #Loop to find total mass in/out from all boundary conditions
    bndryData = simulation.boundary_data
    bcLineIDs = bndryData.groupby(by='Name').mean(numeric_only=True)
    bcLineIDs_sorted = bcLineIDs.sort_values(by=['BC Line ID']).reset_index()
    
    bcTotalVolInOutAll = [0]
    bcTotalVolInAll = [0]
    bcTotalVolOutAll = [0]
    
    bcTotalMassInOutAll = [0]
    bcTotalMassInAll = [0]
    bcTotalMassOutAll = [0]
    
    for index, row in bcLineIDs_sorted.iterrows():
        #Total Flow/Mass from boundary condition
        bc_name = row['Name']
        bc_id = row['BC Line ID']
        bndryData_n = bndryData.loc[bndryData['BC Line ID'] == bc_id]
        bndryData_n_Face_df = bndryData_n[['Face Index']]
        bndryData_n_Face_arr = bndryData_n_Face_df.to_numpy()
        bndryData_n_Face_arrF = bndryData_n_Face_arr.flatten()
        
        bc_edgeFlow_xda = simulation.mesh.face_flow.sel(nedge=bndryData_n_Face_arrF)
        bc_edgeVol_xda = bc_edgeFlow_xda * simulation.mesh[variables.CHANGE_IN_TIME]
        bc_totalVol_xda = bc_edgeVol_xda.sum()
        bc_totalVol_xda_val = bc_totalVol_xda.values
        bc_totalVol_xda_val_np = np.array([bc_totalVol_xda_val])
        bc_name_vol = bc_name + '_vol'
        df[bc_name_vol] = bc_totalVol_xda_val_np
        bcTotalVolInOutAll = bcTotalVolInOutAll + bc_totalVol_xda_val_np
        
        bc_edgeMass_xda = simulation.mesh.mass_flux_total.sel(nedge=bndryData_n_Face_arrF)
        bc_totalMass_xda = bc_edgeMass_xda.sum()
        bc_totalMass_xda_val = bc_totalMass_xda.values
        bc_totalMass_xda_val_np = np.array([bc_totalMass_xda_val])
        bc_name_mass = bc_name + '_mass'
        df[bc_name_mass] = bc_totalMass_xda_val_np
        bcTotalMassInOutAll = bcTotalMassInOutAll + bc_totalMass_xda_val_np
        
        #Flow/Mass into domain form boundary condition
        bc_edgeVol_xda_in = bc_edgeVol_xda.where(bc_edgeVol_xda<=0, other=0)
        bc_totalVol_xda_in = bc_edgeVol_xda_in.sum()
        bc_totalVol_xda_val_in = bc_totalVol_xda_in.values
        bc_totalVol_xda_val_np_in = np.array([bc_totalVol_xda_val_in])
        bc_name_in_vol = bc_name + '_in_vol'
        df[bc_name_in_vol] = bc_totalVol_xda_val_np_in
        bcTotalVolInAll = bcTotalVolInAll + bc_totalVol_xda_val_np_in
        
        bc_edgeMass_xda_in = bc_edgeMass_xda.where(bc_edgeMass_xda<=0, other=0)
        bc_totalMass_xda_in = bc_edgeMass_xda_in.sum()
        bc_totalMass_xda_val_in = bc_totalMass_xda_in.values
        bc_totalMass_xda_val_np_in = np.array([bc_totalMass_xda_val_in])
        bc_name_in_mass = bc_name + '_in_mass'
        df[bc_name_in_mass] = bc_totalMass_xda_val_np_in
        bcTotalMassInAll = bcTotalMassInAll + bc_totalMass_xda_val_np_in
        
        #Flow/Mass out of domain form boundary condition
        bc_edgeVol_xda_out = bc_edgeVol_xda.where(bc_edgeVol_xda>=0, other=0)
        bc_totalVol_xda_out = bc_edgeVol_xda_out.sum()
        bc_totalVol_xda_val_out = bc_totalVol_xda_out.values
        bc_totalVol_xda_val_np_out = np.array([bc_totalVol_xda_val_out])
        bc_name_out_vol = bc_name + '_out_vol'
        df[bc_name_out_vol] = bc_totalVol_xda_val_np_out
        bcTotalVolOutAll = bcTotalVolOutAll + bc_totalVol_xda_val_np_out
        
        bc_edgeMass_xda_out = bc_edgeMass_xda.where(bc_edgeMass_xda>=0, other=0)
        bc_totalMass_xda_out = bc_edgeMass_xda_out.sum()
        bc_totalMass_xda_val_out = bc_totalMass_xda_out.values
        bc_totalMass_xda_val_np_out = np.array([bc_totalMass_xda_val_out])
        bc_name_out = bc_name + '_out'
        df[bc_name_out] = bc_totalMass_xda_val_np_out
        bcTotalMassOutAll = bcTotalMassOutAll + bc_totalMass_xda_val_np_out
         
    df['bcTotalVolInOutAll'] = bcTotalVolInOutAll
    df['bcTotalVolInAll'] = bcTotalVolInAll
    df['bcTotalVolOutAll'] = bcTotalVolOutAll

    df['bcTotalMassInOutAll'] = bcTotalMassInOutAll
    df['bcTotalMassInAll'] = bcTotalMassInAll
    df['bcTotalMassOutAll'] = bcTotalMassOutAll

    vol_end_calc = vol_start_sum_val_np + -1*bcTotalVolInAll + -1*bcTotalVolOutAll
    mass_end_calc = mass_start_sum_val_np + -1*bcTotalMassInAll + -1*bcTotalMassOutAll

    df['vol_end_calc'] = vol_end_calc
    df['error_vol'] = vol_end_calc - vol_end_sum_val_np
    df['prct_error_vol'] = ((vol_end_calc - vol_end_sum_val_np) / bcTotalVolInAll) * 100
    
    df['mass_end_calc'] = mass_end_calc
    df['error_mass'] = mass_end_calc - mass_end_sum_val_np
    df['prct_error_mass'] = ((mass_end_calc - mass_end_sum_val_np) / bcTotalMassInAll) * 100
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
    vol_start_sum = vol_start.sum()
    mass_start_sum = mass_start.sum()
    vol_start_sum_val = vol_start_sum.values
    mass_start_sum_val = mass_start_sum.values
    vol_start_sum_val_np = np.array([vol_start_sum_val])
    mass_start_sum_val_np = np.array([mass_start_sum_val])

    #Find Mass at the end of simulation
    t_max_index = len(simulation.mesh.time) - 1
    vol_end = simulation.mesh.volume[t_max_index][0:nreal_index]
    conc_end = simulation.mesh.concentration[t_max_index][0:nreal_index]
    conc_end_100 = conc_end.copy(deep=True)
    conc_end_100 = conc_end_100.where(conc_end_100==100, other=100)
    mass_end = vol_end * conc_end_100
    vol_end_sum = vol_end.sum()
    mass_end_sum = mass_end.sum()
    vol_end_sum_val = vol_end_sum.values
    mass_end_sum_val = mass_end_sum.values
    vol_end_sum_val_np = np.array([vol_end_sum_val])
    mass_end_sum_val_np = np.array([mass_end_sum_val])
    
    #Construct dataframe to be returned
    d = {'Vol_start': vol_start_sum_val_np,
         'Mass_start': mass_start_sum_val_np,
         'Vol_end': vol_end_sum_val_np,
         'Mass_end': mass_end_sum_val_np}
    df = pd.DataFrame(data=d)
    
    
    #Loop to find total mass in/out from all boundary conditions
    bndryData = simulation.boundary_data
    bcLineIDs = bndryData.groupby(by='Name').mean(numeric_only=True)
    bcLineIDs_sorted = bcLineIDs.sort_values(by=['BC Line ID']).reset_index()

    bcTotalVolInOutAll = [0]
    bcTotalVolInAll = [0]
    bcTotalVolOutAll = [0]

    bcTotalMassInOutAll = [0]
    bcTotalMassInAll = [0]
    bcTotalMassOutAll = [0]
    
    for index, row in bcLineIDs_sorted.iterrows():
        #Total Mass from boundary condition
        bc_name = row['Name']
        bc_id = row['BC Line ID']
        bndryData_n = bndryData.loc[bndryData['BC Line ID'] == bc_id]
        bndryData_n_Face_df = bndryData_n[['Face Index']]
        bndryData_n_Face_arr = bndryData_n_Face_df.to_numpy()
        bndryData_n_Face_arrF = bndryData_n_Face_arr.flatten()
        bc_edgeFlow_xda = simulation.mesh.face_flow.sel(nedge=bndryData_n_Face_arrF)
        bc_edgeVol_xda = bc_edgeFlow_xda * simulation.mesh[variables.CHANGE_IN_TIME]
        bc_totalVol_xda = bc_edgeVol_xda.sum()
        bc_totalVol_xda_val = bc_totalVol_xda.values
        bc_totalVol_xda_val_np = np.array([bc_totalVol_xda_val])
        bc_name_vol = bc_name + '_vol'
        df[bc_name_vol] = bc_totalVol_xda_val_np
        bcTotalVolInOutAll = bcTotalVolInOutAll + bc_totalVol_xda_val_np
        
        bc_edgeConc100_xda = bc_edgeFlow_xda.copy(deep=True)
        bc_edgeConc100_xda = bc_edgeConc100_xda.where(bc_edgeConc100_xda==100, other=100)
        bc_edgeMass_xda = bc_edgeFlow_xda * simulation.mesh[variables.CHANGE_IN_TIME] * bc_edgeConc100_xda
        bc_totalMass_xda = bc_edgeMass_xda.sum()
        bc_totalMass_xda_val = bc_totalMass_xda.values
        bc_totalMass_xda_val_np = np.array([bc_totalMass_xda_val])
        bc_name_mass = bc_name + '_mass'
        df[bc_name_mass] = bc_totalMass_xda_val_np
        bcTotalMassInOutAll = bcTotalMassInOutAll + bc_totalMass_xda_val_np
        
        #Flow/Mass into domain form boundary condition
        bc_edgeVol_xda_in = bc_edgeVol_xda.where(bc_edgeVol_xda<=0, other=0)
        bc_totalVol_xda_in = bc_edgeVol_xda_in.sum()
        bc_totalVol_xda_val_in = bc_totalVol_xda_in.values
        bc_totalVol_xda_val_np_in = np.array([bc_totalVol_xda_val_in])
        bc_name_in_vol = bc_name + '_in_vol'
        df[bc_name_in_vol] = bc_totalVol_xda_val_np_in
        bcTotalVolInAll = bcTotalVolInAll + bc_totalVol_xda_val_np_in
        
        bc_edgeMass_xda_in = bc_edgeMass_xda.where(bc_edgeMass_xda<=0, other=0)
        bc_totalMass_xda_in = bc_edgeMass_xda_in.sum()
        bc_totalMass_xda_val_in = bc_totalMass_xda_in.values
        bc_totalMass_xda_val_np_in = np.array([bc_totalMass_xda_val_in])
        bc_name_in_mass = bc_name + '_in_mass'
        df[bc_name_in_mass] = bc_totalMass_xda_val_np_in
        bcTotalMassInAll = bcTotalMassInAll + bc_totalMass_xda_val_np_in
        
        #Flow/Mass out of domain form boundary condition
        bc_edgeVol_xda_out = bc_edgeVol_xda.where(bc_edgeVol_xda>=0, other=0)
        bc_totalVol_xda_out = bc_edgeVol_xda_out.sum()
        bc_totalVol_xda_val_out = bc_totalVol_xda_out.values
        bc_totalVol_xda_val_np_out = np.array([bc_totalVol_xda_val_out])
        bc_name_out_vol = bc_name + '_out_vol'
        df[bc_name_out_vol] = bc_totalVol_xda_val_np_out
        bcTotalVolOutAll = bcTotalVolOutAll + bc_totalVol_xda_val_np_out
        
        bc_edgeMass_xda_out = bc_edgeMass_xda.where(bc_edgeMass_xda>=0, other=0)
        bc_totalMass_xda_out = bc_edgeMass_xda_out.sum()
        bc_totalMass_xda_val_out = bc_totalMass_xda_out.values
        bc_totalMass_xda_val_np_out = np.array([bc_totalMass_xda_val_out])
        bc_name_out = bc_name + '_out'
        df[bc_name_out] = bc_totalMass_xda_val_np_out
        bcTotalMassOutAll = bcTotalMassOutAll + bc_totalMass_xda_val_np_out
         
    df['bcTotalVolInOutAll'] = bcTotalVolInOutAll
    df['bcTotalVolInAll'] = bcTotalVolInAll
    df['bcTotalVolOutAll'] = bcTotalVolOutAll

    df['bcTotalMassInOutAll'] = bcTotalMassInOutAll
    df['bcTotalMassInAll'] = bcTotalMassInAll
    df['bcTotalMassOutAll'] = bcTotalMassOutAll

    vol_end_calc = vol_start_sum_val_np + -1*bcTotalVolInAll + -1*bcTotalVolOutAll
    mass_end_calc = mass_start_sum_val_np + -1*bcTotalMassInAll + -1*bcTotalMassOutAll

    df['vol_end_calc'] = vol_end_calc
    df['error_vol'] = vol_end_calc - vol_end_sum_val_np
    df['prct_error_vol'] = ((vol_end_calc - vol_end_sum_val_np) / bcTotalVolInAll) * 100
    
    df['mass_end_calc'] = mass_end_calc
    df['error_mass'] = mass_end_calc - mass_end_sum_val_np
    df['prct_error_mass'] = ((mass_end_calc - mass_end_sum_val_np) / bcTotalMassInAll) * 100
    return df


def _mass_bal_val(df, col) -> float:
    mass = df[col].values[0]
    return mass