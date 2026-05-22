# Phase G-1 (2026-05-21): ``from __future__ import annotations`` defers
# annotation evaluation so the ``cwr.ClearwaterRiverine`` annotations
# below do not resolve at import time. Without this, importing this
# module while ``clearwater_riverine/__init__.py`` is mid-execution
# (the ``from .model import ClearwaterRiverine`` line transitively
# triggers ``from .postproc_util import ...``) hits a partially-
# initialized package and raises ``AttributeError``. Annotations are
# only used as type hints; this is the cheapest fix.
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import clearwater_riverine as cwr
from clearwater_riverine.variables import(
    BOUNDARY_FACE_INDEX,
    BOUNDARY_NAME,
    CHANGE_IN_TIME,
    EDGE_FACE_CONNECTIVITY,
    FLOW_ACROSS_FACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)
from clearwater_data.variables import VariableRegistry


def accumulate_chunk_mass_balance(
        acc: Optional[dict],
        registry: VariableRegistry,
        constituent_name: str,
        start_datetime: datetime,
        end_datetime: datetime,
        drop_last_slot: bool,
        is_last: bool,
    ) -> dict:
    """Fold one chunk's contribution into the cross-chunk accumulator (C3a).

    Mirrors ``_calculate_boundary_data``'s exact reductions, applied to the
    chunk-resident registry and summed across chunks. Chunk windows overlap
    by one slot (chunk k's last slot == chunk k+1's first slot), so for the
    per-slot FLOW the shared trailing slot is dropped on every chunk except
    the final one (``drop_last_slot``) -- the next chunk owns it as its
    first slot. ``{constituent}_mass_flux`` is per-transition (length T-1)
    and already partitions exactly across chunks, so it is summed in full
    every chunk. The whole-run union therefore counts each timestep and each
    transition exactly once, reproducing the non-chunked totals.

    Domain start snapshot is captured the first time a constituent is seen
    (chunk 1 is resident and contains ``start_datetime``); the end snapshot
    is captured on the final chunk (``is_last``; it contains
    ``end_datetime``).
    """
    if acc is None:
        acc = {}
    c = acc.setdefault(
        constituent_name, {"boundaries": {}, "start": None, "end": None}
    )
    nreal = registry.get_variable(NUMBER_OF_REAL_CELLS).get_data()

    if c["start"] is None:
        v = registry.get_at_time(VOLUME, start_datetime)[0:nreal]
        x = registry.get_at_time(constituent_name, start_datetime)[0:nreal]
        c["start"] = {
            "total_volume": float(v.sum().item()),
            "total_mass": float((v * x).sum().item()),
        }
    if is_last:
        v = registry.get_at_time(VOLUME, end_datetime)[0:nreal]
        x = registry.get_at_time(constituent_name, end_datetime)[0:nreal]
        c["end"] = {
            "total_volume": float(v.sum().item()),
            "total_mass": float((v * x).sum().item()),
        }

    boundary_index = registry.get_variable(BOUNDARY_FACE_INDEX).get_data()
    boundary_names = registry.get_variable(BOUNDARY_NAME).get_data()
    flow = registry.get_variable(FLOW_ACROSS_FACE).get_data()
    if drop_last_slot:
        flow = flow.isel(time=slice(0, -1))
    dt = registry.get_variable(CHANGE_IN_TIME).get_data()
    mass_flux = registry.get_variable(f"{constituent_name}_mass_flux").get_data()

    for boundary_name in np.unique(boundary_names):
        edges_mask = boundary_names == boundary_name
        boundary_edges = boundary_index[edges_mask]
        boundary_volume = flow.sel(nedge=boundary_edges) * dt
        boundary_mass = mass_flux.sel(nedge=boundary_edges)
        m = c["boundaries"].setdefault(
            boundary_name,
            {k: 0.0 for k in (
                "volume_total", "mass_total", "volume_in",
                "mass_in", "volume_out", "mass_out",
            )},
        )
        m["volume_total"] += boundary_volume.sum().item()
        m["mass_total"] += boundary_mass.sum().item()
        m["volume_in"] += boundary_volume.where(boundary_volume < 0).sum().item()
        m["mass_in"] += boundary_mass.where(boundary_mass < 0).sum().item()
        m["volume_out"] += boundary_volume.where(boundary_volume > 0).sum().item()
        m["mass_out"] += boundary_mass.where(boundary_mass > 0).sum().item()
    return acc


def _global_mass_balance_from_accumulator(
        chunk_accumulator: dict,
        constituent_name: str,
        calculate_answer: bool,
        answer_value: Optional[float],
    ):
    """Chunked global mass balance from the cross-chunk accumulator (C3a).

    Reconstructs exactly the structures the non-chunked path feeds to
    ``_calculate_error_metrics`` (which is reused unchanged), so the chunked
    closure number is computed by the identical error math. The answer-case
    boundary mass mirrors the non-chunked ``boundary_volume * answer_value``
    (answer_value > 0 preserves sign, so the in/out split matches).
    """
    ca = chunk_accumulator[constituent_name]
    total_volume_start = ca["start"]["total_volume"]
    total_volume_end = ca["end"]["total_volume"]
    if calculate_answer:
        total_mass_start = total_volume_start * answer_value
        total_mass_end = total_volume_end * answer_value
    else:
        total_mass_start = ca["start"]["total_mass"]
        total_mass_end = ca["end"]["total_mass"]

    df = pd.DataFrame(data={
        'volume_start': [total_volume_start],
        'mass_start': [total_mass_start],
        'volume_end_modeled': [total_volume_end],
        'mass_end_modeled': [total_mass_end],
    })

    records = []
    for boundary_name, m in ca["boundaries"].items():
        if calculate_answer:
            mass_total = m["volume_total"] * answer_value
            mass_in = m["volume_in"] * answer_value
            mass_out = m["volume_out"] * answer_value
        else:
            mass_total, mass_in, mass_out = (
                m["mass_total"], m["mass_in"], m["mass_out"]
            )
        records.append({
            'boundary_name': boundary_name,
            'volume_total': m["volume_total"],
            'mass_total': mass_total,
            'volume_in': m["volume_in"],
            'mass_in': mass_in,
            'volume_out': m["volume_out"],
            'mass_out': mass_out,
        })

    boundary_df = pd.DataFrame.from_records(records).set_index("boundary_name")
    total = boundary_df.sum()
    total["boundary_name"] = "all_boundaries"
    boundary_df = pd.concat(
        [boundary_df.reset_index(), pd.DataFrame([total])], ignore_index=True
    ).set_index("boundary_name")

    _calculate_error_metrics(df=df, boundary_df=boundary_df, variable_name="volume")
    _calculate_error_metrics(df=df, boundary_df=boundary_df, variable_name="mass")
    return {'global': df, 'boundary': boundary_df}


def calculate_global_mass_balance(
        registry: VariableRegistry,
        constituent_name: str,
        start_datetime: datetime,
        end_datetime: datetime,
        calculate_answer: bool,
        answer_value: Optional[float],
        chunk_accumulator: Optional[dict] = None,
    ):
    # Chunked runs: the registry only holds the final chunk post-run, so the
    # whole-run balance is reconstructed from the cross-chunk accumulator
    # (C3a). The non-chunked path below is unchanged.
    if chunk_accumulator is not None:
        return _global_mass_balance_from_accumulator(
            chunk_accumulator, constituent_name, calculate_answer, answer_value
        )

    ## TODO: logic for if run without mass flux calcs.
    if f"{constituent_name}_mass_flux" not in registry:
        raise ValueError("Rerun model with `mass_flux_calculation` set to True.")

    # calculate starting mass
    real_cell_count = registry.get_variable(NUMBER_OF_REAL_CELLS).get_data()
    volume_start = registry.get_at_time(VOLUME, start_datetime)[0:real_cell_count]
    volume_end = registry.get_at_time(VOLUME, end_datetime)[0:real_cell_count]
    if calculate_answer:
        concentration_start = answer_value
        concentration_end = answer_value
    else:
        concentration_start = registry.get_at_time(constituent_name, start_datetime)[0:real_cell_count]
        concentration_end = registry.get_at_time(constituent_name, end_datetime)[0:real_cell_count]
    mass_start = volume_start * concentration_start
    mass_end = volume_end * concentration_end

    # aggregate volume and mass over entire domain
    total_volume_start = volume_start.sum().item()
    total_volume_end = volume_end.sum().item()
    total_mass_start = mass_start.sum().item()
    total_mass_end  = mass_end.sum().item()

    # construct dataframe to be returned
    d = {
        'volume_start': [total_volume_start],
        'mass_start': [total_mass_start],
        'volume_end_modeled': [total_volume_end],
        'mass_end_modeled': [total_mass_end]
    }
    df = pd.DataFrame(data=d)
    boundary_df = _calculate_boundary_data(registry, constituent_name, calculate_answer, answer_value)

    _calculate_error_metrics(df=df, boundary_df=boundary_df, variable_name="volume")
    _calculate_error_metrics(df=df, boundary_df=boundary_df, variable_name="mass")
    return {
        'global': df,
        'boundary': boundary_df,
    }


def _calculate_boundary_data(
    registry: VariableRegistry,
    constituent_name: str,
    calculate_answer: bool,
    answer_value: float,
):
    # get boundary information
    boundary_index = registry.get_variable(BOUNDARY_FACE_INDEX).get_data()
    boundary_names = registry.get_variable(BOUNDARY_NAME).get_data()

    records = [] 
    for boundary_name in np.unique(boundary_names):
        # determine which edges are associated with the boundary
        edges_mask = boundary_names == boundary_name
        boundary_edges = boundary_index[edges_mask]

        # get flow, volume, and mass
        boundary_flow = registry.get_variable(FLOW_ACROSS_FACE).get_data().sel(nedge=boundary_edges)
        boundary_volume = boundary_flow * registry.get_variable(CHANGE_IN_TIME).get_data()
        if calculate_answer == True:
            boundary_mass = boundary_volume * answer_value
        else:
            boundary_mass = registry.get_variable(f"{constituent_name}_mass_flux").get_data().sel(nedge=boundary_edges)
    
        # calculate totals, 
        metrics = {
            'boundary_name': boundary_name,
            "volume_total": boundary_volume.sum().item(),
            "mass_total": boundary_mass.sum().item(),
            "volume_in": boundary_volume.where(boundary_volume < 0).sum().item(),
            "mass_in": boundary_mass.where(boundary_mass < 0).sum().item(),
            "volume_out": boundary_volume.where(boundary_volume > 0).sum().item(),
            "mass_out": boundary_mass.where(boundary_mass > 0).sum().item(),
        }
        records.append(metrics)

    # add total across all boundaries 
    boundary_df = pd.DataFrame.from_records(records).set_index("boundary_name")
    total = boundary_df.sum()
    total["boundary_name"] = "all_boundaries"

    # append to the boundary_df
    boundary_df_with_total = pd.concat([boundary_df.reset_index(), pd.DataFrame([total])], ignore_index=True)
    boundary_df_with_total.set_index("boundary_name", inplace=True)
    return boundary_df_with_total

def _calculate_error_metrics(
        df,
        boundary_df,
        variable_name,
    ):
    df[f'{variable_name}_end_calculated'] = df[f'{variable_name}_start'] + \
        boundary_df.loc["all_boundaries", f"{variable_name}_in"] * -1 + \
        boundary_df.loc["all_boundaries", f"{variable_name}_out"] * -1

    df[f"{variable_name}_error"] = _total_error(
        df[f"{variable_name}_end_calculated"],
        df[f"{variable_name}_end_modeled"]
    )

    df[f"{variable_name}_percent_error"] = _percent_error(
        df[f"{variable_name}_end_calculated"],
        df[f"{variable_name}_end_modeled"],
        boundary_df.loc["all_boundaries", f"{variable_name}_in"]
    )


def _total_error(
    calculated_end_result,
    modeled_end_result,
):
    return calculated_end_result - modeled_end_result


def _percent_error(
    calculated_end_result,
    modeled_end_result,
    boundary_inflow,
):
    return ((calculated_end_result - modeled_end_result) / boundary_inflow) * 100











def _calculate_mass_flux(
    registry: VariableRegistry,
):
    negative_condition = registry.get_variable(FLOW_ACROSS_FACE).get_data()

def _run_simulation(ras_hdf, diff_coef, intl_cnd, bndry):
    """Returns a Clearwater Riverine Simulation object that has water quality results"""
    fpath = ras_hdf
    simulation = cwr.ClearwaterRiverine(fpath, diff_coef, verbose=False)
    simulation.initial_conditions(intl_cnd)
    simulation.boundary_conditions(bndry)
    simulation.simulate_wq()
    return simulation


def _mass_bal_global(
    simulation:cwr.ClearwaterRiverine,
    constituent_name: str,
    boundary_data_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Returns entire domain and overall simulation period mass balance values"""
    if _mass_flux_calculation_required(
        simulation=simulation,
        constituent_name=constituent_name,
    ):
        _calculate_mass_flux(
            simulation=simulation,
            constituent_name=constituent_name,
        )
    
    #Find Mass at the start of simulation
    nreal_index = simulation.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] + 1
    vol_start = simulation.mesh.volume[0][0:nreal_index]
    conc_start = simulation.mesh[constituent_name].data[0][0:nreal_index]
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
    conc_end = simulation.mesh[constituent_name].data[t_max_index][0:nreal_index]
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
    bndryData = _parse_boundary_data(
        simulation=simulation,
        boundary_data_path=boundary_data_path
    )
    bcLineIDs = bndryData.groupby(by='Name').mean(numeric_only=True)
    bcLineIDs_sorted = bcLineIDs.sort_values(by=['BC Line ID']).reset_index()
    
    bcTotalVolInOutAll = [0]
    bcTotalVolInAll = [0]
    bcTotalVolOutAll = [0]
    
    bcTotalMassInOutAll = [0]
    bcTotalMassInAll = [0]
    bcTotalMassOutAll = [0]
    
    for _, row in bcLineIDs_sorted.iterrows():
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
        
        bc_edgeMass_xda = simulation.constituent_dict[constituent_name].total_mass_flux[:, bndryData_n_Face_arrF]
        bc_totalMass_xda = bc_edgeMass_xda.sum()
        #bc_totalMass_xda_val = bc_totalMass_xda.values
        #bc_totalMass_xda_val_np = np.array([bc_totalMass_xda_val])
        bc_totalMass_xda_val_np = bc_totalMass_xda
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
        
        #bc_edgeMass_xda_in = bc_edgeMass_xda.where(bc_edgeMass_xda<=0, other=0)
        bc_edgeMass_xda_in = np.where(bc_edgeMass_xda<=0, bc_edgeMass_xda, bc_edgeMass_xda*0)
        bc_totalMass_xda_in = bc_edgeMass_xda_in.sum()
        #bc_totalMass_xda_val_in = bc_totalMass_xda_in.values
        #bc_totalMass_xda_val_np_in = np.array([bc_totalMass_xda_val_in])
        bc_totalMass_xda_val_np_in = bc_totalMass_xda_in
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
        
        #bc_edgeMass_xda_out = bc_edgeMass_xda.where(bc_edgeMass_xda>=0, other=0)
        bc_edgeMass_xda_out = np.where(bc_edgeMass_xda>=0, bc_edgeMass_xda, bc_edgeMass_xda*0)
        bc_totalMass_xda_out = bc_edgeMass_xda_out.sum()
        #bc_totalMass_xda_val_out = bc_totalMass_xda_out.values
        #bc_totalMass_xda_val_np_out = np.array([bc_totalMass_xda_val_out])
        bc_totalMass_xda_val_np_out = bc_totalMass_xda_out
        bc_name_out_mass = bc_name + '_out_mass'
        df[bc_name_out_mass] = bc_totalMass_xda_val_np_out
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


def _mass_bal_global_100_Ans(
    simulation:cwr.ClearwaterRiverine,
    constituent_name: str,
    boundary_data_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """Returns entire domain and overall simulation period mass balance values
       assuming intial conditions are 100 mg/L everywhere and any boundary
       conditions inputs are also 100mg/L
    """
    
    #Find Mass at the start of simulation
    nreal_index = simulation.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] + 1
    vol_start = simulation.mesh.volume[0][0:nreal_index]
    conc_start = simulation.mesh[constituent_name][0][0:nreal_index]
    conc_start_100 = xr.full_like(
        conc_start,
        fill_value=100,
        dtype=np.float64
    )
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
    conc_end = simulation.mesh[constituent_name][t_max_index][0:nreal_index]
    conc_end_100 = xr.full_like(
        conc_end,
        fill_value=100,
        dtype=np.float64
    )
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
    bndryData = _parse_boundary_data(
        simulation,
        boundary_data_path=boundary_data_path
    )
    bcLineIDs = bndryData.groupby(by='Name').mean(numeric_only=True)
    bcLineIDs_sorted = bcLineIDs.sort_values(by=['BC Line ID']).reset_index()

    bcTotalVolInOutAll = [0]
    bcTotalVolInAll = [0]
    bcTotalVolOutAll = [0]

    bcTotalMassInOutAll = [0]
    bcTotalMassInAll = [0]
    bcTotalMassOutAll = [0]
    
    for _, row in bcLineIDs_sorted.iterrows():
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
        bc_edgeConc100_xda = xr.full_like(
            bc_edgeFlow_xda,
            fill_value=100,
            dtype=np.float64
        )
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
        bc_name_out = bc_name + '_out_mass'
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

def _mass_flux_calculation_required(
    simulation: cwr.ClearwaterRiverine,
    constituent_name: str,
) -> bool:
    """Determines if mass flux calculation is needed."""
    base_condition =  np.zeros(
        (len(simulation.mesh.time),
        len(simulation.mesh.nedge))
    )
    constituent = simulation.constituent_dict[constituent_name]
    if np.array_equal(constituent.total_mass_flux, base_condition):
        return True
    else:
        return False

def _calculate_mass_flux(
    simulation: cwr.ClearwaterRiverine,
    constituent_name: str,
) -> None:
    """Calculates mass flux.
    Used when user has loaded in a model mesh.
    """
    print('Calculating mass fluxes...')
    constituent = simulation.constituent_dict[constituent_name]

    for t in range(len(simulation.mesh.time) - 1):
        simulation._mass_flux(
            simulation.mesh[constituent_name],
            constituent.advection_mass_flux,
            constituent.diffusion_mass_flux,
            constituent.total_mass_flux,
            t
        )
    print('Max flux calculations complete!')

def _parse_boundary_data(
    simulation: cwr.ClearwaterRiverine,
    boundary_data_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    try:
        boundary_data = simulation.boundary_data
    except AttributeError:
        if boundary_data_path:
            boundary_data = pd.read_csv(
                boundary_data_path
            )
        else:
            raise ValueError("boundary_data_path input required")
    return boundary_data