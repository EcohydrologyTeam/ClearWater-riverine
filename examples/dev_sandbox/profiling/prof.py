# memory profiling

# import packages
import sys
import numpy as np
import pandas as pd
import holoviews as hv
import xarray as xr


import clearwater_riverine as cwr
from clearwater_modules.tsm.model import EnergyBudget

# SET PATHS
root = r'C:\Users\sjordan\OneDrive - LimnoTech\Documents\GitHub\ClearWater-riverine\examples\dev_sandbox\data\sumwere_test_cases\plan28_testTSM'
ras_filepath = r'W:\2ERDC12 - Clearwater\ClearwaterHECRAS_testCases\sumwereCreek_TSM_testing_timestep\clearWaterTestCases.p37.hdf'
initial_condition_path = f'{root}/cwr_initial_conditions_waterTemp_p28.csv'
boundary_condition_path = r'W:\2ERDC12 - Clearwater\Clearwater_testing_TSM\plan28_testTSM_pwrPlnt_May2022\cwr_boundary_conditions_waterTemp_p37.csv'



def interpolate_to_model_timestep(meteo_df, model_dates, col_name):
    merged_df = pd.merge_asof(
        pd.DataFrame({'time': model_dates}),
        meteo_df,
        left_on='time',
        right_on='Datetime')
    merged_df[col_name.lower()] = merged_df[col_name].interpolate(method='linear')
    merged_df.drop(
        columns=['Datetime', col_name],
        inplace=True)
    merged_df.rename(
        columns={'time': 'Datetime'},
        inplace=True,
    )
    return merged_df

def run_n_timesteps_profiled(
    time_steps: int,
    reaction: EnergyBudget,
    transport: cwr.ClearwaterRiverine,
    meteo_params,
    concentration_update = None
):
    for i in range(1, time_steps):
        # update transport model using values with output from reaction model, if available
        transport.update(concentration_update)

        # increment the timestep of the reaction model, using meteo parameters + output from transport model
        updated_state_values = {
            'water_temp_c': transport.mesh['concentration'].isel(time=i, nface=slice(0,transport.mesh.nreal + 1)),
            'volume': transport.mesh['volume'].isel(time=i, nface = slice(0, transport.mesh.nreal + 1)),
            'surface_area': transport.mesh['faces_surface_area'].isel(nface = slice(0, transport.mesh.nreal + 1)),
            'q_solar': transport.mesh.concentration.isel(time=i, nface = slice(0, transport.mesh.nreal + 1)) * 0 + meteo_params['q_solar'][i],
            'air_temp_c': transport.mesh.concentration.isel(time=i, nface = slice(0, transport.mesh.nreal + 1)) * 0 + meteo_params['air_temp_c'][i],
        }
        reaction.increment_timestep(updated_state_values)
        ds = reaction.dataset.where(~np.isinf(reaction.dataset), 0)
        concentration_update = {"concentration": ds.water_temp_c.isel(seconds=i)}

def initialize_clearwater_riverine():
    transport_model = cwr.ClearwaterRiverine(
        ras_filepath,
        0.001,
        verbose=True
    )

    transport_model.initialize(
        initial_condition_path=initial_condition_path,
        boundary_condition_path=boundary_condition_path,
        units='degC',
    )   
    return transport_model

def define_meteo_params(transport_model):
    
    xarray_time_index = pd.DatetimeIndex(
        transport_model.mesh.time.values
    )

    q_solar = pd.read_csv(
        f'{root}/cwr_boundary_conditions_q_solar_p28.csv', 
        parse_dates=['Datetime'])
    q_solar.dropna(axis=0, inplace=True)

    q_solar_interp = interpolate_to_model_timestep(
        q_solar,
        xarray_time_index,
        'q_Solar'
    )

    air_temp_c = pd.read_csv(
        f'{root}/cwr_boundary_conditions_TairC_p28.csv', 
        parse_dates=['Datetime'])
    air_temp_c.dropna(axis=0, inplace=True)

    air_temp_c_interp = interpolate_to_model_timestep(
        air_temp_c,
        xarray_time_index,
        'TairC'
    )

    air_temp_c_interp['air_temp_c'] = (air_temp_c_interp.tairc - 32)* (5/9)


    q_solar_array = q_solar_interp.q_solar.to_numpy()
    air_temp_array = air_temp_c_interp.air_temp_c.to_numpy()


    # for each individual timestep
    all_meteo_params = {
        'q_solar': q_solar_array,
        'air_temp_c': air_temp_array,
    }


    # for initial conditions
    initial_meteo_params = {
        'air_temp_c': air_temp_array[0],
        'q_solar': q_solar_array[0],
    }
    return all_meteo_params, initial_meteo_params

def initialize_clearwater_modules(initial_state_values, initial_meteo_params):
    # updateable static variable = unexpected keyword argument
    reaction_model = EnergyBudget(
        initial_state_values,
        time_dim='seconds',
        meteo_parameters= initial_meteo_params,
        track_dynamic_variables = False,
        use_sed_temp = False,
        updateable_static_variables=['air_temp_c', 'q_solar']
    )
    return reaction_model


# Clearwater-Riverine
# Paths
def main(iters):

    transport_model = initialize_clearwater_riverine()

    initial_state_values = {
        'water_temp_c': transport_model.mesh['concentration'].isel(time=0, nface=slice(0, transport_model.mesh.nreal + 1)),
        'volume': transport_model.mesh['volume'].isel(time=0, nface=slice(0, transport_model.mesh.nreal + 1)),
        'surface_area': transport_model.mesh['faces_surface_area'].isel(nface=slice(0, transport_model.mesh.nreal + 1)),
    }


    all_meteo_params, initial_meteo_params = define_meteo_params(transport_model)

    reaction_model = initialize_clearwater_modules(initial_state_values, initial_meteo_params)

    run_n_timesteps_profiled(
        iters,
        reaction_model,
        transport_model,
        all_meteo_params,
    )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            iters = int(sys.argv[1])
            print(f'Running {iters} iterations.')
        except ValueError:
            raise ValueError('Argument must be an integer # of iterations.')
    else:
        print('No argument given, defaulting to 100 iteration.')
        iters = 100
            
    main(iters=iters)
