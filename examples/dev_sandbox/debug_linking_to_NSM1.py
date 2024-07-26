# import modules
from pathlib import Path
import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
import geoviews as gv
# from holoviews import opts
import panel as pn
hv.extension("bokeh")

import clearwater_riverine as cwr
from clearwater_modules.nsm1.model import NutrientBudget


project_path = Path.cwd().parent
src_path = project_path / 'src'
print(project_path)
print(src_path)

#point to config
network_path = Path(r'W:\2ERDC12 - Clearwater\Clearwater_testing_NSM\plan_48_simulation')
wetted_surface_area_path = network_path / "wetted_surface_area.zarr"
q_solar_path = network_path / 'cwr_boundary_conditions_q_solar_p28.csv'
air_temp_path = network_path / 'cwr_boundary_conditions_TairC_p28.csv'
config_file = network_path / 'demo_config.yml'
print(config_file.exists())

start_index =  0 # int((8*60*60)/30)
end_index = 48*60*60
print(start_index, end_index)

transport_model = cwr.ClearwaterRiverine(
    config_filepath=config_file,
    verbose=True,
    datetime_range= (start_index, end_index)
)

wetted_sa = xr.open_zarr(wetted_surface_area_path)
wetted_sa = wetted_sa.compute()
wetted_sa_subset = wetted_sa.isel(time=slice(start_index, end_index+1))

transport_model.mesh['wetted_surface_area'] = xr.DataArray(
    wetted_sa_subset['wetted_surface_area'].values,
    dims=('time', 'nface')
)




# Provide xr.data array values for initial state values
initial_state_values = {
    'Ap': transport_model.mesh['Ap'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'DOX': transport_model.mesh['DOX'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'NH4': transport_model.mesh['NH4'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'NO3': transport_model.mesh['NO3'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'TIP': transport_model.mesh['TIP'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'Ab': transport_model.mesh['Ab'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'OrgN': transport_model.mesh['OrgN'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'N2': transport_model.mesh['N2'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'OrgP': transport_model.mesh['OrgP'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'POC': transport_model.mesh['POC'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'DOC': transport_model.mesh['DOC'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'DIC': transport_model.mesh['DIC'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'POM': transport_model.mesh['POM'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'CBOD': transport_model.mesh['CBOD'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'PX': transport_model.mesh['PX'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'Alk': transport_model.mesh['Alk'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'volume': transport_model.mesh['volume'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal+1)
    ),
    'surface_area': transport_model.mesh['wetted_surface_area'].isel(
        time=0,
        nface=slice(0, transport_model.mesh.nreal + 1)
    ),
}




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


xarray_time_index = pd.DatetimeIndex(
    transport_model.mesh.time.values
)


# Q Solar
q_solar = pd.read_csv(
    q_solar_path, 
    parse_dates=['Datetime'])
q_solar.dropna(axis=0, inplace=True)

q_solar_interp = interpolate_to_model_timestep(
    q_solar,
    xarray_time_index,
    'q_Solar'
)



# Air Temperature
air_temp_c = pd.read_csv(
    air_temp_path, 
    parse_dates=['Datetime'])
air_temp_c.dropna(axis=0, inplace=True)

air_temp_c_interp = interpolate_to_model_timestep(
    air_temp_c,
    xarray_time_index,
    'TairC'
)

air_temp_c_interp['air_temp_c'] = (air_temp_c_interp.tairc - 32)* (5/9)




# process for clearwater-modules input
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



time_Steps = len(transport_model.mesh.time)


reaction_model = NutrientBudget(
    time_steps=time_Steps,
    initial_state_values=initial_state_values,
    #updateable_static_variables=['q_solar'],
    
    algae_parameters = {
    'AWd': 100,
    'AWc': 40,
    'AWn': 7.2,
    'AWp': 1,
    'AWa': 1000,
    'KL': 10,
    'KsN': 0.04,
    'KsP': 0.0012,
    'mu_max_20': 1,
    'kdp_20': 0.15,
    'krp_20': 0.2,
    'vsap': 0.15,
    'growth_rate_option': 3,
    'light_limitation_option': 1 
    },

    global_parameters={
    'use_NH4': True,
    'use_NO3': True, 
    'use_OrgN': False,
    'use_OrgP': False,
    'use_TIP': True,  
    'use_SedFlux': False,
    'use_DOX': True,
    'use_Algae': True,
    'use_Balgae': False,
    'use_POC': False,
    'use_DOC': False,
    'use_DIC': False,
    'use_N2': False,
    'use_Pathogen': False,
    'use_Alk': False,
    'use_POM': False 
    },

    global_vars = {
    'vson': 0.01,
    'vsoc': 0.01,
    'vsop': 0.01,
    'vs': 0.01,
    'SOD_20': .5,
    'SOD_theta': 1.047,
    'vb': 0.01,
    'fcom': 0.4,
    'kaw_20_user': 0,
    'kah_20_user': 1,
    'hydraulic_reaeration_option': 1,
    'wind_reaeration_option': 1,    
    'dt': 0.0003472222,
    'depth': 1.5,
    'TwaterC': 25,
    'theta': 1.047,
    'velocity': 1,
    'flow': 150,
    'topwidth': 100,
    'slope': .0002,
    'shear_velocity': 0.05334,
    'pressure_atm': 1013.25,
    'wind_speed': 3,
    'q_solar': 500,
    'Solid': 1,
    'lambda0': 0.02,
    'lambda1': 0.0088,
    'lambda2': 0.054,
    'lambdas': 0.056,
    'lambdam': 0.174, 
    'Fr_PAR': 0.47  
    },

    track_dynamic_variables=False, 
    time_dim='seconds'    
    )




def setup_function_logger(name):
    logger = logging.getLogger('function_logger')
    handler = logging.FileHandler(f'{name}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # Adjust the level based on your needs
    return logger




def run_n_timesteps(
    time_steps: int,
    reaction: NutrientBudget,
    transport: cwr.ClearwaterRiverine,
    meteo_params: dict,
    concentration_update = None,
    logging = True,
    log_file_name='log',
):
    """Function to couple Clearwater Riverine and Modules for n timesteps."""
    print('func--> run_n_timesteps: ', time_steps) #####for debugging
    # Set up logger
    if logging:
        logger = setup_function_logger(f'{log_file_name}')

    for i in range(1, time_steps):
        if i % 100 == 0:
            status = {
                'timesteps': i,
                'cwr': transport.mesh.nbytes * 1e-9,
                'cwm': reaction.dataset.nbytes*1e-9,
            }
            if logging:
                logger.debug(status)

        # Top of timestep: update transport model using values with output from reaction model, if available
        transport.update(concentration_update)

        # Update state values
        updated_state_values = {
            'Ap': transport.mesh['Ap'].isel(
                time=i,
                nface=slice(0,transport.mesh.nreal+1)
            ), 
            'NH4': transport.mesh['NH4'].isel(
                time=i,
                nface=slice(0,transport.mesh.nreal+1)
            ), 
            'NO3': transport.mesh['NO3'].isel(
                time=i,
                nface=slice(0,transport.mesh.nreal+1)
            ), 
            'TIP': transport.mesh['TIP'].isel(
                time=i,
                nface=slice(0,transport.mesh.nreal+1)
            ), 
            'DOX': transport.mesh['DOX'].isel(
                time=i,
                nface=slice(0,transport.mesh.nreal+1)
            ), 
            #'q_solar': transport.mesh.Ap.isel(
            #    time=i,
            #    nface=slice(0, transport.mesh.nreal + 1)
            #) * 0 + meteo_params['q_solar'][i],
        }

        # Bottom of timestep: update nutrient budget (NSM1)
        reaction.increment_timestep(updated_state_values)

        # Prepare data for input back into Riverine
        ds = reaction.dataset.copy()
        ds['Ap'] = ds['Ap'].where(
            ~np.isinf(ds['Ap']),
            transport.mesh['Ap'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )            
        ds['Ap'] = ds['Ap'].fillna(
            transport.mesh['Ap'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )


        ds['DOX'] = ds['DOX'].where(
            ~np.isinf(ds['DOX']),
            transport.mesh['DOX'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )            
        ds['DOX'] = ds['DOX'].fillna(
            transport.mesh['DOX'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )


        ds['NH4'] = ds['NH4'].where(
            ~np.isinf(ds['NH4']),
            transport.mesh['NH4'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )            
        ds['NH4'] = ds['NH4'].fillna(
            transport.mesh['NH4'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )


        ds['NO3'] = ds['NO3'].where(
            ~np.isinf(ds['NO3']),
            transport.mesh['NO3'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )            
        ds['NO3'] = ds['NO3'].fillna(
            transport.mesh['NO3'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )


        ds['TIP'] = ds['TIP'].where(
            ~np.isinf(ds['TIP']),
            transport.mesh['TIP'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )            
        ds['TIP'] = ds['TIP'].fillna(
            transport.mesh['TIP'].isel(
                nface=slice(0, transport.mesh.nreal+1),
                time=i
            )
        )


        concentration_update = {"Ap": ds.Ap.isel(seconds=i),
                                "DOX": ds.DOX.isel(seconds=i),
                                "NH4": ds.NH4.isel(seconds=i),
                                "NO3": ds.NO3.isel(seconds=i),
                                "TIP": ds.TIP.isel(seconds=i)
                                }



TIME_STEPS = len(transport_model.mesh.time) - 1



run_n_timesteps(
    TIME_STEPS,
    reaction_model,
    transport_model,
    all_meteo_params,
    logging=True,
)



#dir_str = os.path.abspath("")
#dir = Path(dir_str)
#dir_sandbox = dir / "examples" / "dev_sandbox"
#config_filepath = dir_sandbox / "demo_config.yml"
#print(dir)
#print(config_filepath)
#print(config_filepath.exists())


#setup model
#transport_model = cwr.ClearwaterRiverine(
#    config_filepath=config_filepath,
#    verbose=True,
#)


#make an array the shape of the temp array with 5,000 everywhere
#arbitrary_values_tracer = xr.full_like(transport_model.mesh.conservative_tracer, 200) 
#print(arbitrary_values_tracer)
#arbitrary_values_temp = xr.full_like(transport_model.mesh.temperature, 5000) 
#update_concentration = {'conservative_tracer_IncorrectName':arbitrary_values_tracer.isel(time=1),
#                        'conservative_tracer':arbitrary_values_tracer.isel(time=1),
#                        'temperature_IncorrectName': arbitrary_values_temp.isel(time=1),
#                        'temperature': arbitrary_values_temp.isel(time=1)}

#transport_model.update(update_concentration)

