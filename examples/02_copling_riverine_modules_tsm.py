# %% [markdown]
# # ClearWater-Riverine Demo 2: Coupling Transport to Water Quality Reactions with ClearWater-Modules
# 
# **Objective**: Demonstrate a more complex scenario of coupled transport and reaction models in Sumwere Creek, using the [ClearWater-modules](https://github.com/EcohydrologyTeam/ClearWater-modules) to simulate heat exchange with the atmosphere.
# 
# This second notebook builds on the introduction to using [ClearWater-riverine](https://github.com/EcohydrologyTeam/ClearWater-riverine) provided in demo notebook 1.
# 
# ## Background 
# This notebook couples Clearwater-riverine (transport) with Clearwater-modules (reactions) - specifically, the Temperature Simulation Model (TSM). The Temperature Simulation Module (TSM) is an essential component of ClearWater (Corps Library for Environmental Analysis and Restoration of Watersheds). TSM plays a crucial role in simulating and predicting water temperature within aquatic ecosystems. TSM utilizes a comprehensive energy balance approach to account for various factors contributing to heat inputs and outputs in the water environment. It considers both external forcing functions and heat exchanges occurring at the water surface and the sediment-water interface. The primary contributors to heat exchange at the water surface include shortwave solar radiation, longwave atmospheric radiation, heat conduction from the atmosphere to the water, and direct heat inputs. Conversely, the primary factors that remove heat from the system are longwave radiation emitted by the water, evaporation, and heat conduction from the water to the atmosphere. 
# The core principle behind TSM is the application of the laws of conservation of energy to compute water temperature. This means that the change in heat content of the water is directly related to changes in temperature, which, in turn, are influenced by various heat flux components. The specific heat of water is employed to establish this relationship. Each term of the heat flux equation can be calculated based on the input provided by the user, allowing for flexibility in modeling different environmental conditions
# 
# ## Example Case Study
# 
# This example shows how to run Clearwater Riverine coupled with Clearwater Modules in a fictional location, "Sumwere Creek" (shown below). The flow field for Sumwere Creek comes from a HEC-RAS 2D model, which has a domain of 2x2 km and a base mesh cell size of 100x100 meters. 
# 
# ![image.png](../docs/imgs/SumwereCreek_coarse.png)
# 
# The upstream boundary for Sumwere Creek is at the top left of the model domain, flowing into the domain at a constant 3 cms. At the first bend in the creek, there is an additional boundary representing a spring-fed tributary to the creek (1 cms). Further downstream, there is a meander in the stream forming a slow-flowing oxbow lake. There is another boundary flowing into that oxbow lake, representing a powerplant discharge (0.5 cms). 
# 
# The downstream boundary is a constant stage set at 20.75. The upstream inflows have a water temperature of 15 degrees C; the spring-fed creek has constant inflows of 5 C, and the powerplant is steady at 20 C with periodic higher temperature (25 C) discharges in a downstream meander.  
# 
# We simulate this scenario over the course of two full days, using meteorological parameters from Arizona (extreme temperature swings between night and day) to help show off the impacts of TSM.
# 
# ### Data Availability
# All data required run this notebook is available at this [Google Drive](https://drive.google.com/drive/folders/19uCjAJPZh4g6r1ZWzk1D_B8jZGluSc4N?usp=drive_link). 
# This notebook will use the `sumwere_creek_coarse_p48` model. Please download that entire folder and place it in the `data_temp` folder of this repository to run the rest of the notebook. 
# 
# Alternatively, if you would like to run a different version of the model (see the [ReadMe](https://docs.google.com/document/d/1FKjrTZHUYmYxo0mgn72dOezHtq-CFR86rQ1ObD1ZY0c/edit) for details), download that folder, place it in the `data_temp` folder. You may need to adjust path names in the notebook accordingly.

# %% [markdown]
# ## Model Set-Up
# ### General Imports

# %%
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
import geoviews as gv
# from holoviews import opts
import panel as pn
hv.extension("bokeh")
import warnings

from shared import process_meteo_data
from shared import setup_function_logger

import clearwater_riverine as cwr
from clearwater_modules.tsm.model import EnergyBudget

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')

# Find project directory (i.e. the parent to `/examples` directory for this notebook)
project_path = Path.cwd().parent

# Your source directory should be: 
src_path = project_path / 'src'


# ## Instantiate Models
# ### Clearwater-Riverine
model_name = 'sumwere_creek_coarse_p48'

# required for riverine
test_case_path = project_path / 'examples/data_temp' / model_name
riverine_config = test_case_path / 'demo_config.yml'

# requierd information for modules
wetted_surface_area_path = test_case_path / "wetted_surface_area.zarr"
q_solar_path = test_case_path / 'cwr_boundary_conditions_q_solar_p28.csv'
air_temp_path = test_case_path / 'cwr_boundary_conditions_TairC_p28.csv'

start_index = int(8*60*(60/30))  # start at 8:00 am on the first day of the simulation (30 second model)
end_index = start_index + int(8*60*(60/30))  # end 8 hours later (30 second model)

transport_model = cwr.ClearwaterRiverine(
    config_filepath = riverine_config,
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

# ### Clearwater-Modules
# #### Initial State Values
# The initial state values are `water_temp_c`, `volume`, and `surface_area` come from Clearwater-riverine mesh at the first timestep.
# Provide xr.data array values for initial state values
initial_state_values = {
    'water_temp_c': transport_model.mesh['temperature'].isel(
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

# Create a pandas datetime index from the transport model
xarray_time_index = pd.DatetimeIndex(
    transport_model.mesh.time.values
)

# Next, interpolate the meteorological station data to the same timestep as our model. To simplify this process in this example, we leverage the `process_meteo_data` function in the shared modules within this example folder.

# Read CSV data into pandas dataframes
q_solar = process_meteo_data(
    q_solar_path,
    xarray_time_index,
    'q_Solar'
)

air_temp_c = process_meteo_data(
    air_temp_path,
    xarray_time_index,
    'TairC'
)

air_temp_c['air_temp_c'] = (air_temp_c.tairc - 32)* (5/9)

# Finally, we can create dictionaries containing all meteorological data and the initial conditions. These will be used as inputs to Clearwater Modules. 
# process dataframes for ClearWater 
q_solar_array = q_solar.q_solar.to_numpy()
air_temp_array = air_temp_c.air_temp_c.to_numpy()

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

# #### Temperature Parameters
# Users can also optionally define any `temp_parameters` that should be different than the default.
# In this case, our model timestep `dt` is equal to 30 seconds. However, by default in Clearwater-Modules, it is 1 day. We will therefore need to define `dt`:

temp_parameters = {
    'dt': 30 / 86400  # 30 seconds in days
}

# %% [markdown]
# #### Instantiate Clearwater Modules
# We instantiate Clearwater Modules with the following:
# * `time_steps` (required): the number of timesteps to run. 
# * `initial_state_values` (required): our initial conditions of water temperature, cell volumes, and cell surface areas.
# * `updateable_static_variables` (optional): by default, the meteorological variables are static in TSM. If we want these to update over time, we must provide a list of variables that we want to be updateable as input when instantiating the model. 
# * `meteo_parameters` (optional): intitial meteorological parameters. If not provided, all meteo parameters will fall to default values.
# * `temp_parameters` (optional): update parameters for the TSM model. If not provided, all temperature parameters will fall to default values.
# * `track_dynamic_variables` (optional): boolean indicating whether or not the user wants to track all intermediate information used in the calculations. We set this to `False` to save on memory.
# * `use_sed_temp` (optional): boolean indicating whether to use the sediment temperature in TSM calculations. We opt to turn this off for simplicity.
# * `time_dim` (optional): the model timestep

# Instantiate the same number of timesteps for Modules as we have in Clearwater Riverine
time_steps = len(transport_model.mesh.time)

reaction_model = EnergyBudget(
    time_steps=time_steps,
    initial_state_values=initial_state_values,
    updateable_static_variables=['air_temp_c', 'q_solar'],
    meteo_parameters= initial_meteo_params,
    temp_parameters=temp_parameters,
    use_sed_temp=False,
    track_dynamic_variables=False, 
    time_dim='seconds'
    )

# ## Couple Models
# 
# ### Set-Up Coupling Function
# Now that we have instantiated both our `Clearwater-Riverine` and `Clearwater-Modules` models, we can couple them. We will do so using the `run_n_timesteps` function, which runs `n` number of timesteps, with the following process:
# 1. Optionally sets up a logger. 
# 2. Top of the timestep: Increment the transport model (Riverine). After the first timestep, information from Clearwater-Modules will be passed back into Clearwater-Riverine.
# 3. Create inputs for Clearwater Modules with outputs from Clearwater Riverine and meteorological data
# 4. Bottom of the tiemestep: Increment the reaction model (Modules).
# 5. Create inputs for Clearwater Riverine with outputs from Clearwater Modules.
# 
# The inputs for the function are as follows:
# * `time_steps`: The number of timesteps to run.
# * `reaction`: The Clearwater-Modules model (instantiated above)
# * `transport`: The Clearwater-Riverine model (instantiated above)
# * `meteo_params`: Meteorological inputs (defined above)
# * `riverine_to_modules`: A list of inputs to modules for the specified model
# * `modules_to_riverine`: A list of inputs from riverine back to modules.
# * `modules_to_riverine_matching`: A dictionary mapping the names of clearwater modules variable names (keys) to clearwater riverine module names (values). Only needed where the values are different.
# * `concentration_update`: this will be None on the first timestep; it will get updated by Clearwater Modules.
# * `logging`: Boolean to log (True) or not (False). False by default.
# * `log_file_name`: Name of log file. `log` by default.
# * `logging_interval`: Number of timesteps that should pass between logs. Selecting a small number will slow down the model.
# 

def run_n_timesteps(
    time_steps: int,
    reaction: EnergyBudget,
    transport: cwr.ClearwaterRiverine,
    meteo_params: dict,
    riverine_to_modules: list,
    modules_to_riverine: list,
    modules_to_riverine_matching={},
    concentration_update=None,
    logging=False,
    log_file_name='log',
    logging_interval=5000,
):
    """Function to couple Clearwater Riverine and Modules for n timesteps."""

    # 1. Set up logger
    if logging:
        logger = setup_function_logger(f'{log_file_name}')

    # Loop through all timesteps
    for i in range(1, time_steps):
        if logging:
            if i % logging_interval == 0:
                status = {
                    'timesteps': i,
                    'cwr': transport.mesh.nbytes * 1e-9,
                    'cwm': reaction.dataset.nbytes*1e-9,
                }
                logger.debug(status)

        # 2. Top of timestep: Update transport model
        transport.update(concentration_update)

        # 3. Update state values
        # 3.1 Update using outputs from Clearwater Riverine
        updated_state_values = {}
        for state_variable_name in riverine_to_modules:
            if state_variable_name in modules_to_riverine_matching:
                riverine_key = modules_to_riverine_matching[state_variable_name]
            else:
                riverine_key = state_variable_name
            updated_state_values[state_variable_name] = transport.mesh[riverine_key].isel(
                    time=i,
                    nface=slice(0, transport.mesh.nreal + 1)
            )

        # 3.2 Update meteorological inputs
        for meteo_param in meteo_params.keys():
            updated_state_values[meteo_param] = xr.full_like(
                updated_state_values[riverine_to_modules[0]],
                meteo_params[meteo_param][i]
            )

        # 4. Bottom of timestep: update energy budget (TSM)
        reaction.increment_timestep(updated_state_values)

        # 5. Prepare data for input back into Riverine
        concentration_update = {}
        for variable in modules_to_riverine:
            if variable in modules_to_riverine_matching:
                riverine_key = modules_to_riverine_matching[variable]
            else:
                riverine_key = variable

            reaction.dataset[variable] = reaction.dataset[variable].where(
                ~np.isinf(reaction.dataset[variable]),
                transport.mesh[riverine_key].isel(
                    nface=slice(0, transport.mesh.nreal+1),
                    time=i
                )
            )
            reaction.dataset[variable] = reaction.dataset[variable].fillna(
                transport.mesh[riverine_key].isel(
                    nface=slice(0, transport.mesh.nreal+1),
                    time=i
                )
            )
            concentration_update[riverine_key] = reaction.dataset[variable].isel(seconds=i)

# ### Run the Coupling Function
# Earlier in the notebook, we set up most of what we need to couple the models. However, we still need to define a few key inputs that help pass information back and forth between Clearwater Riverine with the following input parameters:
# * `riverine_to_modules`: We are passing water temperature, surface area, and volume from Riverine to Modules.
# * `modules_to_riverine`: We are passing water temperature from Modules back to Riverine.
# * `modules_to_riverine_matching`: Water temperature and surface area are named differently in the two models. We'll need to use this dictionary to show that link. 

# Define the names of variables being passed from Riverine to Modules
riverine_to_modules = [f.name for f in reaction_model.state_variables]

modules_to_riverine = ['water_temp_c']

modules_to_riverine_matching = {
    'water_temp_c': 'temperature',
    'surface_area': 'wetted_surface_area'
}

# Now we have all the inputs required to run the function! 
# Let's run it for all timesteps below:

run_n_timesteps(
    time_steps=time_steps,
    reaction=reaction_model,
    transport=transport_model,
    meteo_params=all_meteo_params,
    riverine_to_modules=riverine_to_modules,
    modules_to_riverine=modules_to_riverine,
    modules_to_riverine_matching=modules_to_riverine_matching,
)
