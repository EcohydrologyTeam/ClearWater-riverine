# %%
import unittest
import os
import sys
import matplotlib.pyplot as plt

# % [markdown]
# Set paths to the src directories of Riverine and TSM

# %%
local_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.dirname(os.path.dirname(local_path))
riverine_path = os.path.join(repo_path, 'riverine', 'src')
sys.path.append(riverine_path)
sys.path.append(local_path)
from riverine.stations import Station, WaterQualityStation, MetStation


# %%
def test_function_excel(station):
    # Read data from Excel
    date_column_name = 'Date'
    variables = ['Tair', 'Tdew', 'Wind Speed', 'Wind Direction', 'Cloudiness', 'Solar Radiation']
    units = ['degC', 'degC', 'm/s', 'rad', 'fraction', 'W/m2']
    station.read_excel(os.path.join(local_path, 'input_files', 'Berlin_Reservoir_meteorology_2006.xlsx'),
                       sheet_name='Berlin_Reservoir_meteorology',
                       parse_dates=[date_column_name],
                       variables=variables,
                       skiprows=2)

    # Set the units for each data frame
    station.set_units(variables, units)

    # Create a constant time series
    station.make_constant_time_series('Twater', 5.0, units='degC', start='2006-01-01 00:00',
                                      end='2006-12-31 23:00', freq='1H')

    # Interpolate to a single date-time value
    interpolation_date = '2006-01-01 01:30'
    print(f'\nInterpolating all variables to {interpolation_date}:')
    interpolated_values = []
    for variable in variables:
        interpolated_value = station.interpolate_variable(variable, interpolation_date)
        print(f'{variable:20s} {interpolated_value:5.2f}')
        interpolated_values.append(interpolated_value)

    # Plot the data
    station.plot()
    plt.show()

# %%
station = Station(39.0, -120.0, 10.0)
test_function_excel(station)
