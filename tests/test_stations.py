import unittest
import os
import sys
import matplotlib.pyplot as plt

'''
Set paths to the src directories of Riverine and TSM
'''

# Local (tests) path
local_path = os.path.dirname(os.path.realpath(__file__))
print(f'Local (tests) path: {local_path}')  # ~/riverine/tests

# Repo path (two directories up the path)
repo_path = os.path.dirname(os.path.dirname(local_path))
print(f'Repository path: {repo_path}')  # ../../

# Source path (folder that contains the riverine folder/module)
riverine_path = os.path.join(repo_path, 'riverine', 'src')
print(f'Riverine source path: {riverine_path}')  # ../../riverine/src

# Append paths to Python path
sys.path.append(riverine_path)
sys.path.append(local_path)

# Note: the following form of the input works since Station, WaterQualityStation, and MetStation were imported
# in riverine/stations/__init__.py
# Otherwise, the import would be:
#   from riverine.stations.station import Station
#   from riverine.stations.wq_station import WaterQualityStation
#   from riverine.stations.met_station import MetStation
from riverine.stations import Station, WaterQualityStation, MetStation


def test_function_csv(station):
    # Read data from CSV
    date_column_name = 'Date'
    variables = ['Tair', 'Tdew', 'Wind Speed', 'Wind Direction', 'Cloudiness', 'Solar Radiation']
    units = ['degC', 'degC', 'm/s', 'rad', 'fraction', 'W/m2']
    station.read_csv(os.path.join(local_path, 'input_files', 'Berlin_Reservoir_meteorology_2006.csv'),
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
    for variable in variables:
        interpolated_value = station.interpolate_variable(variable, interpolation_date)
        print(f'{variable:20s} {interpolated_value:5.2f}')

    # Plot the data
    # station.plot()
    # plt.show()


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
    for variable in variables:
        interpolated_value = station.interpolate_variable(variable, interpolation_date)
        print(f'{variable:20s} {interpolated_value:5.2f}')

    # Plot the data
    station.plot()
    plt.show()


class Test_Riverine(unittest.TestCase):
    def setUp(self):
        pass

    def test_station_csv(self):
        print('test_station_csv')
        station = Station(39.0, -120.0, 10.0)
        test_function_csv(station)

    def test_water_quality_station_csv(self):
        print('test_water_quality_station_csv')
        station = WaterQualityStation(39.0, -120.0, 10.0)
        test_function_csv(station)

    def test_met_station_csv(self):
        print('test_met_station_csv')
        station = MetStation(39.0, -120.0, 10.0)
        test_function_csv(station)

    def test_station_excel(self):
        print('test_station_excel')
        station = Station(39.0, -120.0, 10.0)
        test_function_excel(station)


if __name__ == '__main__':
    unittest.main()
