import unittest
import datetime
import os
import sys

'''
Set path to the src directory
'''

local_path = os.path.dirname(os.path.realpath(__file__))
inputs_path = os.path.join(local_path, 'input_files')
outputs_path = os.path.join(local_path, 'output_files')
repo_path = os.path.join(local_path.split('ClearWater')[0], 'ClearWater')
src_path = os.path.join(repo_path, 'riverine', 'src', 'riverine', 'ras2d')
riverine_path = os.path.join(repo_path, 'riverine', 'src')

# Append paths to Python path
sys.path.append(riverine_path)
sys.path.append(local_path)
print(f'local path: {local_path}')
print(f'inputs path: {inputs_path}')
print(f'outputs path: {outputs_path}')
print(f'repo path: {repo_path}')
print(f'src path: {src_path}')
print(f'riverine path: {riverine_path}')

sys.path.insert(0, src_path)

import RAS2D
from riverine.stations import Station, WaterQualityStation, MetStation


class Test_RAS2D_Muncie(unittest.TestCase):

    def test_advection(self):
        print('test_advection()')
        # Muncie, Indiana, East Indiana State Plane, NAD83, feet
        ras_file_path = os.path.join(inputs_path, 'Muncie.p04.hdf')

        # Time step (5 minutes)
        dt: float = 5 * 60.0

        # Initial temperature
        Ti: float = 20.0

        # Offset the data from 1900 to 2006
        time_delta = datetime.datetime(2006, 1, 1) - datetime.datetime(1900, 1, 1) - datetime.timedelta(days=1)
        datetime.timedelta()
        
        # Read data from CSV
        met_station = MetStation(39.0, -120.0, 10.0)
        date_column_name = 'Date'
        variables = ['Tair', 'Tdew', 'Wind Speed', 'Wind Direction', 'Cloudiness', 'Solar Radiation']
        units = ['degC', 'degC', 'm/s', 'rad', 'fraction', 'W/m2']
        met_station.read_csv(os.path.join(local_path, 'input_files', 'Berlin_Reservoir_meteorology_2006.csv'),
                        parse_dates=[date_column_name],
                        variables=variables,
                        skiprows=2)

        # Set the units for each data frame
        met_station.set_units(variables, units)

        # Derive air pressure, relative humidity, saturation vapor pressure, and vapor pressure
        met_station.derive_atmospheric_pressure_time_series()
        met_station.derive_relative_humidity()
        met_station.derive_saturation_vapor_pressure()
        met_station.derive_vapor_pressure()

        # Simulate WQ with advection-diffusion
        RAS2D.wq_simulation(ras_file_path, Ti, met_station, diffusion_coef=0.2, dt=dt, time_delta=time_delta)


if __name__ == '__main__':
    unittest.main()