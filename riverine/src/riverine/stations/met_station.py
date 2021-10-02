from station import Station
import matplotlib.pyplot as plt
import os
import sys
import numpy
import pandas


class MetStation(Station):
    '''
    Station: reads, writes, and processes meteorology data.
    It holds any number of arbitrary parameters. This module
    assumes that all data are in SI units.

    Required meteorological variables and their units
        Tair: Air temperature (degC)
        Tdew: Dewpoint temperature (degC)
        Wind Speed: Wind speed (m/2)
        Wind Direction: Wind direction (rad)
        Cloudiness: Fraction of cloud cover (0.0 - 0.9)
        Solar Radiation: Solar radiation (W/m2)
    '''
    def __init__(self, latitude: float, longitude: float, elevation: float, dust_coefficient: float = 0.2,
                anemometer_height=2.0, wind_a: float = 0.3, wind_b: float = 1.5, wind_c: float = 1.0,
                wind_kh_kw: float = 1.0, use_Richardson_Number: bool = True):

        super().__init__(latitude, longitude, elevation)

        self.variables = ['Tair', 'Tdew', 'Wind Speed', 'Wind Direction', 'Cloudiness', 'Solar Radiation']
        self.units = ['degC', 'degC', 'm/s', 'rad', 'fraction', 'W/m2']

        '''
        Dust Coefficient:
            The dust coefficient accounts for the amount of attenuation of downwelling 
            shortwave radiation by dust, due to scattering and absorption. It varies 
            from 0 (typical of rural areas) to 0.2 (typical of urban aras).
        '''
        self.dust_coefficient = dust_coefficient

        '''
        Wind:
            Wind function: a+bu**c
            Wind speed coefficients (a, b, and c):
                wind_a x 1E-6
                wind_b x 1E-6
                wind_c
            Anemometer (wind gauge) height (m). Standard height is 2 m.
            Surface characterization - Need to implement setting this value and accounting for roughness
        '''
        self.anemometer_height = anemometer_height
        self.wind_a = wind_a
        self.wind_b = wind_b
        self.wind_c = wind_c
        self.wind_kh_kw = wind_kh_kw

        '''
        Richardson Number
        '''
        self.use_Richardson_Number = use_Richardson_Number

        '''
        Cloudiness:
            Cloudiness(fraction) is the fraction of sky covered with clouds.
            It varies from 0 to 0.9.  Cloudiness is a required parameter for
            both calculated solar radiation and downwelling longwave radiation.
            Clouds decrease solar radiation and increase downwelling longwave
            radiation.

            A rough guideline for cloudiness is:

            Overcast skies: 0.9
            Broken skies: 0.5 - 0.9
            Scatter clouds: 0.1 - 0.5
            Clear skies: 0.1
        '''
    def derive_relative_humidity(self):
        ''' Derive relative humidity (%) from air temperature (C) and dewpoint temperature (C) '''
        # TODO: Check this!
        variable_name = 'Relative Humidity'
        rh = 100.0 - 5.0 * (self.data['Tair'] - self.data['Tdew']) # percent
        df = pandas.DataFrame(rh, index=self.dates, columns=[variable_name])
        self.data[variable_name] = df

    def derive_saturation_vapor_pressure(self):
        ''' Derive saturation vapor pressure (mb) from air temperature (C) '''
        # TODO: Check this
        variable_name = 'Saturation Vapor Pressure'
        esat_mb = 6.1078 * numpy.exp((17.269 * self.data['Tair']) / (237.3 + self.data['Tair']))
        df = pandas.DataFrame(esat_mb, index=self.dates, columns=[variable_name])
        self.data[variable_name] = df

    def derive_vapor_pressure(self):
        ''' Derive vapor pressure (mb) from relative humidity (%) and saturation vapor pressure (mb) '''
        # TODO: Check this
        variable_name = 'Vapor Pressure'
        # Td = T - ((100 - RH)/5.0
        # RH = eair_mb/esat_mb * 100%
        eair_mb = self.data['Relative Humidity']/100.0 * self.data['Saturation Vapor Pressure']
        df = pandas.DataFrame(eair_mb, index=self.dates, columns=[variable_name])
        self.data[variable_name] = df

    def derive_atmospheric_pressure_time_series(self):
        '''
        Derive the time series of atmospheric pressure, given the Tair time series,

        Standard atmospheric pressure at sea level is 101.325 Pa (1013.25 mb, 1 atm)
        and the station elevation.
        '''

        Tair = self.data['Tair']
        TairK = Tair - 273.15
        elevation = self.elevation
        df = Tair.copy()
        df['Air Pressure'] = estimate_atmospheric_pressure(TairK.values, elevation)
        df = df.drop(columns='Tair')
        df.attrs['units'] = 'Pa'
        self.data['Air Pressure'] = df

    '''
    Specify properties for accessing select data within a MetStation object
    '''

    @property
    def dust_coefficient(self) -> float:
        return self._dust_coefficient

    @dust_coefficient.setter
    def dust_coefficient(self, value: float):
        self._dust_coefficient = value

    @property
    def anemometer_height(self) -> float:
        return self._anemometer_height

    @anemometer_height.setter
    def anemometer_height(self, value: float):
        self._anemometer_height = value


def estimate_atmospheric_pressure(TairK: float, elevation: float) -> float:
    '''
    Estimate the atmospheric pressure in Pacals from the station elevation using the barometric formula:
        P = P₀ exp(-gM(h-h₀)/(RT))

        * h is the altitude at which we want to calculate the pressure, expressed in meters.
        * P is the air pressure at altitude h.
        * P₀ is the pressure at the reference level, h₀ (sea level, 0 m elevation).
        * T is the temperature at altitude h, expressed in Kelvins.
        * g is the gravitational acceleration. For Earth, g = 9.80665 m/s².
        * M is the molar mass of air. For Earthly air, M = 0.0289644 kg/mol.
        * R is the universal gas constant. Its value is equal to R = 8.31432 N·m/(mol·K).

    Reference: https://www.omnicalculator.com/physics/air-pressure-at-altitude
    '''

    T = TairK
    P0 = 101.325 # Pa
    h = elevation # meters
    h0 = 0.0 # meters
    g = 9.80665 # m/s2
    M = 0.0289644 # kg/mol
    R = 8.31432 # N·m/(mol·K)

    return P0 * numpy.exp(-g*M*(h-h0)/(R*T))


if __name__ == '__main__':
    # Create a Station object
    met_station = MetStation(39.0, -120.0, 0.001)
    # Read data from CSV
    date_column_name = 'Date'
    variables = ['Tair', 'Tdew', 'Wind Speed', 'Wind Direction', 'Cloudiness', 'Solar Radiation']
    units = ['degC', 'degC', 'm/s', 'rad', 'fraction', 'W/m2']
    local_path = os.path.dirname(os.path.realpath(__file__))
    met_station.read_csv(os.path.join(local_path, 'Berlin_Reservoir_meteorology_2006.csv'),
                     parse_dates=[date_column_name],
                     variables=variables,
                     skiprows=2)
    # Set the units for each data frame
    met_station.set_units(variables, units)

    # Create a constant time series
    met_station.make_constant_time_series('Twater', 5.0, units='degC',
                                      start='2006-01-01 00:00', end='2006-12-31 23:00', freq='1H')
    # Interpolate to a single date-time value
    interpolated_value = met_station.interpolate_variable('Tair', '2006-01-01 01:30')
    print(f'The value at 2006-01-01 01:30 is: {interpolated_value:.2f}')

    # Interpolate to a single date-time value
    interpolation_date = '2006-01-01 01:30'
    print(f'\nInterpolating all variables to {interpolation_date}:')
    for variable in variables:
        interpolated_value = met_station.interpolate_variable(variable, interpolation_date)
        print(f'{variable:20s} {interpolated_value:5.2f}')

    # # Plot the data
    # met_station.plot()
    # plt.show()

    interpolated_value = met_station.interpolate_variable('Wind Speed', '2007-01-01 00:00')
    print(interpolated_value)

    met_station.derive_atmospheric_pressure_time_series()
    met_station.plot(['Air Pressure'])
    print(met_station.data['Air Pressure'])
    plt.show()