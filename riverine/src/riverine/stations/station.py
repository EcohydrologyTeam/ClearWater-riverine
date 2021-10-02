import pandas
import seaborn
import numpy
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, local_path)


class Station:
    '''
    Station: reads, writes, and processes meteorology and water quality data.
    It holds any number of arbitrary parameters. This module assumes that all 
    data are in SI units.
    '''

    figure_size = (15, 8)
    data = {}

    def __init__(self, latitude: float, longitude: float, elevation: float):
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

    def read_csv(self, file_path: str, parse_dates=['Date'], variables=['default'], skiprows: int = 0):
        '''
        Read data from a CSV file into a pandas data frame, label the columns, 
        separate into one data frame for each variable, and set a units attribute 
        on each data frame.
        '''
        df = pandas.read_csv(file_path, parse_dates=parse_dates, skiprows=skiprows)

        # Store the dates to use with other time series
        self.dates = df['Date']

        # Set the Date column as the index of the data frame, for plotting and analysis
        df = df.set_index('Date')


        df.columns = variables
        for variable in variables:
            self.data[variable] = df[[variable]].copy()
        self.imported_dataframe = df

    def read_excel(self, file_path: str, sheet_name=0, parse_dates=['Date'], variables=['default'],
                   skiprows: int = 0, header=0, names=None, index_col=None, usecols=None, squeeze=False,
                   dtype=None, engine=None, converters=None, true_values=None, false_values=None, nrows=None,
                   na_values=None, keep_default_na=True, na_filter=True, verbose=False, date_parser=None,
                   thousands=None, comment=None, skipfooter=0, convert_float=None, mangle_dupe_cols=True,
                   storage_options=None):
        '''
        Read data from a Microsoft Excel file into a pandas data frame, label the columns, 
        separate into one data frame for each variable, and set a units attribute on each 
        data frame.
        '''
        df = pandas.read_excel(
            file_path, sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols,
            squeeze=squeeze, dtype=dtype, engine=engine, converters=converters, true_values=true_values,
            false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values,
            keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose, parse_dates=parse_dates,
            date_parser=date_parser, thousands=thousands, comment=comment, skipfooter=skipfooter,
            convert_float=convert_float, mangle_dupe_cols=mangle_dupe_cols, storage_options=storage_options)

        # Store the dates to use with other time series
        self.dates = df['Date']
            
        # Set the Date column as the index of the data frame, for plotting and analysis
        df = df.set_index('Date')
        df.columns = variables
        for variable in variables:
            self.data[variable] = df[[variable]].copy()
        self.imported_dataframe = df

    def set_units(self, variables=[], units=[]):
        for v, u in zip(variables, units):
            self.data[v].attrs['units'] = u

    def make_constant_time_series(self, variable_name, value, units='', start=None, end=None, periods=None, freq=None,
                                  tz=None, normalize=False, name=None, closed=None, **kwargs):
        '''
        Example:
        make_constant_time_series('Tair', 5.0, start='2006-01-01 00:00', end='2006-12-31 23:00', freq='1H')
        '''
        dates = pandas.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz,
                              normalize=normalize, name=name, closed=closed, **kwargs)
        df = pandas.DataFrame(value, index=dates, columns=[variable_name])
        df.attrs['units'] = units
        self.data[variable_name] = df

    def write_hdf5(self, file_path, data_path):
        with pandas.HDFStore(file_path, 'w') as f:
            self.data.to_hdf(f, data_path)

    def read_hdf5(self, file_path, data_path):
        self.data = pandas.read_hdf(file_path, data_path)

    def plot(self, variables: list = []):
        seaborn.set(rc={'figure.figsize': self.figure_size})
        if variables == []:
            variables = self.data.keys()
        for variable in variables:
            df = self.data[variable]
            ax = df.plot()
            units = df.attrs['units']
            ax.ticklabel_format(axis='y', useOffset=False)
            ax.set_ylabel(f'{variable} ({units})')

    def interpolate_variable(self, variable_name, target_date: str) -> float:
        '''
        Interpolate data to a single date-time point.

        Example:
        df_interp = self.interpolate('2006-01-01 01:30')

        Method:
        Take a subset of the Series including only the entry 
        above and below target_date and then use interpolate.

        Returns:
        A pandas DataFrame with one row (for multi-column data frames)
        or a single value (for single-column data frames)
        '''

        # Get data frame for a single variable
        df = self.data[variable_name]
        # Interpolate
        df_interp = interpolate_dataframe(df, target_date)
        # Return a single floating point value
        try:
            interp_value: float = df_interp[variable_name].iloc[0]
        except AttributeError:
            interp_value = numpy.nan
        return interp_value



def interpolate_dataframe(df, target_date: str) -> pandas.DataFrame:
    '''
    Interpolate data to a single date-time point.

    Example:
    df_interp = self.interpolate('2006-01-01 01:30')

    Method:
    Take a subset of the Series including only the entry 
    above and below target_date and then use interpolate.

    Returns:
    A pandas DataFrame with one row (for multi-column data frames)
    or a single value (for single-column data frames)
    '''

    # Ensure the data are sorted by date
    ts1 = df.sort_index()
    # Index of first entry after target
    b = (ts1.index > target_date).argmax()
    s = ts1.iloc[b-1:b+1]
    # Insert empty value at target time
    s = s.reindex(pandas.to_datetime(list(s.index.values) + [pandas.to_datetime(target_date)]))
    # Interpolate
    df_interp = s.interpolate('time').loc[target_date]
    return df_interp


if __name__ == '__main__':
    # Create a Station object
    station = Station(39.0, -120.0, 10.0)
    # Read data from CSV
    date_column_name = 'Date'
    variables = ['Tair', 'Tdew', 'Wind Speed', 'Wind Direction', 'Cloudiness', 'Solar Radiation']
    units = ['degC', 'degC', 'm/s', 'rad', 'fraction', 'W/m2']
    station.read_csv(os.path.join(local_path, 'Berlin_Reservoir_meteorology_2006.csv'),
                     parse_dates=[date_column_name],
                     variables=variables,
                     skiprows=2)
    # Set the units for each data frame
    station.set_units(variables, units)
    # Create a constant time series
    station.make_constant_time_series('Twater', 5.0, units='degC',
                                      start='2006-01-01 00:00', end='2006-12-31 23:00', freq='1H')
    # Interpolate to a single date-time value
    interpolated_value = station.interpolate_variable('Tair', '2006-01-01 01:30')
    print(f'The value at 2006-01-01 01:30 is: {interpolated_value:.2f}')
    # Plot the data
    station.plot()
    plt.show()
