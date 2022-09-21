#native
import csv
import datetime
import os
import re
import shutil

#third party
import numpy as np
import pandas as pd
import pkg_resources 

class _BinConfig:
    """
    Internal Class - Stores binary config information
    """
    pass

class EFDCBinaryReader:
    """ Binary file processor for Environmental Fluid Dynamics Code Model results.
    
        Reader converts Fortran binary file into a pandas dataframe. 

        Parameters:
        ----------------
        date_begin: Python datetime object which contains the start for the module run. The resultant 
            timesteps in the binary output are numeric offsets from the start (t=0) and do not contain 
            any date information. Therefore users need to provide the start datetime for the model.

        path_efdcinp: File pathway to the EFDC.inp card deck. This input file contains the run configuration 
            information which the EFDCBinaryReader will use to determine the timestep and output frequency 
            of model results.

        path_config (optional): File pathway to binary_config csv file which contain information including 
            conversion factors and fieldnames. This module contains a default set of config parameters that 
            apply to most of our EFDC model runs, however it is good practice to verify the binary config 
            settings are valid for the run you are trying to process using the 'display_config' method. You 
            can export a copy of the config which you can modify using the 'export_binary_config' method, and then 
            update the EFDCBinaryReader's config using the 'update_binary_config' method.
    """

    def __init__(self, date_begin, path_efdcinp, path_config=None):
        
        self.date_begin = date_begin
        self.path_efdcinp = path_efdcinp
        self.path_config = path_config
        if path_config is None: self.path_config = pkg_resources.resource_filename(__name__, 'data/bin_config.csv')

        self._init_bin_config()    
        self._init_efdc_parameters()

    def _init_bin_config(self):
        """ Internal Method - Takes the path_config and reads it into dictionary of (_BinConfig) """
        results = {}
        try:
            with open(self.path_config) as csvfile:
                reader = csv.DictReader(csvfile)
                for line in reader:
                    bin_config = _BinConfig()
                    for k in line.keys(): setattr(bin_config, k, line[k]) 
                    results[line['bin_file']] = bin_config
            self._bin_config = results
        except IOError as e:
            print('IOError occured: {0}'.format(e))
            raise e

    def update_binary_config(self, path_config):
        """ Take a file pathway to a binary config csv and updates the objects configuration to use the new file.  """
        self.path_config = path_config
        _init_bin_config()

    def export_binary_config(self, export_path):
        """ copies the binary config csv to the location specifed in 'export_path' """
        shutil.copy(pkg_resources.resource_filename(__name__, 'data/bin_config.csv'), export_path)

    def display_config(self, bin_name):
        """ Displays the config settings for the input bin file name """
        config = self._get_config(str.lower(bin_name))
        print("Field name: {0}".format(config.field_name))
        print('Assumed binary file units: {0}'.format(config.bin_units))
        print('Output units: {0}'.format(config.converted_units))
        print('Applying conversion factor: 1 {0} = {1} {2}'.format(config.bin_units, config.conversion_factor, config.converted_units))
        pass
       
    def _get_config(self, bin_name):
        """ Internal Method - returns _BinConfig object from _bin_config dictionary with error handeling for invalid key """
        try:
            return self._bin_config[str.lower(bin_name)]
        except KeyError as e:
            print("Invalid input: No configuration for for binary file '{0}'. Add '{0}' for binary config csv and then update binary config using the 'update_binary_config' method".format(bin_name))
            raise e

    def _init_efdc_parameters(self):
        """
        Opens the EFDC.inp file and read required settings

        The following settings we be loaded from the EFDC.inp file
            C9.LWD - number of water cells 
            C7.NTC - number of time periods in simulation
            C7.NTSPTC - number of timesteps per period in NTC
            C8.TREF - length of single simulation period (1 NTC)
            c70.NSDUMP - timesteps between dumps (export of results to binary file)
        """
        #we could probably clean this method up a bit, but this'll work for now
        try:
            with open(self.path_efdcinp, 'r') as f:
                lines = iter(f.readlines())
                for line in lines:
                    #C7 - NTC & NTSPTC
                    match_result = re.match('\s*C7 ', line)
                    if match_result is not None:
                        #first match is the heading, we need the second match which has the parameters
                        next_line = next(lines)
                        if "*" not in next_line:
                            params = re.split(r"\s+",next_line.lstrip(' ')) #EFDC use space delimiting 
                            ntc = int(params[0]) #NTC is 1st parameter in the line
                            ntsptc = int(params[1]) #NTSPTC is 2nd 
                    
                    #C8 - TREF
                    match_result = re.match('\s*C8 ', line)
                    if match_result is not None:
                        #first match is the heading, we need the second match which has the parameters
                        next_line = next(lines)
                        if "*" not in next_line:
                            params = re.split(r"\s+",next_line.lstrip(' ')) #EFDC use space delimiting 
                            tref = float(params[2]) #TREF is 3rd parameter in the line
                    
                    #C9 - LWD
                    match_result = re.match('\s*C9 ', line)
                    if match_result is not None:
                        #first match is the heading, we need the second match which has the parameters
                        next_line = next(lines)
                        if "*" not in next_line:
                            params = re.split(r"\s+",next_line.lstrip(' ')) #EFDC use space delimiting 
                            lwd = int(params[6]) #LDW is 7th parameter in the line 
                    
                    #C70 - NSDUMP
                    match_result = re.match('\s*C70 ', line)
                    if match_result is not None:
                        #first match is the heading, we need the second match which has the parameters
                        next_line = next(lines)
                        if "*" not in next_line:
                            params = re.split(r"\s+",next_line.lstrip(' ')) #EFDC use space delimiting 
                            nsdump = int(params[2]) #LDW is 3rd parameter in the line
        except IOError as e:
            print('IOError occured: {0}'.format(e))
            raise e
    
        #Use EFDC parameters to calculate processing variables
        calculation_timestep = tref / ntsptc
        simulation_duration = ntc * tref
        output_interval = nsdump * calculation_timestep
        output_count = int(simulation_duration / output_interval)  
            
        self.cell_count = lwd
        self.output_count = output_count
        self.time_delta = datetime.timedelta(seconds=output_interval)

    def _read_bin_file(self, path_bin, bin_name):
        """
        Internal Method - Reads the binary file and returns as dataframe with a timeseries of the data

        EFDC binaries are FORTRAN 'unformmated files' When these files are created FORTRAN prepends a 4-byte header and appends an identical 
        4 byte footer for each write statement. Reading the binaries means triming off the header and footer and coverting the internal 4 byte 
        chucks into numerical values. 
        """
        print("Processing file {0}", bin_name)
        self.display_config(str.lower(bin_name))
        try:
            config = self._get_config(bin_name)
            self.cell_count = self.cell_count
            self.output_count = self.output_count 
            with open(path_bin,'rb') as f:
                contents = np.fromfile ( f, dtype= np.float32, count = -1 )
                if str.lower(config.quan_type) == "scalar":
                    array = contents.reshape((self.output_count,self.cell_count+2)).T #map bytes to array, Note +2 which accounts for header and footer
                    array = array[1:self.cell_count+1,:] #trim off first and last element which correspond to header and footer bytes 
                    data = float(config.conversion_factor) * array #apply conversion factor 
                else:
                    array = contents.reshape((self.output_count,self.cell_count*2+2)).T #vectors have both u & v component so we need double the number of internal bytes
                    u = array[1:self.cell_count+1] #vector u * v arrays are written sequentially
                    v = array[self.cell_count+1:self.cell_count*2+1] #these steps also trim the header and footer bytes
                    data = float(config.conversion_factor) * np.sqrt(u**2 + v**2) #combine vector components into single array   
            return(data)   
        except IOError as e:
            print('IOError occured: {0}'.format(e))
            raise e

    def _format_bin_array(self, array, bin_name):
        """
        Internal Method - Take the resultant numpy array from _read_bin_file and converts it to a pandas DataFrame object

        Takes the output array from 'process_bin_file' and coverts it into a pandas DataFrame with columns 
        'datetime', 'grid_no', and 'feild_name' (as specified in the _bin_config).
        """
        config = self._get_config(str.lower(bin_name))
        timesteps = [self.date_begin + self.time_delta * i for i in range(0, self.output_count)]
        array_grids = np.repeat(np.arange(1,self.cell_count+1), self.output_count)
        array_dates = np.tile(timesteps, self.cell_count)
        array_values = array.reshape(array.size)
        df_results = pd.DataFrame({'grid_no': array_grids,
                    'datetime': array_dates,
                    config.field_name: array_values
                    })
        return(df_results)

    def process_bin_file(self, path_bin):
        """
        Processes binary file and returns a Pandas DataFrame object with columns 'datetime', 'grid_no' and field name specified in binary config csv.
        
        Parameters:
            path_bin = File pathway to the binary file to be processed. 
        Returns:
            Pandas DataFrame object
        """
        bin_name = str(path_bin.split("\\")[-1:][0])
        array = self._read_bin_file(path_bin, bin_name)
        df = self._format_bin_array(array, bin_name)
        config = self._get_config(bin_name)      
        return (df)
