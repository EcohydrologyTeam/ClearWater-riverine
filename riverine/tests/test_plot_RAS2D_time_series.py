import unittest
import os
import sys
import numpy
import datetime
import pyproj
import matplotlib
import pandas as pd

'''
Set path to the src directory
'''

local_path = os.path.dirname(os.path.realpath(__file__))
inputs_path = os.path.join(local_path, 'input_files')
outputs_path = os.path.join(local_path, 'output_files')
repo_path = os.path.join(local_path.split('ClearWater')[0], 'ClearWater')
src_path = os.path.join(repo_path, 'riverine', 'src', 'riverine', 'ras2d')
print(f'repo path: {repo_path}')
print(f'src path: {src_path}')
print(f'inputs path: {inputs_path}')
print(f'outputs path: {outputs_path}')

sys.path.insert(0, src_path)
import RAS2D

class Test_RAS2D_Muncie(unittest.TestCase):
    def setUp(self):
        # Open HEC-RAS 2D output file
        # Muncie, Indiana, East Indiana State Plane, NAD83, feet
        ras_file_path = os.path.join(inputs_path, 'Muncie.p04.hdf')
        self.ras2d_data = RAS2D.RAS_HDF5(ras_file_path, variables=[])
        self.ras2d_data.read()

        # Pick a better year, since Muncie uses 1900
        self.dates = [d + datetime.timedelta(days=120*365) for d in self.ras2d_data.results['dates']]

        # wkt = """PROJCS["NAD_1983_StatePlane_Indiana_East_FIPS_1301", GEOGCS["GCS_North_American_1983", DATUM["D_North_American_1983", SPHEROID["GRS_1980",6378137.0,298.257222101]], PRIMEM["Greenwich",0.0], UNIT["Degree",0.0174532925199433]], PROJECTION["Transverse_Mercator"], PARAMETER["False_Easting",100000.0], PARAMETER["False_Northing",250000.0], PARAMETER["Central_Meridian",-85.66666666666667], PARAMETER["Scale_Factor",0.9999666666666667], PARAMETER["Latitude_Of_Origin",37.5], UNIT["Meter",1.0]]"""
        # crs = pyproj.CRS(wkt)
        self.crs="epsg:2245"

        '''
        Other possible coordinate systems to use
        df = df.to_crs("EPSG:4326") # WGS 84
        df = df.to_crs("EPSG:32616") # UTM Zone 16 (feet)
        df = df.to_crs(epsg=3857) # Web Mercator
        '''

        # Create colormap similar to the HEC-RAS-2D water depth colormap
        colors = [
            (255, 255, 255),
            (98, 208, 230),
            (87, 188, 217),
            (70, 156, 195),
            (55, 123, 176),
            (38, 87, 160),
            (26, 57, 143),
            (24, 49, 143), 
            (16, 24, 130)
            ]
        position = numpy.array([0, 0.01, 1, 2, 3, 4, 5, 6, 7])/7.0
        position = position.tolist()
        self.ras_cmap = RAS2D.make_colormap(colors, eight_bit=True, position=position)

    def test_plot_RAS2D_chloropleth(self):
        print('test_plot_RAS2D')

        # Set output file path, which will be string-intepolated with the current index
        # as it loops through the data
        outfile_interp_path = os.path.join(outputs_path, 'muncie_chloropleth_%03d.png')

        # Plot the 2D time series

        # *** Just plot one figure:
        start_index = 250
        end_index = 251
        # **********************

        RAS2D.plot_ras2d(self.ras2d_data.geometry['elements_array'], self.ras2d_data.geometry['nodes_array'], 
            self.ras2d_data.results['depth'], self.dates, outfile_interp_path=outfile_interp_path,
            xmin=404000.0, xmax=413000.0, ymin=1800000.0, ymax=1806000.0,
            vmin=0.0, vmax=20.0, variable_name="Depths", 
            xlabel="Easting (ft)", ylabel="Northing (ft)", clabel="Depths (ft)",
            crs=self.crs, cmap=self.ras_cmap, alpha=0.7, start_index=start_index, end_index=end_index)

    def test_plot_RAS2D_quantiles(self):
        print('test_plot_RAS2D')

        # Set output file path, which will be string-intepolated with the current index
        # as it loops through the data
        outfile_interp_path = os.path.join(outputs_path, 'muncie_quantiles_%03d.png')
        self.ras_cmap = matplotlib.cm.get_cmap('Set1_r')

        ''' Plot the 2D time series '''

        # *** Just plot one figure:
        start_index = 250
        end_index = 251
        # **********************

        RAS2D.plot_ras2d(self.ras2d_data.geometry['elements_array'], self.ras2d_data.geometry['nodes_array'], 
            self.ras2d_data.results['depth'], self.dates, outfile_interp_path=outfile_interp_path,
            xmin=404000.0, xmax=413000.0, ymin=1800000.0, ymax=1806000.0,
            vmin=None, vmax=None, variable_name="Depths", 
            xlabel="Easting (ft)", ylabel="Northing (ft)", clabel="Depths (ft)",
            crs=self.crs, cmap=self.ras_cmap, alpha=0.7, scheme='Quantiles', start_index=start_index, end_index=end_index)


if __name__ == '__main__':
    unittest.main()