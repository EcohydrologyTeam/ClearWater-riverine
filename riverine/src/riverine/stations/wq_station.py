import pandas as pd
import matplotlib.pyplot as plt
import os, sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, local_path)

from station import Station


class WaterQualityStation(Station):
    '''
    Station: reads, writes, and processes water quality data.
    It holds any number of arbitrary parameters. This module 
    assumes that all data are in SI units.
    '''

    def __init__(self, latitude: float, longitude: float, elevation: float):
        super().__init__(latitude=latitude, longitude=longitude, elevation=elevation)
