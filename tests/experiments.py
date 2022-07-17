import unittest
import os
import sys
import matplotlib.pyplot as plt
from numba import jit, njit, types
import numba
from dataclasses import dataclass
from collections import namedtuple


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

@dataclass
class PathwayClass:
    name: str
    value: float
    long_name: str
    units: str
    description: str

# Constituents
Pathway = namedtuple('Pathway', 'name value long_name units description')

def set_value(nt: namedtuple, value: float) -> namedtuple:
    return Pathway(name=nt.name, value=nt.value, long_name=nt.long_name, units=nt.units, description=nt.description)


# @jit(nopython=True, parallel=False)
@njit
def test():
    # aaa = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    # aaa: numba.types.ClassDataType = Parameter(name='DO', value=9.0, units='mg/L')
    q_net = Pathway(name='q_net', value=0.0, long_name='Net Solar Radiation', units='W/m2', description='')
    q_sediment = Pathway(name='q_sediment', value=0.0, long_name='Sediment Heat Flux', units='W/m2', description='')
    dTwaterCdt = Pathway(name='dTwaterCdt', value=0.0, long_name='Water Temperature Rate of Change', units='degC', description='')
    dTsedCdt = Pathway(name='dTsedCdt', value=0.0, long_name='Sediment Temperature Rate of Change', units='degC', description='')
    TwaterC = Pathway(name='TwaterC', value=0.0, long_name='Water Temperature', units='degC', description='Updated water temperature')

    q_net = set_value(q_net, 5.0)

    q_net = PathwayClass(name='q_net', value=0.0, long_name='Net Solar Radiation', units='W/m2', description='')

class Test_Experiments(unittest.TestCase):
    def setUp(self):
        pass

    def test_this(self):
        print('test_this')
        test()



if __name__ == '__main__':
    unittest.main()
