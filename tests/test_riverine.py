import unittest
import os
import sys

'''
Set paths to the src directories of Riverine and TSM
'''

# Clearwater repo path
repo_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__))))
print(f'ClearWater repo path: {repo_path}') # ../../

# Riverine path
# riverine_path = os.path.join(repo_path, 'riverine', 'src', 'riverine')
riverine_path = os.path.join(repo_path, 'riverine', 'src')
print(f'Riverine source path: {riverine_path}') # ../../riverine/src
sys.path.append(riverine_path)

import riverine

class Test_Riverine(unittest.TestCase):
    def setUp(self):
        pass

    def test_riverine(self):
        print('test_riverine')
        modules = ['TSM']
        npts = 50  # Compute 50 cells
        nsteps = 1  # Just use one time point for now
        wq = riverine.Riverine(modules, npts, nsteps)
        wq.compute_water_quality()

if __name__ == '__main__':
    unittest.main()