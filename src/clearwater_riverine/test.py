import sys 
sys.path.insert(0, './io/')

from utilities import MeshManager
import input

diffusion_coefficient_input = 0.1
hdf_fpath = '../../examples/data/OhioRiver_m.p22.hdf'

mesh_data = MeshManager(diffusion_coefficient_input)
ras_data = input.RASInput(hdf_fpath, mesh_data)
reader = input.RASReader()
reader.read_to_xarray(ras_data, hdf_fpath)