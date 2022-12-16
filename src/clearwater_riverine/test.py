
from mesh import MeshPopulator

diffusion_coefficient_input = 0.1
hdf_fpath = '../../examples/data/OhioRiver_m.p22.hdf'

# new framework - handle everything with MeshPopulator
mesh_data = MeshPopulator(0.1)
mesh_data.read_ras(hdf_fpath)
mesh, units = mesh_data.calculate_required_parameters()
print(mesh)


# old framework
# import sys
# sys.path.insert(0, './io/')
# from utilities import MeshManager, WQVariableCalculator
# import inputs
# mesh_data = MeshManager(diffusion_coefficient_input)
# ras_data = input.RASInput(hdf_fpath, mesh_data)
# reader = input.RASReader()

# print("Reading values...")
# reader.read_to_xarray(ras_data, hdf_fpath)

# print(mesh_data.mesh)

# print("Calculating values...")
# calculator = WQVariableCalculator(mesh_data)
# calculator.calculate(mesh_data)