
from mesh import model_mesh, ClearWaterXarray

diffusion_coefficient_input = 0.1
m = model_mesh(0.1)

hdf_fpath = '../../examples/data/OhioRiver_m.p22.hdf'

m = m.cwr.read_ras(hdf_fpath)
m = m.cwr.calculate_required_parameters()
m.cwr.save_clearwater_xarray('../../examples/data_temp/ohio-river.zarr')

# print(m.mesh.attrs['face_normalunitvector_and_length'])