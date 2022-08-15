import h5py
import numpy as np
import pandas as pd
import xarray as xr
import numba
import datetime

# variable_paths = {
#     "dt": {
#         "dims": "time",
#         "path":f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Time Step',
#     },
#     "faces_surface_area": {
#         "dims": "nface",
#         "path": f'Geometry/2D Flow Areas/{project_name}/Cells Surface Area'
#     },
#     "edge_length": {
#         "dims": "nedge",
#         "path": 
#     },

# }


out["edge_length"] = xr.DataArray(
    data = ras2d_data.geometry['face_length'],
    dims = ("nedge"), 
    attrs={
        'units': 'feet' # will need to update units based on prj file
})


out["edge_velocity"] = xr.DataArray(
    data=ras2d_data.results['face_velocity'],
    dims=("time", 'nedge'),
    attrs={
        'units':'feet per second' # will need to update units based on prj file
    })

out["edge_vertical_area"] = xr.DataArray(
    data=face_areas_0,
    dims=("time", 'nedge'),
    attrs={
        'units':'feet per second' # will need to update units based on prj file
    })

}

def parse_attributes(dataset):
    '''Parse the HDF5 attributes array, convert binary strings to Python strings, and return a dictionary of attributes'''
    attrs = {}
    for key, value in dataset.attrs.items():
        if type(value) == np.bytes_:
            attrs[key] = value.decode('ascii')
        elif type(value) == np.ndarray:
            values = []
            for v in value:
                if type(v) == np.bytes_:
                    values.append(v.decode('ascii'))
                else:
                    values.append(v)
            attrs[key] = values
        else:
            attrs[key] = value
    return attrs

def read_dataset_xarray(dataset, dims) -> xr.DataArray:
    '''Read n-dimensional HDF5 dataset and return it as an xarray.DataArray'''
    attrs = parse_attributes(dataset)
    coords = {}
    data_array = xr.DataArray(dataset[()], dims = dims, attrs = attrs)
    return data_array

def parse_project_name(infile) -> str:
    '''Parse the name of a project'''
    project_name = infile['Geometry/2D Flow Areas/Attributes'][()][0][0].decode('UTF-8')
    return project_name



def define_ugrid(infile) -> xr.Datset:
    '''Define UGRID-compliant xarray'''

    # initialize mesh
    mesh = xr.Dataset()

    # initialize topology
    mesh["mesh2d"] = xr.DataArray(
        data=0,
        attrs={
            # required topology attributes
            'cf_role': 'mesh_topology',
            'long_name': 'Topology data of 2D mesh',
            'topology_dimension': 2,
            'node_coordinates': 'node_x node_y',
            'face_node_connectivity': 'face_nodes',
            # optionally required attributes
            'face_dimension': 'face',
            'edge_node_connectivity': 'edge_nodes',
            'edge_dimension': 'edge',
            # optional attributes 
            'face_edge_connectivity': 'face_edges',
            'face_face_connectivity': 'face_face_connectivity',
            'edge_face_connectivity': 'edge_face_connectivity',
            'boundary_node_connectivity': 'boundary_node_connectivity',
            'face_coordinates': 'face x face_y',
            'edge_coordinates': 'edge_x edge_y',
            },
        )

        # define project name 
    project_name = parse_project_name(infile)

    # assign coordinates 
    # x-coordinates
    mesh = mesh.assign_coords(
        node_x=xr.DataArray(
            data = infile[f'Geometry/2D Flow Areas/{project_name}/Cells FacePoint Indexes'].T[0],
            # data=[f[0] for f in ras2d_data.geometry['nodes_array']],
            dims=("node",),
            )   
        )
    # y-coordinates
    mesh = mesh.assign_coords(
            node_y=xr.DataArray(
            data=infile[f'Geometry/2D Flow Areas/{project_name}/Cells FacePoint Indexes'].T[1],
            dims=("node",),
        )
    )
    # time
    time_stamps_binary = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp'][()]
    time_stamps = [x.decode("utf8") for x in time_stamps_binary]
    mesh = mesh.assign_coords(
            time=xr.DataArray(
            data=[datetime.datetime.strptime(x, '%d%b%Y %H:%M:%S') for x in time_stamps], # '02JAN1900 22:55:00'
            dims=("time",),
                )
        )
        
    # define topology
    # face nodes 
    mesh["face_nodes"] = xr.DataArray(
        data=infile[f'Geometry/2D Flow Areas/{project_name}/Cells FacePoint Indexes'][()],
        coords={
            "face_x": ("nface", infile[f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate'][()].T[0]),
            # ("nface", [f[0] for f in ras2d_data.geometry['cells_center_coordinate']]),
            "face_y": ("nface", infile[f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate'][()].T[1]),
            # [f[1] for f in ras2d_data.geometry['cells_center_coordinate']]),
        },
        dims=("nface", "nmax_face"),
        attrs={
            'cf_role': 'face_node_connectivity',
            'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
            'start_index': 0, 
            '_FillValue': -1
    })
    # edge nodes 
    mesh["edge_nodes"] = xr.DataArray(
        data=infile[f'Geometry/2D Flow Areas/{project_name}/Faces FacePoint Indexes'][()],
        dims=("nedge", '2'),
        attrs={
            'cf_role': 'edge_node_connectivity',
            'long_name': 'Vertex nodes of mesh edges',
            'start_index': 0
        })
    # edge face connectivity
    mesh["edge_face_connectivity"] = xr.DataArray(
        data=infile[f'Geometry/2D Flow Areas/{project_name}/Faces Cell Indexes'][()],
        dims=("nedge", '2'),
        attrs={
            'cf_role': 'edge_face_connectivity',
            'long_name': 'neighbor faces for edges',
            'start_index': 0
        })
        
    return mesh



def populate_ugrid(infile):
    # initialize topology
    mesh = define_ugrid(infile)