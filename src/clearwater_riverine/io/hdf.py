from typing import Dict, Any

import h5py
import xarray as xr
import variables
import numpy as np
import pandas as pd
import datetime

from utilities import MeshManager


def _hdf_internal_paths(project_name):
    hdf_paths = {
        variables.NODE_X: f'Geometry/2D Flow Areas/{project_name}/FacePoints Coordinate',
        variables.NODE_Y: f'Geometry/2D Flow Areas/{project_name}/FacePoints Coordinate',
        variables.TIME: 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
        variables.FACE_NODES: f'Geometry/2D Flow Areas/{project_name}/Cells FacePoint Indexes',
        variables.EDGE_NODES: f'Geometry/2D Flow Areas/{project_name}/Faces FacePoint Indexes',
        variables.EDGE_FACE_CONNECTIVITY: f'Geometry/2D Flow Areas/{project_name}/Faces Cell Indexes',
        variables.FACE_X: f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate', 
        variables.FACE_Y: f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate',
        variables.FACE_SURFACE_AREA: f'Geometry/2D Flow Areas/{project_name}/Cells Surface Area',
        variables.EDGE_VELOCITY: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Velocity',
        variables.EDGE_LENGTH: f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length',
        variables.WATER_SURFACE_ELEVATION: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Water Surface',
        variables.FLOW_ACROSS_FACE: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Flow',
        variables.VOLUME: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Volume',
        'project_name': 'Geometry/2D Flow Areas/Attributes',
        'binary_time_stamps': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
        'volume elevation info': f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Info',
        'volume_elevation_values': f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Values',
        'area_elevation_info': f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Info',
        'area_elevation_values': f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Values',
        'normalunitvector_length': f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length',
        'boundary_condition_external_faces': 'Geometry/Boundary Condition Lines/External Faces',
        'boundary_condition_attributes': 'Geometry/Boundary Condition Lines/Attributes/',
    }
    return hdf_paths

def _parse_attributes(dataset) -> Dict[str, Any]:
    """Parse the HDF5 attributes array, convert binary strings to Python strings, and return a dictionary of attributes"""
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

def _hdf_to_xarray(dataset, dims, attrs=None) -> xr.DataArray:
    """Read n-dimensional HDF5 dataset and return it as an xarray.DataArray"""
    if attrs == None:
        attrs = _parse_attributes(dataset)
    data_array = xr.DataArray(dataset[()], dims = dims, attrs = attrs)
    return data_array

def _hdf_to_dataframe(dataset) -> pd.DataFrame:
    """Read n-dimensional HDF5 dataset and return it as an pandas DataFrame"""
    attrs = _parse_attributes(dataset)
    df = pd.DataFrame(dataset[()], columns = attrs['Column'])
    return df

class HDFReader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as infile:
            self.project_name = infile['Geometry/2D Flow Areas/Attributes'][()][0][0].decode('UTF-8')
            self.paths = _hdf_internal_paths(self.project_name)
    
    def define_coordinates(self, ras_data: MeshManager):
        """Populate Coordinates and Dimensions"""
        # x-coordinates
        with h5py.File(self.file_path, 'r') as infile:
            
            ras_data.mesh = ras_data.mesh.assign_coords(
                node_x=xr.DataArray(
                data = infile[self.paths[variables.NODE_X]][()].T[0],
                dims=('node',),
                )   
            )
            # y-coordinates
            ras_data.mesh = ras_data.mesh.assign_coords(
                node_y=xr.DataArray(
                    data=infile[self.paths[variables.NODE_X]][()].T[1],
                    dims=('node',),
                )
            )
            # time
            time_stamps_binary = infile[self.paths['binary_time_stamps']][()]
            time_stamps = [x.decode("utf8") for x in time_stamps_binary]
            ras_data.mesh = ras_data.mesh.assign_coords(
                time=xr.DataArray(
                data=[datetime.datetime.strptime(x, '%d%b%Y %H:%M:%S') for x in time_stamps],
                dims=('time',),
                )
            )

    def define_topology(self, ras_data: MeshManager):
        with h5py.File(self.file_path, 'r') as infile:
            ras_data.mesh[variables.FACE_NODES] = xr.DataArray(
                data=infile[f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes'][()],
                coords={
                    "face_x": ('nface', infile[self.paths[variables.FACE_X]][()].T[0]),
                    "face_y": ('nface', infile[self.paths[variables.FACE_Y]][()].T[1]),
                },
                dims=('nface', 'nmax_face'),
                attrs={
                    'cf_role': 'face_node_connectivity',
                    'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                    'start_index': 0, 
                    '_FillValue': -1
            })
            ras_data.mesh['edge_nodes'] = xr.DataArray(
                data=infile[self.paths[variables.EDGE_NODES]][()],
                dims=("nedge", '2'),
                attrs={
                    'cf_role': 'edge_node_connectivity',
                    'long_name': 'Vertex nodes of mesh edges',
                    'start_index': 0
                })
            ras_data.mesh['edge_face_connectivity'] = xr.DataArray(
                data=infile[self.paths[variables.EDGE_FACE_CONNECTIVITY]][()],
                dims=("nedge", '2'),
                attrs={
                    'cf_role': 'edge_face_connectivity',
                    'long_name': 'neighbor faces for edges',
                    'start_index': 0
                })

    def define_hydrodynamics(self, ras_data: MeshManager):
        """
        Populates data variables in UGRID-compliant xarray.

        Parameters:
            infile (h5py._hl.files.File):           HDF File containing RAS2D output 
            project_name (str):                     Name of the 2D Flow Area being modeled
            diffusion_coefficient_input (float):    User-defined diffusion coefficient for entire modeling domain. 

        Returns: 
            mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.

        Notes:
            This function requires cleanup. 
            Should remove some messy calculations and excessive code for vertical area calculations.         
        """
        with h5py.File(self.file_path, 'r') as infile:
            ras_data.mesh[variables.EDGES_FACE1] = _hdf_to_xarray(
                ras_data.mesh['edge_face_connectivity'].T[0],
                ('nedge'),
                attrs={'Units':''}
            )  
            ras_data.mesh[variables.EDGES_FACE2] = _hdf_to_xarray(
                ras_data.mesh['edge_face_connectivity'].T[1], 
                ('nedge'),
                attrs={'Units':''}
            )
            
            nreal = ras_data.mesh[variables.EDGE_FACE_CONNECTIVITY].T[0].values.max()
            ras_data.mesh.attrs[variables.NUMBER_OF_REAL_CELLS] = nreal
            
            ras_data.mesh[variables.FACE_SURFACE_AREA] = _hdf_to_xarray(
                infile[self.paths[variables.FACE_SURFACE_AREA]],
                ("nface")
            )
            ras_data.mesh[variables.EDGE_VELOCITY] = _hdf_to_xarray(
                infile[self.paths[variables.EDGE_VELOCITY]], 
                ('time', 'nedge')
            )
            ras_data.mesh[variables.EDGE_LENGTH] = _hdf_to_xarray(
                infile[self.paths[variables.EDGE_LENGTH]][:,2],
                ('nedge'), 
                attrs={'Units': 'ft'}
            )
            ras_data.mesh[variables.WATER_SURFACE_ELEVATION] = _hdf_to_xarray(
                infile[self.paths[variables.WATER_SURFACE_ELEVATION]], 
                (['time', 'nface'])
            )
            try:
                ras_data.mesh[variables.VOLUME] = _hdf_to_xarray(
                    infile[self.paths[variables.VOLUME]], 
                    ('time', 'nface')
                ) 
                ras_data.mesh[variables.VOLUME][:, ras_data.mesh.attrs[variables.NUMBER_OF_REAL_CELLS]+1:] = 0 # revisit this
            except KeyError: 
                ras_data.volume_calculation_required = True
                ras_data.face_volume_elevation_info = _hdf_to_dataframe(infile[self.paths['volume elevation info']])
                ras_data.face_volume_elevation_values = _hdf_to_dataframe(infile[self.paths['volume_elevation_values']])
            try:
                ras_data.mesh[variables.FLOW_ACROSS_FACE] = _hdf_to_xarray(
                    infile[self.paths[variables.FLOW_ACROSS_FACE]],
                    ('time', 'nedge')
                )
            except:
                ras_data.face_area_calculation_required = True
                ras_data.face_area_elevation_info = _hdf_to_dataframe(infile[self.paths['area_elevation_info']])
                ras_data.face_area_elevation_values = _hdf_to_dataframe(infile[self.paths['area_elevation_values']])
                ras_data.face_normalunitvector_and_length = _hdf_to_dataframe(infile[self.paths['normalunitvector_length']])
                ras_data.face_cell_indexes_df = _hdf_to_dataframe(infile[self.paths[variables.EDGE_FACE_CONNECTIVITY]])
        
    def define_boundary_hydrodynamics(self, ras_data: MeshManager):
        with h5py.File(self.file_path, 'r') as infile:
            external_faces = pd.DataFrame(infile[self.paths['boundary_condition_external_faces']][()])
            attributes = pd.DataFrame(infile[self.paths['boundary_condition_attributes']][()])
            str_df = attributes.select_dtypes([object])
            str_df = str_df.stack().str.decode('utf-8').unstack()
            for col in str_df:
                attributes[col] = str_df[col]
            boundary_attributes = attributes
            # merge attributes and boundary condition data 
            boundary_attributes['BC Line ID'] = boundary_attributes.index
            ras_data.boundary_data = pd.merge(external_faces, boundary_attributes, on = 'BC Line ID', how = 'left')
