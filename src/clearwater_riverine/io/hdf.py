from typing import Dict, Any

import h5py
import xarray as xr
# import variables
import numpy as np
import pandas as pd
import datetime

from clearwater_riverine import variables

def _hdf_internal_paths(project_name):
    """ Define HDF paths to relevant data"""
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
    """ Reads RAS hydrodynamic data required for WQ calculations in Clearwater Riverine Model from HDF file"""
    def __init__(self, file_path: str) -> None:
        """Opens HDF file and reads information required to set-up model mesh"""
        self.file_path = file_path
        self.infile = h5py.File(file_path, 'r')
        self.project_name = self.infile['Geometry/2D Flow Areas/Attributes'][()][0][0].decode('UTF-8')
        self.paths = _hdf_internal_paths(self.project_name)
    
    def define_coordinates(self, mesh: xr.Dataset):
        """Populate Coordinates and Dimensions"""
        # x-coordinates
        mesh = mesh.assign_coords(
            node_x=xr.DataArray(
            data = self.infile[self.paths[variables.NODE_X]][()].T[0],
            dims=('node',),
            )   
        )
        # y-coordinates
        mesh = mesh.assign_coords(
            node_y=xr.DataArray(
                data=self.infile[self.paths[variables.NODE_X]][()].T[1],
                dims=('node',),
            )
        )
        # time
        time_stamps_binary = self.infile[self.paths['binary_time_stamps']][()]

        # pandas is working faster than numpy for binary conversion
        time_stamps = pd.Series(time_stamps_binary).str.decode('utf8')
        xr_time_stamps = pd.to_datetime(time_stamps, format='%d%b%Y %H:%M:%S')

        mesh = mesh.assign_coords(
            time=xr.DataArray(
                data=xr_time_stamps,
                dims=('time',),
            )
        )
        return mesh

    def define_topology(self, mesh: xr.Dataset):
        """Define mesh topology """
        mesh[variables.FACE_NODES] = xr.DataArray(
            data=self.infile[f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes'][()],
            coords={
                "face_x": ('nface', self.infile[self.paths[variables.FACE_X]][()].T[0]),
                "face_y": ('nface', self.infile[self.paths[variables.FACE_Y]][()].T[1]),
            },
            dims=('nface', 'nmax_face'),
            attrs={
                'cf_role': 'face_node_connectivity',
                'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                'start_index': 0, 
                '_FillValue': -1
        })
        mesh[variables.EDGE_NODES] = xr.DataArray(
            data=self.infile[self.paths[variables.EDGE_NODES]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_node_connectivity',
                'long_name': 'Vertex nodes of mesh edges',
                'start_index': 0
            })
        mesh[variables.EDGE_FACE_CONNECTIVITY] = xr.DataArray(
            data=self.infile[self.paths[variables.EDGE_FACE_CONNECTIVITY]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_face_connectivity',
                'long_name': 'neighbor faces for edges',
                'start_index': 0
            })

    def define_hydrodynamics(self, mesh: xr.Dataset):
        """Populates hydrodynamic data in UGRID-compliant xarray."""
        mesh[variables.EDGES_FACE1] = _hdf_to_xarray(
            mesh['edge_face_connectivity'].T[0],
            ('nedge'),
            attrs={'Units':''}
        )  
        mesh[variables.EDGES_FACE2] = _hdf_to_xarray(
            mesh['edge_face_connectivity'].T[1], 
            ('nedge'),
            attrs={'Units':''}
        )
        
        nreal = mesh[variables.EDGE_FACE_CONNECTIVITY].T[0].values.max()
        mesh.attrs[variables.NUMBER_OF_REAL_CELLS] = nreal
        
        mesh[variables.FACE_SURFACE_AREA] = _hdf_to_xarray(
            self.infile[self.paths[variables.FACE_SURFACE_AREA]],
            ("nface")
        )
        mesh[variables.EDGE_VELOCITY] = _hdf_to_xarray(
            self.infile[self.paths[variables.EDGE_VELOCITY]], 
            ('time', 'nedge'), 
        )
        mesh[variables.EDGE_LENGTH] = _hdf_to_xarray(
            self.infile[self.paths[variables.EDGE_LENGTH]][:,2],
            ('nedge'), 
            attrs={'Units': 'ft'}
        )
        mesh[variables.WATER_SURFACE_ELEVATION] = _hdf_to_xarray(
            self.infile[self.paths[variables.WATER_SURFACE_ELEVATION]], 
            (['time', 'nface'])
        )
        try:
            mesh[variables.VOLUME] = _hdf_to_xarray(
                self.infile[self.paths[variables.VOLUME]], 
                ('time', 'nface')
            ) 
            # mesh[variables.VOLUME][:, mesh.attrs[variables.NUMBER_OF_REAL_CELLS]+1:] = 0 # revisit this
        except KeyError: 
            mesh.attrs['volume_calculation_required'] = True
            mesh.attrs['face_volume_elevation_info'] = _hdf_to_dataframe(self.infile[self.paths['volume elevation info']])
            mesh.attrs['face_volume_elevation_values'] = _hdf_to_dataframe(self.infile[self.paths['volume_elevation_values']])
        try:
            mesh[variables.FLOW_ACROSS_FACE] = _hdf_to_xarray(
                self.infile[self.paths[variables.FLOW_ACROSS_FACE]],
                ('time', 'nedge')
            )
        except:
            mesh.attrs['face_area_calculation_required'] = True
            mesh.attrs['face_area_elevation_info'] = _hdf_to_dataframe(self.infile[self.paths['area_elevation_info']])
            mesh.attrs['face_area_elevation_values'] = _hdf_to_dataframe(self.infile[self.paths['area_elevation_values']])
            mesh.attrs['face_normalunitvector_and_length'] = _hdf_to_dataframe(self.infile[self.paths['normalunitvector_length']])
            mesh.attrs['face_cell_indexes_df'] = _hdf_to_dataframe(self.infile[self.paths[variables.EDGE_FACE_CONNECTIVITY]])
        

    
    def define_boundary_hydrodynamics(self, mesh: xr.Dataset):
        """Read necessary information on hydrodynamics"""
        external_faces = pd.DataFrame(self.infile[self.paths['boundary_condition_external_faces']][()])
        attributes = pd.DataFrame(self.infile[self.paths['boundary_condition_attributes']][()])
        str_df = attributes.select_dtypes([object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            attributes[col] = str_df[col]
        boundary_attributes = attributes
        # merge attributes and boundary condition data 
        boundary_attributes['BC Line ID'] = boundary_attributes.index
        mesh.attrs['boundary_data'] = pd.merge(external_faces, boundary_attributes, on = 'BC Line ID', how = 'left')
        
    def close(self):
        """Close HDF file"""
        self.infile.close()

