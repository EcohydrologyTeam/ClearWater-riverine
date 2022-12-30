import sys

import xarray as xr
import pandas as pd

sys.path.insert(0, './io/')
import inputs
import outputs
import utilities 

def model_mesh(diffusion_coefficient_input: float) -> xr.Dataset:
    """
    Initialize the Clearwater Model Mesh

    Args:
        diffusion_coefficient_input (float):    User-defined diffusion coefficient for entire modeling domain. 

    Returns:
        ds (xr.Dataset): xarray dataset initialized with UGRID-compliant CF conventions
    """
    ds = xr.Dataset()
    ds["mesh2d"] = xr.DataArray(
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
    ds.attrs = {
        'Conventions': 'CF-1.8 UGRID-1.0 Deltares-0.10',
        'diffusion_coefficient': diffusion_coefficient_input}

    return ds

@xr.register_dataset_accessor("cwr")
class ClearWaterXarray:
    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj
        self._obj.attrs['volume_calculation_required'] = False 
        self._obj.attrs['face_area_calculation_required'] = False
        self._obj.attrs['face_area_elevation_info'] = pd.DataFrame()
        self._obj.attrs['face_area_elevation_values'] = pd.DataFrame()
        self._obj.attrs['face_normalunitvector_and_length'] = pd.DataFrame()
        self._obj.attrs['face_cell_indexes_df'] = pd.DataFrame()
        self._obj.attrs['face_volume_elevation_info'] = pd.DataFrame()
        self._obj.attrs['face_volume_elevation_values'] = pd.DataFrame()
        self._obj.attrs['boundary_data'] = pd.DataFrame()
        self._obj.attrs['units'] = "Unknown"

    def read_ras(self, file_path: str) -> xr.Dataset:
        """Read information in RAS output file to the mesh
        Args:
            file_path (str): RAS output filepath
        """
        ras_data = inputs.RASInput(file_path, self._obj)
        reader = inputs.RASReader()
        reader.read_to_xarray(ras_data, file_path)
        self._obj = ras_data.mesh
        return self._obj

    def calculate_required_parameters(self) -> xr.Dataset:
        """Calculate additional values required for advection-diffusion transport equation"""
        calculator = utilities.WQVariableCalculator(self._obj)
        calculator.calculate(self._obj)
        return self._obj

    def save_clearwater_xarray(self, output_file_path: str) -> None:
        """Save mesh
        Args:
            output_file_path (str): name of file path to save clearwater output
        """
        # must delete certain attributes before saving 
        del self._obj.attrs['face_area_elevation_info']
        del self._obj.attrs['face_area_elevation_values']
        del self._obj.attrs['face_normalunitvector_and_length']
        del self._obj.attrs['face_cell_indexes_df']
        del self._obj.attrs['face_volume_elevation_info']
        del self._obj.attrs['face_volume_elevation_values']
        del self._obj.attrs['boundary_data']

        # write output
        mesh_data = outputs.ClearWaterRiverineOutput(output_file_path, self._obj)
        writer = outputs.ClearWaterRiverineWriter()
        writer.write_mesh(mesh_data, output_file_path)
