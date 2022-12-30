import sys

import xarray as xr
import pandas as pd

sys.path.insert(0, './io/')
import inputs
import outputs
import utilities 

def model_mesh(diffusion_coefficient_input: float) -> xr.Dataset:
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

'''
class ClearWaterMesh:   
    def __init__(self, diffusion_coefficient_input: float):
        self.mesh = xr.Dataset()
        self.mesh["mesh2d"] = xr.DataArray(
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
        self.mesh.attrs['diffusion_coefficient'] = diffusion_coefficient_input
        self.mesh.attrs['volume_calculation_required'] = False 
        self.mesh.attrs['face_area_calculation_required'] = False
        self.mesh.attrs['face_area_elevation_info'] = pd.DataFrame()
        self.mesh.attrs['face_area_elevation_values'] = pd.DataFrame()
        self.mesh.attrs['face_normalunitvector_and_length'] = pd.DataFrame()
        self.mesh.attrs['face_cell_indexes_df'] = pd.DataFrame()
        self.mesh.attrs['face_volume_elevation_info'] = pd.DataFrame()
        self.mesh.attrs['face_volume_elevation_values'] = pd.DataFrame()
        self.mesh.attrs['boundary_data'] = pd.DataFrame()
        self.mesh.attrs['units'] = "Unknown"

    def get_mesh(self):
        return self.mesh

    def read_ras(self, file_path: str) -> None:
        """Read information in RAS output file to the mesh
        Args:
            file_path (str): RAS output filepath
        """
        ras_data = inputs.RASInput(file_path, self.mesh)
        reader = inputs.RASReader()
        reader.read_to_xarray(ras_data, file_path)

    def calculate_required_parameters(self) -> None:
        """Calculate additional values required for advection-diffusion transport equation"""
        calculator = utilities.WQVariableCalculator(self.mesh)
        calculator.calculate(self.mesh)
        # return self.mesh_manager.mesh, self.mesh_manager.units
    
    def save_mesh(self, output_file_path: str, output_file_name: str, save_as: str) -> None:
        """Save mesh
        Args:
            output_file_path (str): path to folder where output will be saved
            output_file_name (str): name of output file 
            save_as (str): format to save output 
        """
        mesh_data = outputs.ClearWaterRiverineOutput(output_file_path, output_file_name, self.mesh)
        writer = outputs.ClearWaterRiverineWriter()
        writer.write_mesh(mesh_data, output_file_path, output_file_name, save_as)
'''


# class MeshPopulator:
#     """Populates the mesh with required information for water quality calculations
#     Attributes:
#         mesh_manager (MeshManager): mesh manager containing the project mesh
#             and other information required to perform advection-diffusion transport
#             equations.
#     """
#     def __init__(self, diffusion_coefficient_input: float) -> None:
#         """Initializes mesh manager
#         Args:
#             diffusion_coefficient_input (float): User-defined diffusion coefficient for entire modeling domain. 
#         """
#         self.mesh_manager = utilities.MeshManager(diffusion_coefficient_input)
    
#     def read_ras(self, file_path: str) -> None:
#         """Read information in RAS output file to the mesh
#         Args:
#             file_path (str): RAS output filepath
#         """
#         ras_data = inputs.RASInput(file_path, self.mesh_manager)
#         reader = inputs.RASReader()
#         reader.read_to_xarray(ras_data, file_path)
    
#     def calculate_required_parameters(self) -> None:
#         """Calculate additional values required for advection-diffusion transport equation"""
#         calculator = utilities.WQVariableCalculator(self.mesh_manager)
#         calculator.calculate(self.mesh_manager)
#         # return self.mesh_manager.mesh, self.mesh_manager.units
    
#     def save_mesh(self, output_file_path: str, output_file_name: str, save_as: str) -> None:
#         """Save mesh
#         Args:
#             output_file_path (str): path to folder where output will be saved
#             output_file_name (str): name of output file 
#             save_as (str): format to save output 
#         """
#         mesh_data = outputs.ClearWaterRiverineOutput(output_file_path, output_file_name, self.mesh_manager)
#         writer = outputs.ClearWaterRiverineWriter()
#         writer.write_mesh(mesh_data, output_file_path, output_file_name, save_as)