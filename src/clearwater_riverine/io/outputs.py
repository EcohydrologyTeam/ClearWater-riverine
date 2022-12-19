from pathlib import Path
import errno
import os 

import xarray as xr

from utilities import MeshManager

class ZarrWriter:
    def write(self, mesh_data: MeshManager, output_file_path, output_file_name):
        mesh_data.mesh.to_zarr(
            f'{output_file_path}/{output_file_name}.zarr', 
            mode='w', 
            consolidated=True
        )

class NetCDFWriter:
    def write(self, mesh_data: MeshManager, output_file_path, output_file_name):
        mesh_data.mesh.to_netcdf(path=f'{output_file_path}/{output_file_name}.nc')

class ClearWaterRiverineOutput:
    def __init__(self, output_file_path: str, output_file_name: str, mesh_manager: MeshManager):
        self.output_file_path = output_file_path
        self.output_file_name = output_file_name
        if Path(self.output_file_path).is_dir() == False:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), output_file_path
            )
        self.mesh_manager = mesh_manager
    
    def write_mesh(self, writer):
        writer.write(self.mesh_manager, self.output_file_path, self.output_file_name)


class ClearWaterRiverineWriter:
    def write_mesh(self, writeable, output_file_path, output_file_name, save_as):
        writer = writing_factory.get_writer(output_file_path, output_file_name, save_as)
        writeable.write_mesh(writer)
        return writeable

class ClearWaterRiverineOutputFactory:            
    def get_writer(self, output_file_path, output_file_name, save_as):
        self.output_file_path = output_file_path
        self.output_file_name = output_file_name
        self.extension = save_as
        if save_as == 'zarr':
            return ZarrWriter()
        elif save_as  == 'nc':
            return NetCDFWriter()
        else:
            raise ValueError(f"Cannot save as {self.output_extension}.")

writing_factory = ClearWaterRiverineOutputFactory()
