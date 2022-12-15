from pathlib import Path
import errno
import os
import xarray as xr 
import h5py

from hdf import HDFReader
from utilities import MeshManager

class RASInput:
    def __init__(self, file_path: str, mesh_manager: MeshManager):
        self.file_path = file_path
        if Path(self.file_path).is_file() == False:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), file_path
            )
        self.mesh_manager = mesh_manager
    
    def read_to_xarray(self, reader):
        reader.define_coordinates(self.mesh_manager)
        reader.define_topology(self.mesh_manager)
        reader.define_hydrodynamics(self.mesh_manager)
        reader.define_boundary_hydrodynamics(self.mesh_manager)


class RASReader:
    def read_to_xarray(self, readable, file_path):
        reader = factory.get_reader(file_path)
        readable.read_to_xarray(reader)
        return readable

class RASInputFactory:            
    def get_reader(self, file_path):
        self.file_path = file_path
        self.extension = Path(self.file_path).suffix
        if self.extension == '.hdf':
            return HDFReader(self.file_path)
        else:
            raise ValueError("File type is not accepted.")

factory = RASInputFactory()
