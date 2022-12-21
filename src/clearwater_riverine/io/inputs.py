from pathlib import Path
from typing import Type, Union
import errno
import os

from hdf import HDFReader
from utilities import MeshManager

class RASInput:
    """Reads RAS input to an xarray

    Attributes:
        file_path (str): Filepath to RAS output file 
        mesh_manager (MeshManager): mesh manager containing the project mesh
            and other information required to perform advection-diffusion transport
            equations.
    """
    def __init__(self, file_path: str, mesh_manager: Type[MeshManager]) -> None:
        """ Checks if RAS filepath exists
        Args:
            file_path (str): Filepath to RAS output file 
            mesh_manager: mesh manager containing the project mesh
                and other information required to perform advection-diffusion transport
                equations.
        """
        self.file_path = file_path
        if Path(self.file_path).is_file() == False:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), file_path
            )
        self.mesh_manager = mesh_manager
    
    def read_to_xarray(self, reader: Type[HDFReader]) -> None:
        """Reads RAS output using appropriate reader
        Args:
            reader (HDFReader): reader class. Currently only supports reading HDF files.
        """
        reader.define_coordinates(self.mesh_manager)
        reader.define_topology(self.mesh_manager)
        reader.define_hydrodynamics(self.mesh_manager)
        reader.define_boundary_hydrodynamics(self.mesh_manager)


class RASReader:
    """Identify the concrete implementation of the RAS Reader"""
    def read_to_xarray(self, readable: Type[RASInput], file_path: str):
        """Use the RAS filepath to identify the correct reader from the reading_factory
        Args:
            readable (RASInput): abstract interface implemented on any file we weant to read
            file_path (str):  Filepath to RAS output file
        """
        reader = reading_factory.get_reader(file_path)
        readable.read_to_xarray(reader)
        return readable

class RASInputFactory:
    """Creates factory to retrieve the correct reader from the reading factory
    Attributes:
        file_path (str): RAS output file path
        extension (str): Extension of RAS file path
    """           
    def get_reader(self, file_path: str) -> Type[HDFReader]:
        """ Retrieve the correct reader from the reading factory
        Args:
            file_path (str): RAS output file path
        
        Returns:
            reader based on RAS filepath extension. Currently only handles HDFReader.
        """
        self.file_path = file_path
        self.extension = Path(self.file_path).suffix
        if self.extension == '.hdf':
            return HDFReader(self.file_path)
        else:
            raise ValueError("File type is not accepted.")

reading_factory = RASInputFactory()
