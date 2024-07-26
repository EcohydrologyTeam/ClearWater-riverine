from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import (
    Type,
    Union,
    Optional,
    Tuple
)
import errno
import os

import xarray as xr

from clearwater_riverine.io.hdf import HDFReader
# from mesh import ClearWaterMesh

class RASInput:
    """Reads RAS input to an xarray

    Attributes:
        file_path (str): Filepath to RAS output file 
        mesh_manager (MeshManager): mesh manager containing the project mesh
            and other information required to perform advection-diffusion transport
            equations.
    """
    def __init__(self, file_path: str, mesh: xr.Dataset) -> None:
        """ Checks if RAS filepath exists
        Args:
            file_path (str): Filepath to RAS output file 
            mesh: Clearwater Mesh containing the project mesh
                and other information required to perform advection-diffusion transport
                equations.
        """
        self.file_path = file_path
        if Path(self.file_path).is_file() == False:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                file_path
            )
        self.mesh = mesh
    
    def read_to_xarray(
        self,
        reader: Type[HDFReader],
    ) -> None:
        """Reads RAS output using appropriate reader
        Args:
            reader (HDFReader): reader class. Currently only supports reading HDF files.
        """
        self.mesh = reader.define_coordinates(self.mesh)
        reader.define_topology(self.mesh)
        reader.define_hydrodynamics(self.mesh)
        reader.define_boundary_hydrodynamics(self.mesh)
        reader.close()
        


class RASReader:
    """Identify the concrete implementation of the RAS Reader"""
    def read_to_xarray(
        self,
        readable: Type[RASInput],
        file_path: str,
        datetime_range: Optional[Tuple[int, int] | Tuple[str, str]] = None
    ) -> None:
        """Use the RAS filepath to identify the correct reader from the reading_factory
        Args:
            readable (RASInput): abstract interface implemented on any file we weant to read
            file_path (str):  Filepath to RAS output file
        """
        reader = reading_factory.get_reader(
            file_path,
            datetime_range=datetime_range,
        )
        readable.read_to_xarray(reader)
        return readable

class RASInputFactory:
    """Creates factory to retrieve the correct reader from the reading factory
    Attributes:
        file_path (str): RAS output file path
        extension (str): Extension of RAS file path
    """           
    def get_reader(
        self,
        file_path: str,
        datetime_range: Optional[Tuple[int, int] | Tuple[str, str]] = None
) -> Type[HDFReader]:
        """ Retrieve the correct reader from the reading factory
        Args:
            file_path (str): RAS output file path
        
        Returns:
            reader based on RAS filepath extension. Currently only handles HDFReader.
        """
        self.file_path = file_path
        self.extension = Path(self.file_path).suffix
        if self.extension == '.hdf':
            return HDFReader(
                self.file_path,
                datetime_range=datetime_range
            )
        else:
            raise ValueError("File type is not accepted.")

reading_factory = RASInputFactory()

class ZarrLoader:
    """Loads Zarr Output"""
    def load(mesh_file_path: str | Path):
        return xr.open_zarr(
            mesh_file_path
        )

class NetCDFLoader:
    """Loads NetCDF Output"""
    def load(mesh_file_path: str | Path):
        return xr.open_dataset(
            mesh_file_path,
            engine='netcdf4'
        )
class ClearWaterRiverineLoader:
    """Loads Clearwater Riverine mesh

    Attributes:
        mesh_file_path (str | Path): filepath to Clearwater Riverine output file
    """
    def __init__(self, mesh_file_path: str) -> None:
        """Checks if output filepath exists"""
        self.mesh_file_path = mesh_file_path
        dir = Path(self.output_file_path).parents[0]
        if dir.is_dir() == False:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), mesh_file_path
            )
    
    def load_mesh(
        self,
        loader: Union[Type[ZarrLoader], Type[NetCDFLoader]]
    ) -> None:
        """Loads model output using specified loader
        Args:
            looader (ZarrLoader or NetCDFLoader): loader class. 
                Currently only supports loading from NetCDF or zarr.
        """
        loader.load(self.mesh_file_path)

class ClearWaterRiverineLoadingFactory:
    """
    Creates factory to retrieve the correct reader from the loading factory.
    """
    def get_loader(
        self,
        mesh_file_path: str,
    ) -> Union[Type[ZarrLoader], Type[NetCDFLoader]]:
        self.mesh_file_path = mesh_file_path
        self.extension = Path(self.mesh_file_path).suffix
        if self.extension == '.zarr':
            return ZarrLoader()
        elif self.extension  == '.nc':
            return NetCDFLoader()
        else:
            raise ValueError(f"Cannot save as {self.output_extension}.")
    
loading_factory = ClearWaterRiverineLoadingFactory()
