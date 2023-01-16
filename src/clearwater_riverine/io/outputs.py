from pathlib import Path
from typing import Type, Union
import errno
import os 

import xarray as xr

# from utilities import MeshManager

class ZarrWriter:
    """Writes Zarr Output"""
    def write(self, mesh: xr.Dataset, output_file_path):
        mesh.to_zarr(
            output_file_path, 
            mode='w',
            consolidated=True
        )

class NetCDFWriter:
    """Writes NetCDF Output"""
    def write(self, mesh: xr.Dataset, output_file_path):
        mesh.to_netcdf(path=output_file_path)

class ClearWaterRiverineOutput:
    """Writes ClearWater Object to 

    Attributes:
        output_file_path (str): Filepath to location where user will save model mesh 
        output_file_name(str): Name of output mesh file 
    """
    def __init__(self, output_file_path: str, mesh: xr.Dataset) -> None:
        """Checks if output filepath exists"""
        self.output_file_path = output_file_path
        dir = Path(self.output_file_path).parents[0]
        if dir.is_dir() == False:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), output_file_path
            )
        self.mesh = mesh
    
    def write_mesh(self, writer: Union[Type[ZarrWriter], Type[NetCDFWriter]]) -> None:
        """Writes model output using specified writer
        Args:
            writer (ZarrWriter or NetCDFWriter): writer class. 
                Currently only supports writing to NetCDF or zarr.
        """
        writer.write(self.mesh, self.output_file_path)


class ClearWaterRiverineWriter:
    """Identify the concrete implementation of the output writer"""
    def write_mesh(self, writeable: Type[ClearWaterRiverineOutput], output_file_path: str) -> Type[ClearWaterRiverineOutput]:
        """Use the output file path, extension, and save as to identify the correct writer from the writing_factory
        Args:
            readable (RASInput): abstract interface implemented on any file we weant to read
            file_path (str):  Filepath to RAS output file
        """
        writer = writing_factory.get_writer(output_file_path)
        writeable.write_mesh(writer)
        return writeable

class ClearWaterRiverineOutputFactory:   
    """Creates factory to retrieve the correct writer
    Attributes:
        output_file_path (str): Filepath to location where user will save model mesh
        output_file_name (str): Name of output mesh file
        extension (str): Extension of RAS file path
    """           
    def get_writer(self, output_file_path) -> Union[Type[ZarrWriter], Type[NetCDFWriter]]:
        """ Retrieve the correct writer from the writing factory
        Args:
            output_file_path (str): Filepath to location where user will save model mesh
            output_file_name (str): Name of output mesh file
            save_as (str): Extension of RAS file path
        Returns:
            writer based on save_as extension. Currently only handles ZarrWriter or NetCDFWriter.
        """
        self.output_file_path = output_file_path
        self.extension = Path(self.output_file_path).suffix
        # self.output_file_name = output_file_name
        # self.extension = save_as
        if self.extension == '.zarr':
            return ZarrWriter()
        elif self.extension  == '.nc':
            return NetCDFWriter()
        else:
            raise ValueError(f"Cannot save as {self.output_extension}.")

writing_factory = ClearWaterRiverineOutputFactory()
