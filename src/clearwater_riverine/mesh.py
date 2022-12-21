import sys
sys.path.insert(0, './io/')
import inputs
import outputs
import utilities 

class MeshPopulator:
    """Populates the mesh with required information for water quality calculations
    Attributes:
        mesh_manager (MeshManager): mesh manager containing the project mesh
            and other information required to perform advection-diffusion transport
            equations.
    """
    def __init__(self, diffusion_coefficient_input: float) -> None:
        """Initializes mesh manager
        Args:
            diffusion_coefficient_input (float): User-defined diffusion coefficient for entire modeling domain. 
        """
        self.mesh_manager = utilities.MeshManager(diffusion_coefficient_input)
    
    def read_ras(self, file_path: str) -> None:
        """Read information in RAS output file to the mesh
        Args:
            file_path (str): RAS output filepath
        """
        ras_data = inputs.RASInput(file_path, self.mesh_manager)
        reader = inputs.RASReader()
        reader.read_to_xarray(ras_data, file_path)
    
    def calculate_required_parameters(self) -> None:
        """Calculate additional values required for advection-diffusion transport equation"""
        calculator = utilities.WQVariableCalculator(self.mesh_manager)
        calculator.calculate(self.mesh_manager)
        # return self.mesh_manager.mesh, self.mesh_manager.units
    
    def save_mesh(self, output_file_path: str, output_file_name: str, save_as: str) -> None:
        """Save mesh
        Args:
            output_file_path (str): path to folder where output will be saved
            output_file_name (str): name of output file 
            save_as (str): format to save output 
        """
        mesh_data = outputs.ClearWaterRiverineOutput(output_file_path, output_file_name, self.mesh_manager)
        writer = outputs.ClearWaterRiverineWriter()
        writer.write_mesh(mesh_data, output_file_path, output_file_name, save_as)