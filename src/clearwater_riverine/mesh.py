import sys
sys.path.insert(0, './io/')
import inputs
import outputs
import utilities 

class MeshPopulator:
    def __init__(self, diffusion_coefficient_input: float):
        self.mesh_manager = utilities.MeshManager(diffusion_coefficient_input)
    
    def read_ras(self, file_path):
        ras_data = inputs.RASInput(file_path, self.mesh_manager)
        reader = inputs.RASReader()
        reader.read_to_xarray(ras_data, file_path)
    
    def calculate_required_parameters(self):
        calculator = utilities.WQVariableCalculator(self.mesh_manager)
        calculator.calculate(self.mesh_manager)
        # return self.mesh_manager.mesh, self.mesh_manager.units
    
    def save_mesh(self, output_file_path, output_file_name, save_as):
        mesh_data = outputs.ClearWaterRiverineOutput(output_file_path, output_file_name, self.mesh_manager)
        writer = outputs.ClearWaterRiverineWriter()
        writer.write_mesh(mesh_data, output_file_path, output_file_name, save_as)