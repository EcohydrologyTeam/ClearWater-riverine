import clearwater_riverine as cwr
from pathlib import Path
import pandas as pd
# import geoviews as gv

# read config
working_dir = Path.cwd()
network_path = working_dir / 'examples/data_temp/sumwere_creek_coarse_p48'
config_file = network_path / 'example_config_refactor_no_chunk.yml'

# instantiate model
transport_model = cwr.ClearwaterRiverine(
    config_filepath=config_file,
)

# test getting variables
transport_model.registry.get('tracer')

transport_model.run()

print(transport_model.registry.get('tracer_mass_flux'))

mass_balance_model = transport_model.calculate_mass_balance(constituent_name='tracer')
mass_balance_answer = transport_model.calculate_mass_balance(
    constituent_name='tracer',
    calculate_answer=True,
    answer_value=100
)


mass_balance_model
