import clearwater_riverine as cwr
from pathlib import Path
import pandas as pd
import geoviews as gv

# read config
working_dir = Path.cwd()
network_path = working_dir / 'examples/data_temp/sumwere_creek_coarse_p48'
config_file = network_path / 'example_config_refactor.yml'

# instantiate model
transport_model = cwr.ClearwaterRiverine(
    config_filepath=config_file,
)

transport_model.registry.get('water_temperature')
transport_model.registry.get('tracer')


transport_model.run()
