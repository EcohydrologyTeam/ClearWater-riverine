import sys
sys.path.append(r"C:\Users\sjordan\OneDrive - LimnoTech\Documents\GitHub\ClearWater-data")

import clearwater_riverine as cwr
from pathlib import Path
import pandas as pd

# read config
working_dir = Path.cwd()
network_path = working_dir / 'examples/data_temp/sumwere_creek_coarse_p48'
config_file = network_path / 'demo_config.yml'

# instantiate model
transport_model = cwr.ClearwaterRiverine(
    config_filepath=config_file,
    verbose=True,
    start_datetime=pd.to_datetime('2022-05-13 08:00:00'),
    end_datetime=pd.to_datetime('2022-05-13 10:00:00'),
)

transport_model.registry.get('volume')
