import clearwater_riverine as cwr
from pathlib import Path

project_path = Path.cwd()
example_path = project_path / 'examples'

#point to config
network_path = example_path / 'data_temp' / 'sumwere_creek_coarse_p48'
config_file = network_path / 'demo_config_p52.yml'
print(config_file.exists())

start_index =  int((8*60*60)/30)
end_index = 16*60*60
print(start_index, end_index)

transport_model = cwr.ClearwaterRiverine(
    config_filepath=config_file,
    verbose=True,
    datetime_range= (start_index, end_index)
)

for t in range(len(transport_model.mesh.time) - 1):
    transport_model.update()
