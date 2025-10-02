import clearwater_riverine as cwr
from pathlib import Path


#point to config
network_path = Path(r'C:\Users\sjordan\OneDrive - LimnoTech\Documents\GitHub\ClearWater-riverine\examples\data_temp\mississippi_model')

#point to config
config_file = network_path / 'demo_config_20s.yml'
print(config_file.exists())



# run small model - skip warmup 
warmup_time = 20 * 24
start_index = 2880 # 2880 # int((warmup_time * 60) / 10) 
hours_to_run = 10 * 24
end_index = start_index + 2 # start_index + 1300   # start_index + int(hours_to_run * 60 / 10) 

transport_model = cwr.ClearwaterRiverine(
    config_filepath=config_file,
    verbose=True,
    datetime_range= (start_index, end_index)
)


for t in range(len(transport_model.mesh.time) - 1):
    transport_model.update()




# upper_right = [
#     55911,
#     61463,
#     68931,
#     68440,
#     67958,
#     66987,
#     66001,
#     65504,
#     66496,
# ]