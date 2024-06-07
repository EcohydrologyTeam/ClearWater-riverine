import yaml
from pathlib import Path
from typing import Dict

REQUIRED_CONFIG_KEYS = [
    'diffusion_coefficient',
    'flow_field_filepath',
    'constituents'
]

REQUIRED_CONSTITUENT_KEYS = [
    'initial_conditions',
    'boundary_conditions'
]


def validate_config(
    config: Dict,
):
    missing_keys = []
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            missing_keys.append(key)
    return missing_keys

def validate_constituents(config):
    for constituent in config['constituents']:
        for key in REQUIRED_CONSTITUENT_KEYS:
            if key not in config['constituents'][constituent]:
                return (key, constituent)
    return None

def parse_config(
    config_filepath: str | Path,
):
    with open(config_filepath, 'r') as file:
        model_config = yaml.safe_load(file)

    # ensure required information is in config
    missing_keys = validate_config(model_config)
    if len(missing_keys) > 0:
        raise ValueError(f'Missing {missing_keys} from configuration file.')

    cd = validate_constituents(model_config)
    if cd is not None:
        raise ValueError(f'Missing {cd[0]} from {cd[1]} in the model config.')
    
    return model_config