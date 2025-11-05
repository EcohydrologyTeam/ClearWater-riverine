import yaml
from pathlib import Path
from typing import Dict
from clearwater_data.variables import VariableRegistry
from clearwater_data.io.zarr import ZarrDataStore, ZarrDataSource
from clearwater_data.io.csv import CSVDataSource
from clearwater_data.io.base import DataSource, ChunkedDataSource
from clearwater_data.io.float import FloatDataSource
from clearwater_riverine.variables import DIFFUSION_COEFFICIENT

REQUIRED_CONFIG_KEYS = [
    'model',
    'constituents'
]

REQUIRED_CONSTITUENT_KEYS = [
    'initial_conditions',
    'boundary_conditions'
]

def init_from_config(
    config_filepath: str | Path,
    ):
    """Initializes Riverine from config"""
    config = read_config(config_filepath)

    model = config['model']
    constituents = config['constituents']

    data_sources = __init_data_sources(config)
    data_sources['variable_data_sources'][DIFFUSION_COEFFICIENT] = FloatDataSource(**{
        "value": model["diffusion_coefficient"]
    })

    return model, data_sources, constituents

def __init_single_data_source(source_name: str, source_config: dict) -> "DataSource":
    """Helper to initialize a single data source from its config."""
    if "|" in source_name:
        raise ValueError(
            f"Invalid source name: {source_name}. Source names cannot contain the '|' character."
        )

    provider_name = source_config["provider"].lower()

    if provider_name == "csv":
        return CSVDataSource(**source_config["data"])
    elif provider_name == "float":
        return FloatDataSource(**source_config["data"])
    else:
        raise ValueError(
            f"Unknown or unsupported provider `{provider_name}` for data_source `{source_name}`"
        )

def __init_data_sources(
        config: dict
):
    """Init all data sources from config file."""
    data_source: dict[str, DataSource | ChunkedDataSource] = {
        'boundary_conditions': {},
        'initial_conditions': {},
        'variable_data_sources': {},
    }    
    # Initialize all constituent data sources
    for source_name, source_config in config["constituents"].items():
        boundary_conditions = source_config["boundary_conditions"]
        data_source['boundary_conditions'][source_name] = __init_single_data_source(source_name, boundary_conditions)

        initial_conditions = source_config["initial_conditions"]
        data_source['initial_conditions'][source_name] = __init_single_data_source(source_name, initial_conditions)

    return data_source


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

def read_config(
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