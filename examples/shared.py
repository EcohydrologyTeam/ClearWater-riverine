from pathlib import Path
import pandas as pd
import logging

def interpolate_to_model_timestep(meteo_df, model_dates, col_name):
    merged_df = pd.merge_asof(
        pd.DataFrame({'time': model_dates}),
        meteo_df,
        left_on='time',
        right_on='Datetime')
    merged_df[col_name.lower()] = merged_df[col_name].interpolate(method='linear')
    merged_df.drop(
        columns=['Datetime', col_name],
        inplace=True)
    merged_df.rename(
        columns={'time': 'Datetime'},
        inplace=True,
    )
    return merged_df

def process_meteo_data(
    meteo_file_path: Path,
    xarray_time_index,
    col_name: str,
):
    meteo_df = pd.read_csv(
        meteo_file_path,
        parse_dates=['Datetime']
    )
    meteo_df.dropna(axis=0, inplace=True)

    meteo_df_interp = interpolate_to_model_timestep(
        meteo_df,
        xarray_time_index,
        col_name
    )

    return meteo_df_interp  


def setup_function_logger(name):
    logger = logging.getLogger('function_logger')
    handler = logging.FileHandler(f'{name}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # Adjust the level based on your needs
    return logger