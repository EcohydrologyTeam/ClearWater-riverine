"""
Script to help with debugging Clearwater-Riverine.
"""
from pathlib import Path
import os
from typing import (
    Optional,
    Tuple
)
import clearwater_riverine as cwr
from clearwater_riverine.postproc_util import _mass_bal_global

def run_clearwater_riverine_model(
    hdf_file_path: str | Path,
    initial_condition_path: str | Path,
    boundary_condition_path: str | Path,
    config_filepath: str | Path,
    datetime_range: Optional[Tuple[int, int]] = None,
):
    transport_model = cwr.ClearwaterRiverine(
        hdf_file_path,
        diffusion_coefficient_input=0,
        config_filepath=config_filepath,
        verbose=True,
        datetime_range=datetime_range,
    )

    #transport_model.initialize(
    #    initial_condition_path=initial_condition_path,
    #    boundary_condition_path=boundary_condition_path,
    #    units='mg/m3'
    #)

    # transport_model.simulate_wq(save=False,
    #     input_mass_units = 'g',
    #     input_volume_units = 'm3',
    # )

    return transport_model


if __name__ == '__main__':
    hdf_name = 'clearWaterTestCases.p49.hdf'
    initial_condition_name = 'cwr_initial_conditions_fine_mesh.csv'
    boundary_condition_name = 'cwr_boundary_conditions_coriolis_creek.csv'
    root = Path(
        r'\\limno.com\files\AAO\AAOWorking\2ERDC12 - Clearwater\ClearwaterHECRAS_testCases\sumwereCreek_TSM_testing_timestep'
    )

    hdf_file_path = root / hdf_name
    initial_condition_path = root / initial_condition_name
    boundary_condition_path = root / boundary_condition_name

    time = 30
    start_index = int((60/time) * 60 * 12) # start six hours in
    #end_index = int((60 / time) * 60 * 24 * 2) + 1 # end
    end_index = start_index + 10
    print(start_index, end_index)

    # point to config
    dir_str = os.path.abspath("")
    print(dir_str)
    dir = Path(dir_str)
    dir_sandbox = dir / "examples" / "dev_sandbox"
    print(dir)
    config_filepath = dir_sandbox / "demo_config_single.yml"
    print(config_filepath)
    print(config_filepath.exists())

    transport_model = run_clearwater_riverine_model(
        hdf_file_path=hdf_file_path,
        initial_condition_path=initial_condition_path,
        boundary_condition_path=boundary_condition_path,
        config_filepath=config_filepath,
        datetime_range=(start_index, end_index)
    )

    print(f'Running {hdf_name}.')

    _mass_bal_global(transport_model, 'conservative_tracer')

