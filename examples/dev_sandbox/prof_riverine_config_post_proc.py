"""
Script to help with debugging Clearwater-Riverine.
"""
from pathlib import Path
from typing import (
    Optional,
    Tuple
)
import clearwater_riverine as cwr
from clearwater_riverine.postproc_util import _mass_bal_global

def run_clearwater_riverine_model(
    config_filepath: str | Path,
    datetime_range: Optional[Tuple[int, int]] = None,
):
    transport_model = cwr.ClearwaterRiverine(
        config_filepath=config_filepath,
        verbose=True,
        datetime_range=datetime_range,
    )

    for t in range(99):
        transport_model.update()
    
    mass_bal_df = _mass_bal_global(transport_model)

    return transport_model


if __name__ == '__main__':
    config_filepath = r'C:\Users\sjordan\OneDrive - LimnoTech\Documents\GitHub\ClearWater-riverine\examples\dev_sandbox\demo_config.yml'
    start_index = 0
    end_index = 100

    transport_model = run_clearwater_riverine_model(
        config_filepath=config_filepath,
        datetime_range=(start_index, end_index)
    )


