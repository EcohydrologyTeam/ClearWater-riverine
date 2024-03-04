from pathlib import Path
import holoviews as hv
import clearwater_riverine as cwr

def main(
    cwd: str | Path
):
    root = cwd / 'data/sumwere_test_cases/plan24_stormSurgeLow_crsMsh'
    flow_field_fpath = r'W:\2ERDC12 - Clearwater\ClearwaterHECRAS_testCases\sumwereCreek_TSM_testing_timestep\clearWaterTestCases.p24.hdf'
    boundary_condition_path = root / 'cwr_boundary_conditions_p24.csv'
    initial_condition_path = root / 'cwr_initial_conditions_p24.csv'

    transport_model = cwr.ClearwaterRiverine(
        flow_field_fpath,
        diffusion_coefficient_input=0.001,
        verbose=True,
        datetime_range=(4618, 4625)
        )
    
    transport_model.initialize(
        initial_condition_path=initial_condition_path,
        boundary_condition_path=boundary_condition_path,
        units='mg/m3',
    )

    for _ in range(len(transport_model.mesh.time) - 1):
        transport_model.update()


if __name__ == '__main__':
    script_path = Path(__file__).resolve().parent
    main(script_path)