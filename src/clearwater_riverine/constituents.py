from typing import Dict
from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from clearwater_riverine.linalg import RHS
from clearwater_riverine.variables import NUMBER_OF_REAL_CELLS


class Constituent:
    """Constituent class."""
    def __init__(
        self,
        name: str,
        constituent_config: Dict,
        mesh: xr.Dataset,
    ):
        self.name = name
        self.advection_mass_flux = np.zeros((len(mesh.time), len(mesh.nedge)))
        self.diffusion_mass_flux = np.zeros((len(mesh.time), len(mesh.nedge)))
        self.total_mass_flux = np.zeros((len(mesh.time), len(mesh.nedge)))
        self.input_array = np.zeros((len(mesh.time), len(mesh.nface)))
        # TODO: make units optional
        self.units = constituent_config['units']
        self.max_value = None
        self.min_value = None

        # add to model mesh
        mesh[self.name] = xr.DataArray(
            np.full(
                (len(mesh.time), len(mesh.nface)),
                np.nan
            ),
            dims = ('time', 'nface'),
            attrs = {
                'Units': f'{self.units}'
            }
        )

        # define initial and boundary conditions
        self.set_initial_conditions(
            filepath=constituent_config['initial_conditions'],
            mesh=mesh,
        )
        self.set_boundary_conditions(
            filepath=constituent_config['boundary_conditions'],
            mesh=mesh,
        )

        # set up RHS matrix
        self.b = RHS(
            mesh=mesh,
        )


    def set_initial_conditions(
        self,
        filepath: str | Path,
        mesh: xr.Dataset
    ):
        """Define initial conditions for costituents from CSV file. 

        Args:
            filepath (str): Filepath to a CSV containing initial conditions.
                The CSV should have two columns: one called `Cell_Index` and
                one called `Concentration`. The file should the concentration
                in each cell within the model domain at the first timestep. 
        """
        initial_condition_df = pd.read_csv(filepath)
        initial_condition_df['Cell_Index'] = initial_condition_df.Cell_Index.astype(int)
        self.input_array[0, [initial_condition_df['Cell_Index']]] =  initial_condition_df['Concentration']
        mesh[self.name].loc[
            {
                'time': mesh['time'][0],
            }
        ] = self.input_array[0]

    def set_boundary_conditions(
        self,
        filepath: str | Path,
        mesh: xr.Dataset,
    ):
        """Define boundary conditions for Clearwater Riverine model from a CSV file. 

        Args:
            filepath (str): Filepath to a CSV containing boundary conditions. 
            The CSV should have the following columns: `RAS2D_TS_Name` 
            (the timeseries name, as labeled in the HEC-RAS model), `Datetime`,
            `Concentration`. This file should contain the concentration for all
            relevant boundary cells at every RAS timestep. If a timestep / boundary
            cell is not included in this CSV file, the concentration will be set to 0
            in the Clearwater Riverine model.  
        """
        # Read in boundary condition data from user
        bc_df = pd.read_csv(
            filepath,
            parse_dates=['Datetime']
        )

        xarray_time_index = pd.DatetimeIndex(
            mesh.time.values
        )
        model_dataframe = pd.DataFrame({
            'Datetime': xarray_time_index,
            'Time Index': range(len(xarray_time_index))
        })

        result_df = pd.DataFrame()
        for boundary, group_df in bc_df.groupby('RAS2D_TS_Name'):
            # Merge with model timestep
            merged_group = pd.merge_asof(
                model_dataframe,
                group_df,
                on='Datetime'
            )
            # Interpolate
            merged_group['Concentration'] = merged_group['Concentration'].interpolate(
                method='linear'
            )
            # Append to dataframe
            result_df = pd.concat(
                [result_df, merged_group], 
                ignore_index=True
            )
        
        # Merge with boundary data
        boundary_df = pd.merge(
            result_df,
            mesh.boundary_data,
            left_on = 'RAS2D_TS_Name',
            right_on = 'Name',
            how='left'
        )
        boundary_df['Ghost Cell'] = mesh.edges_face2[boundary_df['Face Index'].to_list()]
        boundary_df['Domain Cell'] = mesh.edges_face1[boundary_df['Face Index'].to_list()]

        # Assign to appropriate position in array
        self.input_array[[boundary_df['Time Index']], [boundary_df['Ghost Cell']]] = boundary_df['Concentration']
    
    ## TODO: probably a more elegant way to do this
    def set_value_range(
        self,
        mesh: xr.Dataset
    ):
        self.max_value = int(mesh[self.name].sel(nface=slice(0, mesh.attrs[NUMBER_OF_REAL_CELLS])).max())
        self.min_value = int(mesh[self.name].sel(nface=slice(0, mesh.attrs[NUMBER_OF_REAL_CELLS])).min())

