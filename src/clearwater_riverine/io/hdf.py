from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    Union,
)

import h5py
import xarray as xr
# import variables
import numpy as np
import pandas as pd
from datetime import datetime
from clearwater_data.variables.xarray import DataArrayVariable

from clearwater_riverine.mesh import (
    instantiate_model_mesh,
    load_model_mesh
)
from clearwater_riverine.variables import (
    BOUNDARY_CONDITION_LINE_ID,
    BOUNDARY_FACE_INDEX,
    BOUNDARY_NAME,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    DIFFUSION_COEFFICIENT,
    NEDGE,
    NFACE,
    NODE_X,
    NODE_Y,
    TIME,
    FACE_NODES,
    EDGE_NODES,
    EDGE_FACE_CONNECTIVITY,
    FACE_X,
    FACE_Y,
    FACE_SURFACE_AREA,
    FACE_TO_FACE_DISTANCE,
    GATE_CONNECTIVITY,
    GATE_FLOW,
    EDGE_LENGTH,
    EDGE_VELOCITY,
    EDGE_VERTICAL_AREA,
    LOOKUP_ELEVATION,
    LOOKUP_VOLUME,
    LOOKUP_WETTED_SURFACE_AREA,
    WATER_SURFACE_ELEVATION,
    FLOW_ACROSS_FACE,
    VOLUME,
    VOLUME_ELEVATION_INFO,
    VOLUME_ELEVATION_VALUES,
    VOLUME_ELEVATION_LOOKUP,
    FACE_HYD_DEPTH,
    FACE_VEL_X,
    FACE_VEL_Y,
)

def _parse_attributes(dataset) -> Dict[str, Any]:
    """
    Parse the HDF5 attributes array,
    convert binary strings to Python strings,
    and return a dictionary of attributes.
    """
    attrs = {}
    for key, value in dataset.attrs.items():
        if isinstance(value, np.bytes_):
            attrs[key] = value.decode('ascii')
        elif isinstance(value, np.ndarray):
            values = []
            for v in value:
                if isinstance(v, np.bytes_):
                    values.append(v.decode('ascii'))
                else:
                    values.append(v)
            attrs[key] = values
        else:
            attrs[key] = value
    return attrs


def _hdf_to_xarray(
    dataset,
    dims,
    attrs=None,
    time_constraint: Optional[Tuple] = (None, None),
) -> xr.DataArray:
    """Read n-dimensional HDF5 dataset and return it as an xarray.DataArray"""
    if attrs is None:
        try:
            attrs = _parse_attributes(dataset)
        except AttributeError as e:
            attrs = {}
    if time_constraint != (None, None):
        data_to_read = dataset[()][time_constraint[0]: time_constraint[1]]
    else:
        data_to_read = dataset[()]
    data_array = xr.DataArray(
        data_to_read,
        dims=dims,
        attrs=attrs
    )
    return data_array


def _hdf_to_dataframe(dataset) -> pd.DataFrame:
    """Read n-dimensional HDF5 dataset and return it as an pandas DataFrame"""
    attrs = _parse_attributes(dataset)
    df = pd.DataFrame(
        dataset[()],
        columns=attrs['Column']
    )
    return df


class RASHDFDataSource:
    """
    Reads RAS hydrodynamic data required for WQ calculations.
    """
    def __init__(self, **kwargs) -> None:
        self.ras_hdf_path: str = kwargs.pop("ras_hdf_path")
        self.start_datetime: datetime = kwargs.pop("start_datetime", None)
        self.end_datetime: datetime = kwargs.pop("end_datetime", None)
        self.datetime_range = (self.start_datetime, self.end_datetime)
        self.calculated_variables = kwargs.pop("calculated_variables", {})

        self.mesh = instantiate_model_mesh()
        self.temporal_variables = {
            VOLUME: NFACE,
            EDGE_VELOCITY: NEDGE,
            WATER_SURFACE_ELEVATION: NFACE,
            FLOW_ACROSS_FACE: NEDGE,
        }
        self.static_variables = {
            FACE_SURFACE_AREA: NFACE,
            EDGE_LENGTH: NEDGE,
        }
        self.topology_variables = [
            FACE_X,
            FACE_Y,
            EDGE_FACE_CONNECTIVITY,
            FACE_NODES,
            NODE_X,
            NODE_Y,
        ]
        self.lookup_variables = [
            LOOKUP_ELEVATION,
            LOOKUP_VOLUME,
            LOOKUP_WETTED_SURFACE_AREA,
        ]
        self.boundary_variables = [
            BOUNDARY_CONDITION_LINE_ID,
            BOUNDARY_FACE_INDEX,
            BOUNDARY_NAME,
        ]

        # Add internal ones or modify as needed
        self.calculated_variables.update({
            EDGE_VERTICAL_AREA: True,
            FACE_TO_FACE_DISTANCE: True,
            COEFFICIENT_TO_DIFFUSION_TERM: True,
            CHANGE_IN_TIME: True,
        })

        ## TODO: add datetime validation somewhere
        # self.__validate_datetime_range()

        with h5py.File(self.ras_hdf_path, 'r') as infile:
            # set-up steps
            self.project_name = infile[
                'Geometry/2D Flow Areas/Attributes'
            ][()][0][0].decode('UTF-8')
            self.paths = self.__set_internal_paths()
            self.gate_names = self.__identify_gates(infile)
            self.__parse_datetimes(infile)

            # populate mesh
            self.__define_spatial_coordinates(infile)
            self.__define_topology(infile)
            self.__define_boundary_hydrodynamics(infile)
            self.__read_static_variables(infile)
            
            # populate lookup table
            self.volume_elevation_lookup = self.__create_lookup_xarray(infile)

            # gather additional data
            self.real_cell_count = self.mesh[EDGE_FACE_CONNECTIVITY].T[0].values.max() + 1


    def __identify_gates(
        self,
        infile,
    ):
        """Assesses if gate structures exist in HEC RAS model."""
        try:
            gate_names = list(infile[self.paths['gate_path']].keys())
        except KeyError:
            gate_names = None
        return gate_names


    def __set_internal_paths(self):
        """ Define HDF paths to relevant data"""
        return {
            NODE_X: f'Geometry/2D Flow Areas/{self.project_name}/FacePoints Coordinate',
            NODE_Y: f'Geometry/2D Flow Areas/{self.project_name}/FacePoints Coordinate',
            TIME: 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
            FACE_NODES: f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes',
            EDGE_NODES: f'Geometry/2D Flow Areas/{self.project_name}/Faces FacePoint Indexes',
            EDGE_FACE_CONNECTIVITY: f'Geometry/2D Flow Areas/{self.project_name}/Faces Cell Indexes',
            FACE_X: f'Geometry/2D Flow Areas/{self.project_name}/Cells Center Coordinate',
            FACE_Y: f'Geometry/2D Flow Areas/{self.project_name}/Cells Center Coordinate',
            FACE_SURFACE_AREA: f'Geometry/2D Flow Areas/{self.project_name}/Cells Surface Area',
            EDGE_VELOCITY: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Face Velocity',
            EDGE_LENGTH: f'Geometry/2D Flow Areas/{self.project_name}/Faces NormalUnitVector and Length',
            WATER_SURFACE_ELEVATION: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Water Surface',
            FLOW_ACROSS_FACE: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Face Flow',
            VOLUME: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Volume',
            FACE_HYD_DEPTH: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Hydraulic Depth',
            # Path scaffolding only: FACE_VEL_X / FACE_VEL_Y are NOT in
            # temporal_variables, so __read_temporal_variables never reads
            # them. Wiring them in is a 1-line add once their Phase-D
            # consumer (depth/velocity diffusion closure) is ported.
            FACE_VEL_X: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Velocity - Velocity X',
            FACE_VEL_Y: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{self.project_name}/Cell Velocity - Velocity Y',
            'project_name': 'Geometry/2D Flow Areas/Attributes',
            'binary_time_stamps': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
            'volume elevation info': f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Info',
            'volume_elevation_values': f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Values',
            'area_elevation_info': f'Geometry/2D Flow Areas/{self.project_name}/Faces Area Elevation Info',
            'area_elevation_values': f'Geometry/2D Flow Areas/{self.project_name}/Faces Area Elevation Values',
            'normalunitvector_length': f'Geometry/2D Flow Areas/{self.project_name}/Faces NormalUnitVector and Length',
            'boundary_condition_external_faces': 'Geometry/Boundary Condition Lines/External Faces',
            'boundary_condition_attributes': 'Geometry/Boundary Condition Lines/Attributes/',
            'boundary_condition_fixes': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Boundary Conditions',
            VOLUME_ELEVATION_INFO: f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Info',
            VOLUME_ELEVATION_VALUES: f'Geometry/2D Flow Areas/{self.project_name}/Cells Volume Elevation Values',
            'gate_path': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/SA 2D Area Conn',
        }


    def __parse_datetimes(
        self, 
        infile: h5py.File,
    ):
        """Date handling."""
        # time
        time_stamps_binary = infile[self.paths['binary_time_stamps']][()]

        # pandas is working faster than numpy for binary conversion
        ## TODO: figure out if it's faster to store all timesteps in memory and subset on each read
        ## OR if it's faster to do this binary conversion for each new chunk read.
        time_stamps = pd.Series(time_stamps_binary).str.decode('utf8')
        self.all_datetimes = pd.to_datetime(time_stamps, format='%d%b%Y %H:%M:%S')


    def __define_spatial_coordinates(
        self,
        infile: h5py.File
    ):
        """Populate Coordinates and Dimensions"""
        # x-coordinates
        self.mesh = self.mesh.assign_coords(
            node_x=xr.DataArray(
                data=infile[self.paths[NODE_X]][()].T[0],
                dims=('node',),
            )
        )
        # y-coordinates
        self.mesh = self.mesh.assign_coords(
            node_y=xr.DataArray(
                data=infile[self.paths[NODE_X]][()].T[1],
                dims=('node',),
            )
        )    


    def __define_topology(
        self,
        infile: h5py.File,
    ):
        """Define mesh topology """
        self.mesh[FACE_NODES] = xr.DataArray(
            data=infile[
                f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes'
            ][()],
            coords={
                "face_x": ('nface', infile[self.paths[FACE_X]][()].T[0]),
                "face_y": ('nface', infile[self.paths[FACE_Y]][()].T[1]),
            },
            dims=('nface', 'nmax_face'),
            attrs={
                'cf_role': 'face_node_connectivity',
                'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                'start_index': 0,
                '_FillValue': -1
            }
        )
        self.mesh[EDGE_NODES] = xr.DataArray(
            data=infile[self.paths[EDGE_NODES]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_node_connectivity',
                'long_name': 'Vertex nodes of mesh edges',
                'start_index': 0
            })
        self.mesh[EDGE_FACE_CONNECTIVITY] = xr.DataArray(
            data=infile[self.paths[EDGE_FACE_CONNECTIVITY]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_face_connectivity',
                'long_name': 'neighbor faces for edges',
                'start_index': 0
            })
        
        if self.gate_names is not None:
            connectivity_list = []
            for g in self.gate_names:
                headwater_cells = infile[
                    f"{self.paths['gate_path']}/{g}/HW TW Segments/Headwater Cells"][()].astype(int)
                tailwater_cells = infile[
                    f"{self.paths['gate_path']}/{g}/HW TW Segments/Tailwater Cells"][()].astype(int)
                gate_connectivity = np.stack((tailwater_cells, headwater_cells), axis=1)
                connectivity_list.append(gate_connectivity)
            
            connectivity_array = np.concatenate(connectivity_list, axis=0)

            self.mesh[GATE_CONNECTIVITY] = xr.DataArray(
                data=connectivity_array,
                dims=[GATE_CONNECTIVITY, '2'],
                attrs={
                    'long_name': 'cells connected by gates'
                }
            )

    def read(self, parameter_name:str) -> DataArrayVariable:
        return DataArrayVariable(self.__read(parameter_name))

    def __read(self, parameter_name: str):
        if parameter_name in self.mesh.data_vars:
            return self.mesh[parameter_name]
        else:
            self.__subset_datetimes(
                self.start_datetime,
                self.end_datetime,
            )
            self.__update_time_coordinate()
            self.__update_mesh()
            return self.mesh[parameter_name]
        
    def read_chunk(
        self,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> DataArrayVariable:
        return DataArrayVariable(
            self.__read_chunk(parameter_name, start_time, end_time),
            space_dimension=self.temporal_variables[parameter_name]
        )        

    def __read_chunk(
        self,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime
    ):
        if (
            "time" in self.mesh.coords
            and (self.mesh.time[0] == start_time)
            and (self.mesh.time[-1] == end_time)
            and (parameter_name in self.mesh.data_vars)
        ):
            return self.mesh[parameter_name]
        else:
            self.__subset_datetimes(
                start_time,
                end_time
            )
            self.__update_time_coordinate()
            self.__update_mesh()
            return self.mesh[parameter_name]
        
    
    def __update_mesh(
            self,
    ):
        with h5py.File(self.ras_hdf_path, 'r') as infile:
            self.__read_temporal_variables(infile)

    # def read() --> as variable DataArray variable

    def __subset_datetimes(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ):
        
        subset_dates = self.all_datetimes[
            (self.all_datetimes >= start_datetime) & (self.all_datetimes <= end_datetime)
        ]
        subset_indices = subset_dates.index.intersection(
            self.all_datetimes.index
        )
        self.datetime_range_indices: Tuple[int, int] = (
            subset_indices[0],
            subset_indices[-1] + 1
        )

        if self.datetime_range_indices != (None, None):
            self.datetime_subset = self.all_datetimes[
                    self.datetime_range_indices[0]:
                    self.datetime_range_indices[1]
                    ]
        else:
            self.datetime_subset = self.all_datetimes


    def __update_time_coordinate(
        self,  
    ):
        self.mesh = self.mesh.assign_coords(
            time=xr.DataArray(
                data=self.datetime_subset,
                dims=('time',),
            )
        )

    
    def __read_static_variables(
        self,
        infile: h5py.File,
    ):
        ## TODO: is this needed?
        self.mesh[FACE_SURFACE_AREA] = _hdf_to_xarray(
            infile[self.paths[FACE_SURFACE_AREA]],
            (NFACE)
        )
        self.nface = len(self.mesh[NFACE])

        self.mesh[EDGE_LENGTH] = _hdf_to_xarray(
            infile[self.paths[EDGE_LENGTH]][:, 2],
            ('nedge'),
        )
        self.nedge = len(self.mesh[NEDGE])

    def __read_temporal_variables(
        self,
        infile: h5py.File,
    ):
        for variable in self.temporal_variables.keys():
            hdf_path = self.paths[variable]
            if hdf_path not in infile:
                raise KeyError(
                    f"Required temporal variable '{variable}' is absent from "
                    f"the HEC-RAS HDF (expected dataset: '{hdf_path}'). This "
                    f"RAS run did not output it; fallback computation of this "
                    f"variable from geometry/volume is not yet ported on this "
                    f"branch."
                )
            self.mesh[variable] = _hdf_to_xarray(
                infile[hdf_path],
                ('time', self.temporal_variables[variable]),
                time_constraint=self.datetime_range_indices,
            )

        # add gate flows
        if self. gate_names is not None:
            flow_list = []
            for g in self.gate_names:
                gate_flow = infile[
                    f"{self.paths['gate_path']}/{g}/HW TW Segments/Flow"][()][:,0:-1] * -1
                flow_list.append(gate_flow)
            
            # flow_array = np.stack(flow_list, axis=1)
            flow_array = np.concatenate(flow_list, axis=1) 
            flow_array = flow_array[self.datetime_range_indices[0]: self.datetime_range_indices[1]]

            self.mesh[GATE_FLOW] = xr.DataArray(
                data=flow_array,
                dims=["time", GATE_FLOW],
                attrs={
                    'long_name': 'flow in cells'
                }
            )

    def __create_lookup_xarray(
        self,
        infile: h5py.File
    ):
        """Create volume elevation lookup xarray dataset."""
        volume_elevation_info_df = _hdf_to_dataframe(
            infile[self.paths[VOLUME_ELEVATION_INFO]]
            )
        volume_elevation_vals_df = _hdf_to_dataframe(
            infile[self.paths[VOLUME_ELEVATION_VALUES]]
        )
        # Define cells associated with each lookup value
        volume_elevation_vals_df['Cell'] = np.concatenate(
            [
                np.full(count, cell)
                for cell, count in zip(
                    volume_elevation_info_df.index,
                    volume_elevation_info_df['Count']
                )
            ]
        )

        # Create lookup dataset
        lookup_datasets = []

        for cell in volume_elevation_vals_df['Cell'].unique():
            cell_df = self.__create_cell_lookup_table(
                cell,
                volume_elevation_vals_df,
                infile,
            )
            cell_df = cell_df.rename(
                columns = {
                    "Elevation": LOOKUP_ELEVATION,
                    "Volume": LOOKUP_VOLUME,
                    "Wetted Surface Area": LOOKUP_WETTED_SURFACE_AREA,
                }
            )
            ds_cell = xr.Dataset.from_dataframe(cell_df)
            ds_cell = ds_cell.expand_dims({"nface": [cell]})
            lookup_datasets.append(ds_cell)

        return xr.concat(lookup_datasets, dim="nface", join="outer")

    def __create_cell_lookup_table(
        self,
        cell_no: int,
        df: pd.DataFrame,
        infile: h5py.File
    ) -> pd.DataFrame:
        """Create volume-elevation lookup table for each cell."""
        # Filter for single cell
        df_temp = df[df['Cell'] == cell_no]
        test_df = df_temp.copy().reset_index(drop=True)
        cell_surface_area = _hdf_to_dataframe(
            infile[self.paths[FACE_SURFACE_AREA]]
            )

        # Add row for flat cells (i.e., only one entry in lookup)
        # Create arbitrarily larger value
        increment_val = 0.01
        if len(test_df) == 1:
            # Preemptively add second row before any calculations
            new_row = test_df.iloc[0].copy()
            new_row['Elevation'] += increment_val
            new_row['Volume'] += increment_val
            test_df = pd.concat(
                [test_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

        # Compute differences in elevation and volume between adjacent rows
        # (i.e., vertical layers in the cell)
        test_df['Delta Elev'] = test_df['Elevation'].diff()
        test_df['Delta Volume'] = test_df['Volume'].diff()

        # Calculate the wetted surface area based on the volume and depth
        test_df['Surface Area'] = \
            test_df['Delta Volume'] / test_df['Delta Elev']

        # Average surface area between two elevation bands
        # Approximates wetted surface area between two elevatiosn
        test_df['Wetted Surface Area'] = \
            (test_df['Surface Area'] + test_df['Surface Area'].shift(-1)) / 2

        # Get maximum volume
        max_index = test_df['Volume'].idxmax()

        # Set edge cases (first and last slice)
        # Compare with total surface area for the cell as a whole
        cell_table = cell_surface_area[cell_surface_area.index == cell_no]
        input_value = cell_table['Surface Area'].values[0]
        # Set wetted surface area at the first row to 0 (i.e., first slice)
        test_df.at[max_index, 'Wetted Surface Area'] = input_value
        test_df.at[0, 'Wetted Surface Area'] = 0
        return test_df

    def __define_boundary_hydrodynamics(
        self,
        infile: h5py.File
    ):
        """Read necessary information on hydrodynamics."""
        # Pull important boundary information from the HDF file.
        external_faces = pd.DataFrame(
            infile[self.paths['boundary_condition_external_faces']][()]
        )
        attributes = pd.DataFrame(
            infile[self.paths['boundary_condition_attributes']][()]
        )
        list_of_boundaries = list(
            infile[self.paths['boundary_condition_fixes']].keys()
        )

        # Decode data
        str_df = attributes.select_dtypes([object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            attributes[col] = str_df[col]
        boundary_attributes = attributes

        # merge attributes and boundary condition data
        boundary_attributes['BC Line ID'] = boundary_attributes.index
        boundary_data = pd.merge(
            external_faces,
            boundary_attributes,
            on='BC Line ID',
            how='left'
        )

        # fix boundaries if needed
        boundary_data = self.__fix_boundary_hydrodynamics(
            boundary_data,
            infile
        )
        # store as boundary data
        self.boundary_data = (
            xr.Dataset.from_dataframe(boundary_data)
            # .set_coords(BOUNDARY_NAME)
            # .rename({BOUNDARY_NAME: 'RAS2D_TS_Name'})
        )

    def __fix_boundary_hydrodynamics(
        self,
        boundary_data: pd.DataFrame,
        infile: h5py.File
    ) -> pd.DataFrame:
        """
        Fixes a HEC-RAS bug in designating faces associated with
        boundary conditions.
        """
        df_ls = []

        # Identify correct boundary faces
        for boundary in boundary_data.Name.unique():
            fix_path = self.paths['boundary_condition_fixes']
            fpath = f"{fix_path}/{boundary} - Flow per Face"
            attrs = _parse_attributes(infile[fpath])
            boundary_faces_fix = attrs['Faces']
            boundary_faces_orig = boundary_data[
                (boundary_data.Name == boundary)]['Face Index']

            # compare with boundaries already identified
            # notify users if issues exist
            if set(boundary_faces_fix) != set(boundary_faces_orig):
                print(f'Extra boundary faces identified for {boundary}.')
                diff = set(boundary_faces_orig) - \
                    set(boundary_faces_fix)
                print(f'Removing erroneous boundaries {diff}.')

            # remove erroneous boundaries
            fixed_df = boundary_data[
                (boundary_data.Name == boundary) &
                (boundary_data['Face Index'].isin(boundary_faces_fix))
            ]
            df_ls.append(fixed_df)

        # remove any potential duplicates
        fixed_df_full = pd.concat(df_ls)
        fixed_df_full.drop(
            ['Station Start', 'Station End'],
            axis=1,
            inplace=True
        )
        fixed_df_full.drop_duplicates(inplace=True)

        return fixed_df_full


### SANDBOX:
### DERIVED VARIABLES
        # """Populates hydrodynamic data in UGRID-compliant xarray."""
        # mesh[EDGES_FACE1] = _hdf_to_xarray(
        #     mesh['edge_face_connectivity'].T[0],
        #     ('nedge'),
        #     attrs={'Units': ''}
        # )
        # mesh[EDGES_FACE2] = _hdf_to_xarray(
        #     mesh['edge_face_connectivity'].T[1],
        #     ('nedge'),
        #     attrs={'Units': ''}
        # )
        
        # nreal = mesh[EDGE_FACE_CONNECTIVITY].T[0].values.max()
        # mesh.attrs[NUMBER_OF_REAL_CELLS] = nreal
