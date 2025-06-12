from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
)

import h5py
import xarray as xr
# import variables
import numpy as np
import pandas as pd

from clearwater_riverine.variables import (
    NODE_X,
    NODE_Y,
    TIME,
    FACE_NODES,
    EDGE_NODES,
    EDGES_FACE1,
    EDGES_FACE2,
    EDGE_FACE_CONNECTIVITY,
    FACE_X,
    FACE_Y,
    FACE_SURFACE_AREA,
    EDGE_VELOCITY,
    EDGE_LENGTH,
    WATER_SURFACE_ELEVATION,
    FLOW_ACROSS_FACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
    VOLUME_ELEVATION_INFO,
    VOLUME_ELEVATION_VALUES,
    VOLUME_ELEVATION_LOOKUP,
    FACE_HYD_DEPTH,
    FACE_VEL_X,
    FACE_VEL_Y,
    FACE_VEL_MAG,
)


def _hdf_internal_paths(project_name):
    """ Define HDF paths to relevant data"""
    hdf_paths = {
        NODE_X: f'Geometry/2D Flow Areas/{project_name}/FacePoints Coordinate',
        NODE_Y: f'Geometry/2D Flow Areas/{project_name}/FacePoints Coordinate',
        TIME: 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
        FACE_NODES: f'Geometry/2D Flow Areas/{project_name}/Cells FacePoint Indexes',
        EDGE_NODES: f'Geometry/2D Flow Areas/{project_name}/Faces FacePoint Indexes',
        EDGE_FACE_CONNECTIVITY: f'Geometry/2D Flow Areas/{project_name}/Faces Cell Indexes',
        FACE_X: f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate',
        FACE_Y: f'Geometry/2D Flow Areas/{project_name}/Cells Center Coordinate',
        FACE_SURFACE_AREA: f'Geometry/2D Flow Areas/{project_name}/Cells Surface Area',
        EDGE_VELOCITY: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Velocity',
        EDGE_LENGTH: f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length',
        WATER_SURFACE_ELEVATION: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Water Surface',
        FLOW_ACROSS_FACE: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Flow',
        VOLUME: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Volume',
        FACE_HYD_DEPTH: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Hydraulic Depth',
        FACE_VEL_X: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Velocity - Velocity X',
        FACE_VEL_Y: f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Velocity - Velocity Y',
        'project_name': 'Geometry/2D Flow Areas/Attributes',
        'binary_time_stamps': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp',
        'volume elevation info': f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Info',
        'volume_elevation_values': f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Values',
        'area_elevation_info': f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Info',
        'area_elevation_values': f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Values',
        'normalunitvector_length': f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length',
        'boundary_condition_external_faces': 'Geometry/Boundary Condition Lines/External Faces',
        'boundary_condition_attributes': 'Geometry/Boundary Condition Lines/Attributes/',
        'boundary_condition_fixes': 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Boundary Conditions',
        VOLUME_ELEVATION_INFO: f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Info',
        VOLUME_ELEVATION_VALUES: f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Values',
    }
    return hdf_paths


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
        attrs = _parse_attributes(dataset)
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


class HDFReader:
    """
    Reads RAS hydrodynamic data required for WQ calculations
    in Clearwater Riverine Model from HDF file.
    """
    def __init__(
        self,
        file_path: str,
        datetime_range: Optional[Tuple[int, int] | Tuple[str, str]] = None
    ) -> None:
        """
        Opens HDF file and reads information required to
        set-up model mesh.
        """
        self.file_path = file_path
        self.infile = h5py.File(file_path, 'r')
        self.project_name = self.infile[
            'Geometry/2D Flow Areas/Attributes'
        ][()][0][0].decode('UTF-8')
        self.paths = _hdf_internal_paths(self.project_name)
        self.datetime_range = datetime_range

    def _parse_dates(self):
        """Date handling."""
        # time
        time_stamps_binary = self.infile[self.paths['binary_time_stamps']][()]

        # pandas is working faster than numpy for binary conversion
        time_stamps = pd.Series(time_stamps_binary).str.decode('utf8')
        xr_time_stamps = pd.to_datetime(time_stamps, format='%d%b%Y %H:%M:%S')

        if self.datetime_range is None:
            self.datetime_range_indices = (None, None)
        elif isinstance(self.datetime_range[0], int):
            self.datetime_range_indices: Tuple[int, int] = (
                self.datetime_range[0],
                self.datetime_range[1] + 1
                )
        elif isinstance(self.datetime_range[0], str):
            start_date, end_date = map(
                lambda x: pd.to_datetime(x, format='%m-%d-%Y %H:%M:%S'),
                self.datetime_range
            )
            subset_dates = xr_time_stamps[
                (xr_time_stamps >= start_date) & (xr_time_stamps <= end_date)
            ]
            subset_indices = subset_dates.index.intersection(
                xr_time_stamps.index
            )
            self.datetime_range_indices: Tuple[int, int] = (
                subset_indices[0],
                subset_indices[-1] + 1
            )
        else:
            raise TypeError(
                "Invalid datetime_range, must be tuple of strings or ints"
            )

        if self.datetime_range_indices != (None, None):
            xr_time_stamps = xr_time_stamps[
                    self.datetime_range_indices[0]:
                    self.datetime_range_indices[1]
                    ]

        return xr_time_stamps

    def define_coordinates(self, mesh: xr.Dataset):
        """Populate Coordinates and Dimensions"""
        # x-coordinates
        mesh = mesh.assign_coords(
            node_x=xr.DataArray(
                data=self.infile[self.paths[NODE_X]][()].T[0],
                dims=('node',),
            )
        )
        # y-coordinates
        mesh = mesh.assign_coords(
            node_y=xr.DataArray(
                data=self.infile[self.paths[NODE_X]][()].T[1],
                dims=('node',),
            )
        )

        xr_time_stamps = self._parse_dates()

        mesh = mesh.assign_coords(
            time=xr.DataArray(
                data=xr_time_stamps,
                dims=('time',),
            )
        )
        return mesh

    def define_topology(self, mesh: xr.Dataset):
        """Define mesh topology """
        mesh[FACE_NODES] = xr.DataArray(
            data=self.infile[
                f'Geometry/2D Flow Areas/{self.project_name}/Cells FacePoint Indexes'
            ][()],
            coords={
                "face_x": ('nface', self.infile[self.paths[FACE_X]][()].T[0]),
                "face_y": ('nface', self.infile[self.paths[FACE_Y]][()].T[1]),
            },
            dims=('nface', 'nmax_face'),
            attrs={
                'cf_role': 'face_node_connectivity',
                'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                'start_index': 0,
                '_FillValue': -1
            }
        )
        mesh[EDGE_NODES] = xr.DataArray(
            data=self.infile[self.paths[EDGE_NODES]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_node_connectivity',
                'long_name': 'Vertex nodes of mesh edges',
                'start_index': 0
            })
        mesh[EDGE_FACE_CONNECTIVITY] = xr.DataArray(
            data=self.infile[self.paths[EDGE_FACE_CONNECTIVITY]][()],
            dims=("nedge", '2'),
            attrs={
                'cf_role': 'edge_face_connectivity',
                'long_name': 'neighbor faces for edges',
                'start_index': 0
            })

    def define_hydrodynamics(self, mesh: xr.Dataset):
        """Populates hydrodynamic data in UGRID-compliant xarray."""
        mesh[EDGES_FACE1] = _hdf_to_xarray(
            mesh['edge_face_connectivity'].T[0],
            ('nedge'),
            attrs={'Units': ''}
        )
        mesh[EDGES_FACE2] = _hdf_to_xarray(
            mesh['edge_face_connectivity'].T[1],
            ('nedge'),
            attrs={'Units': ''}
        )

        nreal = mesh[EDGE_FACE_CONNECTIVITY].T[0].values.max()
        mesh.attrs[NUMBER_OF_REAL_CELLS] = nreal

        mesh[FACE_SURFACE_AREA] = _hdf_to_xarray(
            self.infile[self.paths[FACE_SURFACE_AREA]],
            ("nface")
        )
        mesh[EDGE_VELOCITY] = _hdf_to_xarray(
            self.infile[self.paths[EDGE_VELOCITY]],
            ('time', 'nedge'),
            time_constraint=self.datetime_range_indices,

        )
        mesh[EDGE_LENGTH] = _hdf_to_xarray(
            self.infile[self.paths[EDGE_LENGTH]][:, 2],
            ('nedge'),
            attrs={'Units': 'ft'}
        )
        mesh[WATER_SURFACE_ELEVATION] = _hdf_to_xarray(
            self.infile[self.paths[WATER_SURFACE_ELEVATION]],
            (['time', 'nface']),
            time_constraint=self.datetime_range_indices
        )
        try:
            mesh[VOLUME] = _hdf_to_xarray(
                self.infile[self.paths[VOLUME]],
                ('time', 'nface'),
                time_constraint=self.datetime_range_indices
            )
        except KeyError:
            mesh.attrs['volume_calculation_required'] = True
            mesh.attrs['face_volume_elevation_info'] = _hdf_to_dataframe(
                self.infile[self.paths['volume elevation info']]
            )
            mesh.attrs['face_volume_elevation_values'] = _hdf_to_dataframe(
                self.infile[self.paths['volume_elevation_values']]
            )
        try:
            mesh[FLOW_ACROSS_FACE] = _hdf_to_xarray(
                self.infile[self.paths[FLOW_ACROSS_FACE]],
                ('time', 'nedge'),
                time_constraint=self.datetime_range_indices
            )
        except:
            mesh.attrs['face_area_calculation_required'] = True
            mesh.attrs['face_area_elevation_info'] = _hdf_to_dataframe(
                self.infile[self.paths['area_elevation_info']]
            )
            mesh.attrs['face_area_elevation_values'] = _hdf_to_dataframe(
                self.infile[self.paths['area_elevation_values']]
            )
            mesh.attrs['face_normalunitvector_and_length'] = _hdf_to_dataframe(
                self.infile[self.paths['normalunitvector_length']]
            )
            mesh.attrs['face_cell_indexes_df'] = _hdf_to_dataframe(
                self.infile[self.paths[EDGE_FACE_CONNECTIVITY]]
            )
        try:
            mesh[FACE_HYD_DEPTH] = _hdf_to_xarray(
                self.infile[self.paths[FACE_HYD_DEPTH]],
                (['time', 'nface']),
                time_constraint=self.datetime_range_indices
            )
        except KeyError:
            print("'Cell Hydraulic Depth' not found in hdf file; skip reading it. ")
        try:
            mesh[FACE_VEL_X] = _hdf_to_xarray(
                self.infile[self.paths[FACE_VEL_X]],
                (['time', 'nface']),
                time_constraint=self.datetime_range_indices
            )
        except KeyError:
            print("'Cell Velocity - Velocity X' not found in hdf file; skip reading it. ")
        try:
            mesh[FACE_VEL_Y] = _hdf_to_xarray(
                self.infile[self.paths[FACE_VEL_Y]],
                (['time', 'nface']),
                time_constraint=self.datetime_range_indices
            )
        except KeyError:
            print("'Cell Velocity - Velocity Y' not found in hdf file; skip reading it. ")
        try:
            mesh[FACE_VEL_MAG] = (mesh[FACE_VEL_X] ** 2
                                + mesh[FACE_VEL_Y] ** 2) ** 0.5
        except KeyError:
            print("Cell velocities X and Y not found in hdf file; skip calculating velocity magnitude")
        
        mesh.attrs[VOLUME_ELEVATION_LOOKUP] = self._create_lookup_df()
    
    def _create_lookup_df(self):
        volume_elevation_info_df = _hdf_to_dataframe(
            self.infile[self.paths[VOLUME_ELEVATION_INFO]]
            )
        volume_elevation_vals_df = _hdf_to_dataframe(
            self.infile[self.paths[VOLUME_ELEVATION_VALUES]]
        )
        # Define cells associated with each lookup value
        volume_elevation_vals_df['Cell']  = np.concatenate([
            np.full(count, cell)
            for cell, count in zip(volume_elevation_info_df.index, volume_elevation_info_df['Count'])
            ])

        # Create lookup table
        df_ls = []

        for cell in volume_elevation_vals_df['Cell'].unique():
            cell_df = self._create_cell_lookup_table(
                cell,
                volume_elevation_vals_df
            )
            df_ls.append(cell_df)

        return pd.concat(df_ls)

    def _create_cell_lookup_table(
        self,
        cell_no: int,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Filter for single cell
        df_temp = df[df['Cell'] == cell_no]
        test_df = df_temp.copy().reset_index(drop=True)
        cell_surface_area = _hdf_to_dataframe(
            self.infile[self.paths[FACE_SURFACE_AREA]]
            )

        # Compute differences in elevation and volume between adjacent rows 
        # (i.e., vertical layers in the cell)
        test_df['Delta Elev'] = test_df['Elevation'].diff()
        test_df['Delta Volume'] = test_df['Volume'].diff()

        # Calculate the wetted surface area based on the volume and depth
        test_df['Surface Area'] = test_df['Delta Volume'] / test_df['Delta Elev']

        # Average surface area between two elevation bands
        # Approximates wetted surface area between two elevatiosn
        test_df['Wetted Surface Area'] = (test_df['Surface Area'] + test_df['Surface Area'].shift(-1)) / 2

        # Get maximum volume
        max_index = test_df['Volume'].idxmax()

        # Set edge cases (first and last slice)
        # Compare with total surface area for the cell as a whole
        cell_table = cell_surface_area[cell_surface_area.index == cell_no]
        input_value = cell_table['Surface Area'].values[0] 
        # Set wetted surface area at the first row to 0 (i.e., first slice)
        test_df.at[0, 'Wetted Surface Area'] = 0 
        test_df.at[max_index, 'Wetted Surface Area'] = input_value
        return test_df

    def define_boundary_hydrodynamics(self, mesh: xr.Dataset):
        """Read necessary information on hydrodynamics,"""
        # Pull important boundary information from the HDF file.
        external_faces = pd.DataFrame(
            self.infile[self.paths['boundary_condition_external_faces']][()]
        )
        attributes = pd.DataFrame(
            self.infile[self.paths['boundary_condition_attributes']][()]
        )
        list_of_boundaries = list(
            self.infile[self.paths['boundary_condition_fixes']].keys()
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
        boundary_data = self._fix_boundary_hydrodynamics(
            boundary_data,
            list_of_boundaries
        )
        # add to the mesh
        mesh.attrs['boundary_data'] = boundary_data

    def _fix_boundary_hydrodynamics(
        self,
        boundary_data: pd.DataFrame,
        list_of_boundaries: list,
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
            attrs = _parse_attributes(self.infile[fpath])
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

    def close(self):
        """Close HDF file"""
        self.infile.close()
