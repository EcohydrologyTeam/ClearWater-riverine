import pandas as pd
import xarray as xr 
import h5py

class MeshManager:
    """
    Define UGRID-compliant xarray

    Parameters:
        infile (h5py._hl.files.File):    HDF File containing RAS2D output 
        project_name (str):              Name of the 2D Flow Area being modeled

    Returns:
        UGRID-compliant xarray with all geometry / time coordinates populated

    """
    def __init__(self, diffusion_coefficient_input: float):
        """Initialize UGRID-compliant xr.Dataset"""
        self.volume_calculation_required = False 
        self.face_area_calculation_required = False
        self.face_area_elevation_info = pd.DataFrame()
        self.face_area_elevation_values = pd.DataFrame()
        self.face_normalunitvector_and_length = pd.DataFrame()
        self.face_cell_indexes_df = pd.DataFrame()
        self.face_volume_elevation_info = pd.DataFrame()
        self.face_volume_elevation_values = pd.DataFrame()
        self.boundary_data = pd.DataFrame()


        # initialize mesh
        self.mesh = xr.Dataset()
        self.mesh["mesh2d"] = xr.DataArray(
            data=0,
            attrs={
                # required topology attributes
                'cf_role': 'mesh_topology',
                'long_name': 'Topology data of 2D mesh',
                'topology_dimension': 2,
                'node_coordinates': 'node_x node_y',
                'face_node_connectivity': 'face_nodes',
                # optionally required attributes
                'face_dimension': 'face',
                'edge_node_connectivity': 'edge_nodes',
                'edge_dimension': 'edge',
                # optional attributes 
                'face_edge_connectivity': 'face_edges',
                'face_face_connectivity': 'face_face_connectivity',
                'edge_face_connectivity': 'edge_face_connectivity',
                'boundary_node_connectivity': 'boundary_node_connectivity',
                'face_coordinates': 'face x face_y',
                'edge_coordinates': 'edge_x edge_y',
                },
        )
        self.mesh.attrs['diffusion_coefficient'] = diffusion_coefficient_input

    
    
        
"""
print("Populating Mesh...")
        print(" Initializing Geometry...")

        # mesh = define_ugrid(infile, project_name)

        print(" Storing Results...")
        # store additional useful information for various coefficient calculations in the mesh



        # compute necessary values 
        # TO DO: clean all this up; review functions to see if they can be simplified 
        # calculate cell volume
        print(" Computing Necessary Values...")
        # Determine Units 
        units = determine_units(mesh)
        
        # Cell Volume
        try:
            mesh['volume'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Cell Volume'], ('time', 'nface')) 
            mesh['volume'][:,mesh.attrs['nreal'].values+1:] = 0
        except KeyError: 
            print(" Warning! Cell volumes are being manually calculated. Please re-run the RAS model with optional outputs Cell Volume, Face Flow, and Eddy Viscosity selected.")
            cells_volume_elevation_info_df = hdf_to_dataframe(infile[f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Info'])
            cells_volume_elevation_values_df = hdf_to_dataframe(infile[f'Geometry/2D Flow Areas/{project_name}/Cells Volume Elevation Values'])
            # question: is it better to separate all of these things
            # or just input mesh, cells_volume_elevation_info_df / values df 
            cell_volumes = compute_cell_volumes(
                mesh['water_surface_elev'].values,
                mesh['faces_surface_area'].values,
                cells_volume_elevation_info_df['Starting Index'].values,
                cells_volume_elevation_info_df['Count'].values,
                cells_volume_elevation_values_df['Elevation'].values,
                cells_volume_elevation_values_df['Volume'].values,
            )
            mesh['volume_archive'] = hdf_to_xarray(cell_volumes, ('time', 'nface'), attrs={'Units': UNIT_DETAILS[units]['Volume']})
            mesh['volume'] = mesh['volume_archive']

        # Advection Coefficient and Associated Variables
        try:
            mesh['face_flow'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Flow'], ('time', 'nedge'))
            mesh['advection_coeff'] = hdf_to_xarray(infile[f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{project_name}/Face Flow'], ('time', 'nedge'))  * np.sign(abs(mesh['edge_velocity']))
            mesh['advection_coeff'].attrs['Units'] = UNIT_DETAILS[units]['Load']
            mesh['edge_vertical_area'] = mesh['advection_coeff'] / mesh['edge_velocity']
            # mesh['edge_vertical_area'][np.where(mesh['edge_vertical_area'].isnull())] = mesh['edge_vertical_area_archive'][np.where(mesh['edge_vertical_area'].isnull())]
            mesh['edge_vertical_area'] = mesh['edge_vertical_area'].fillna(0)
            mesh['edge_vertical_area'].attrs['Units'] = UNIT_DETAILS[units]['Area']
            # advection_coefficient = mesh['edge_vertical_area'] * mesh['edge_velocity'] 
            # mesh['advection_coeff'] = hdf_to_xarray(advection_coefficient, ('time', 'nedge'), attrs={'Units':'ft3/s'})  
        except KeyError:
            print(" Warning! Flows across the face are being manually calculated. This functionality is not fully tested! Please re-run the RAS model with optional outputs Cell Volume, Face Flow, and Eddy Viscosity selected.")
            faces_area_elevation_info_df = hdf_to_dataframe(infile[f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Info'])
            faces_area_elevation_values_df = hdf_to_dataframe(infile[f'Geometry/2D Flow Areas/{project_name}/Faces Area Elevation Values'])
            faces_normalunitvector_and_length_df = hdf_to_dataframe(infile[f'Geometry/2D Flow Areas/{project_name}/Faces NormalUnitVector and Length'])
            faces_cell_indexes_df = hdf_to_dataframe(infile[f'Geometry/2D Flow Areas/{project_name}/Faces Cell Indexes'])
            # should we be using 0 or 1 ?
            face_areas_0 = compute_face_areas(
                                                mesh['water_surface_elev'].values,
                                                faces_normalunitvector_and_length_df['Face Length'].values,
                                                faces_cell_indexes_df['Cell 0'].values,
                                                faces_area_elevation_info_df['Starting Index'].values,
                                                faces_area_elevation_info_df['Count'].values,
                                                faces_area_elevation_values_df['Z'].values,
                                                faces_area_elevation_values_df['Area'].values,
                                            )
            mesh['edge_vertical_area_archive'] = hdf_to_xarray(face_areas_0, ('time', 'nedge'), attrs={'Units': 'ft'})
            mesh['edge_vertical_area'] = mesh['edge_vertical_area_archive']
            mesh['edge_vertical_area'].attrs['Units'] = UNIT_DETAILS[units]['Area']
            advection_coefficient = mesh['edge_vertical_area'] * mesh['edge_velocity'] 
            mesh['advection_coeff'] = hdf_to_xarray(advection_coefficient, ('time', 'nedge'), attrs={'Units': UNIT_DETAILS[units]['Load']})
            mesh['face_flow'] = hdf_to_xarray(abs(advection_coefficient), ('time', 'nedge'), attrs={'Units': UNIT_DETAILS[units]['Load']})

        # computed values 
        # distance between centroids 
        distances = calc_distances_cell_centroids(mesh)
        mesh['face_to_face_dist'] = hdf_to_xarray(distances, ('nedge'), attrs={'Units': UNIT_DETAILS[units]['Length']})

        # coefficient to diffusion term
        coeff_to_diffusion = calc_coeff_to_diffusion_term(mesh)
        mesh['coeff_to_diffusion'] = hdf_to_xarray(coeff_to_diffusion, ("time", "nedge"), attrs={'Units': UNIT_DETAILS[units]['Load']})

        # sum of diffusion coeff
        sum_coeff_to_diffusion = calc_sum_coeff_to_diffusion_term(mesh)
        mesh['sum_coeff_to_diffusion'] = hdf_to_xarray(sum_coeff_to_diffusion, ('time', 'nface'), attrs={'Units': UNIT_DETAILS[units]['Load']})

        # advection
        # advection_coefficient = mesh['edge_vertical_area'] * mesh['edge_velocity'] 
        # mesh['advection_coeff'] = hdf_to_xarray(advection_coefficient, ('time', 'nedge'), attrs={'Units':'ft3/s'})
        # mesh['advection_coeff_archive'] = 

        # dt
        dt = np.ediff1d(mesh['time'])
        dt = dt / np.timedelta64(1, 's')
        dt = np.insert(dt, len(dt), np.nan)
        mesh['dt'] = hdf_to_xarray(dt, ('time'), attrs={'Units': 's'})

        # ghost cell volumes
        ghost_volumes_in, ghost_volumes_out = calc_ghost_cell_volumes(mesh)
        mesh['ghost_volumes_in'] = hdf_to_xarray(ghost_volumes_in, ('time', 'nface'), attrs={'Units': UNIT_DETAILS[units]['Volume']})
        mesh['ghost_volumes_out'] = hdf_to_xarray(ghost_volumes_out, ('time', 'nface'), attrs={'Units':UNIT_DETAILS[units]['Volume']})
        return mesh

        def to_xarray(self):
            with h5py.File(file_path, 'r') as infile:
                self.project_name = infile['Geometry/2D Flow Areas/Attributes'][()][0][0].decode('UTF-8')
                self.mesh = populate_ugrid(infile, self.project_name, diffusion_coefficient_input)
                self.boundary_data = populate_boundary_information(infile)
"""