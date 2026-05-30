# UGRID topology
NODE_X = 'node_x'
NODE_Y = 'node_y'
TIME = 'time'
FACE_X = 'face_x'
FACE_Y = 'face_y'
EDGE_NODES = 'edge_nodes'
FACE_NODES = 'face_nodes'
EDGE_FACE_CONNECTIVITY = 'edge_face_connectivity'
FACES = 'nface'
MESH_2D = 'mesh_2d'
NEDGE = 'nedge'
NFACE = 'nface'

# Available Variables
EDGES_FACE1 = 'edges_face1'
EDGES_FACE2 = 'edges_face2'
NUMBER_OF_REAL_CELLS = 'nreal'
VOLUME = 'volume'
WET_MASK = 'wet_mask'  # Phase-D Unit A: per-cell wet/dry boolean (time, nface)
FACE_SURFACE_AREA = 'faces_surface_area'
WETTED_SURFACE_AREA = 'wetted_surface_area'
EDGE_VELOCITY = 'edge_velocity'
EDGE_LENGTH = 'edge_length'
CHANGE_IN_TIME = 'dt'
WATER_SURFACE_ELEVATION =  'water_surface_elev'
DIFFUSION_COEFFICIENT = 'diffusion_coefficient'
FACE_HYD_DEPTH = 'face_hydraulic_depth'  #optional output in HEC-RAS hdf file
FACE_VEL_X = 'face_velocity_x'  #optional output in HEC-RAS hdf file
FACE_VEL_Y = 'face_velocity_y'  #optional output in HEC-RAS hdf file
# Phase F T2-C (2026-05-21): optional inputs needed by the diffusion
# dispatch (Elder shear-velocity, eddy-viscosity, array-based). Each
# is optional in the source HDF; the dispatcher checks for presence
# and raises a clear error when a method requests data the HDF did
# not write.
MANNINGS_N = "mannings_n"
EDDY_VISCOSITY = "eddy_viscosity"
CELL_EDDY_VISCOSITY_X = "cell_eddy_viscosity_x"
CELL_EDDY_VISCOSITY_Y = "cell_eddy_viscosity_y"
VOLUME_ELEVATION_INFO = 'volume_elevation_info'
VOLUME_ELEVATION_VALUES = 'volume_elevation_values'
VOLUME_ELEVATION_LOOKUP = 'volume_elevation_lookup'

# Calculated Values
FLOW_ACROSS_FACE = 'face_flow'
ADVECTION_COEFFICIENT = 'advection_coeff'
EDGE_VERTICAL_AREA = 'edge_vertical_area'
FACE_TO_FACE_DISTANCE = 'face_to_face_dist'
COEFFICIENT_TO_DIFFUSION_TERM  = 'coeff_to_diffusion'
SUM_OF_COEFFICIENTS_TO_DIFFUSION_TERM = 'sum_coeff_to_diffusion'
GHOST_CELL_VOLUMES_IN = 'ghost_volumes_in'
GHOST_CELL_VOLUMES_OUT = 'ghost_volumes_out'
FACE_VEL_MAG = 'face_velocity_magnitude'
AVERAGE_DEPTH = 'average_depth'
MAXIMUM_DEPTH = 'maximum_depth'
# Riverine MeshView-compat (2026-05-30): on-demand coupling depth. The
# cell mean water-column depth the v3 NSM coupling consumes, resolved by
# precedence (RAS Cell Hydraulic Depth -> volume/wsa -> WSE-bed). Only
# computed/registered when ``ClearwaterRiverine.enable_coupling_depth()``
# has been called; standalone transport runs never compute it.
COUPLING_DEPTH = 'coupling_depth'

# Structures
GATE_CONNECTIVITY = 'gate_connectivity'
GATE_FLOW = 'gate_flow'

# Lookup tables
LOOKUP_VOLUME = 'lookup_volume'
LOOKUP_ELEVATION = 'lookup_elevation'
LOOKUP_WETTED_SURFACE_AREA = 'lookup_wetted_surface_area'

# Boundary conditions
BOUNDARY_CONDITION_LINE_ID = 'BC Line ID'
BOUNDARY_FACE_INDEX = 'Face Index'
BOUNDARY_NAME = 'Name'