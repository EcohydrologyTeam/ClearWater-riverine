import numpy as np
import xarray as xr

from clearwater_riverine.variables import(
    ADVECTION_COEFFICIENT,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    EDGE_FACE_CONNECTIVITY,
    EDGES_FACE1,
    EDGES_FACE2,
    EDGE_VELOCITY,
    FACES,
    VOLUME,
)

# matrix solver 
class LHS:
    def __init__(self, mesh: xr.Dataset):
        """ Initialize Sparse Matrix used to solve transport equation. 

        Rather than looping through every single cell at every timestep, we can instead set up a sparse 
        matrix at each timestep that will allow us to solve the entire unstructured grid all at once. 
        We will solve an implicit Advection-Diffusion (transport) equation for the fractional total-load 
        concentrations. This discretization produces a linear system of equations that can be represented by 
        a sparse-matrix problem. 

        """
        self.internal_edges = np.where((mesh[EDGES_FACE1] <= mesh.nreal) & (mesh[EDGES_FACE2] <= mesh.nreal))[0]
        self.internal_edge_count = len(self.internal_edges)
        self.real_edges_face1 = np.where(mesh[EDGES_FACE1] <= mesh.nreal)[0]
        self.real_edges_face2 = np.where(mesh[EDGES_FACE2] <= mesh.nreal)[0]
        self.nreal_count = mesh.nreal + 1
                
    def update_values(self, mesh: xr.Dataset, t: float):
        """ Updates values in the LHS matrix based on the timestep. 

        A sparse matrix is a matrix that is mostly zeroes. Here, we will set up an NCELL x NCELL sparse matrix. 
            - The diagonal values represent the reference cell ("P")
            - The non-zero off-diagonal values represent the other cells that share an edge with that cell:
                i.e., neighboring cell ("N") that shares a face ("f") with P. 

        This function populates the sparse matrix with:
            - Values on the Diagonal (associated with the cell with the same index as that row/column):
                - Load at the t+1 timestep (volume at the t + 1 timestep / change in time)
                - Sum of diffusion coefficients associated with a cell
                - FOR DRY CELLS ONLY (volume = 0), insert a dummy value (1) so that the matrix is not singular
            - Values Off-Diagonal:
                - Coefficient to the diffusion term at the t+1 timestep 
            - Advection: a special case (updwinds scheme)
                - When the advection coefficient is positive, the concentration across the face will be the reference cell ("P")
                    so the coefficient will go in the diagonal. This value will then be subtracted from the corresponding neighbor cell.
                - When the advection coefficient is negative, the concentration across the face will be the neighbor cell ("N")
                    so the coefficient will be off-diagonal. This value will the subtracted from the corresponding reference cell.

        Attributes:
            rows / cols: point to the row and column of each cell
            coef: value in the specified row, column pair in the matrix 
        """
        # define edges where flow is flowing in versus out and find all empty cells
        # at the t+1 timestep
        flow_out_indices = np.where((mesh[ADVECTION_COEFFICIENT][t] > 0))[0]
        flow_out_indices_internal = np.where((mesh[ADVECTION_COEFFICIENT][t] > 0) & \
                                             (np.isin(mesh.nedge, self.internal_edges)))[0]
        flow_in_indices = np.where((mesh[ADVECTION_COEFFICIENT][t] < 0) & \
                                   (np.isin(mesh.nedge, self.internal_edges)))[0]
        empty_cells = np.where(mesh[VOLUME][t+1] == 0)[0][0:self.nreal_count]

        # initialize arrays that will define the sparse matrix 
        len_val = self.internal_edge_count * 2 + self.nreal_count * 2 + \
            len(flow_out_indices)* 2  + len(flow_in_indices)*2 + len(empty_cells) + \
                len(self.real_edges_face1) + len(self.real_edges_face2)
        self.rows = np.zeros(len_val)
        self.cols = np.zeros(len_val)
        self.coef = np.zeros(len_val)

        # put dummy values in dry cells
        start = 0
        end = len(empty_cells)
        self.rows[start:end] = empty_cells
        self.cols[start:end] = empty_cells
        self.coef[start:end] = 1

        ###### diagonal terms - load and sum of diffusion coefficients associated with each cell
        start = end
        end = end + self.nreal_count
        self.rows[start:end] = mesh[FACES][0:self.nreal_count]
        self.cols[start:end] = mesh[FACES][0:self.nreal_count]
        seconds = mesh[CHANGE_IN_TIME].values[t] 
        self.coef[start:end] = mesh[VOLUME][t+1][0:self.nreal_count] / seconds 

        # diagonal terms - sum of diffusion coefficients associated with each cell
        start = end
        end = end + len(self.real_edges_face1)

        self.rows[start:end] = mesh[EDGES_FACE1][self.real_edges_face1]
        self.cols[start:end] = mesh[EDGES_FACE1][self.real_edges_face1]
        self.coef[start:end] = mesh[COEFFICIENT_TO_DIFFUSION_TERM][t][self.real_edges_face1]

        start = end
        end = end + len(self.real_edges_face2)
        self.rows[start:end] = mesh[EDGES_FACE2][self.real_edges_face2]
        self.cols[start:end] = mesh[EDGES_FACE2][self.real_edges_face2]
        self.coef[start:end] = mesh[COEFFICIENT_TO_DIFFUSION_TERM][t][self.real_edges_face2]

        ###### Advection
        # if statement to prevent errors if flow_out_indices or flow_in_indices have length of 0
        if len(flow_out_indices) > 0:
            start = end
            end = end + len(flow_out_indices)

            # where advection coefficient is positive, the concentration across the face will be the REFERENCE CELL 
            # so the the coefficient will go in the diagonal - both row and column will equal diag_cell
            self.rows[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[0][flow_out_indices]
            self.cols[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[0][flow_out_indices]
            self.coef[start:end] = mesh[ADVECTION_COEFFICIENT][t][flow_out_indices]  

            # subtract from corresponding neighbor cell (off-diagonal)
            start = end
            end = end + len(flow_out_indices_internal)
            self.rows[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[1][flow_out_indices_internal]
            self.cols[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[0][flow_out_indices_internal]
            self.coef[start:end] = mesh[ADVECTION_COEFFICIENT][t][flow_out_indices_internal] * -1  

        if len(flow_in_indices) > 0:
            # update indices
            start = end
            end = end + len(flow_in_indices)

            ## where it is negative, the concentration across the face will be the neighbor cell ("N")
            ## so the coefficient will be off-diagonal 
            self.rows[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[0][flow_in_indices]
            self.cols[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[1][flow_in_indices]
            self.coef[start:end] = mesh[ADVECTION_COEFFICIENT][t][flow_in_indices] 

            ## update indices 
            start = end
            end = end + len(flow_in_indices)
            ## do the opposite on the corresponding diagonal 
            self.rows[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[1][flow_in_indices]
            self.cols[start:end] = mesh[EDGE_FACE_CONNECTIVITY].T[1][flow_in_indices]
            self.coef[start:end] = mesh[ADVECTION_COEFFICIENT][t][flow_in_indices]  * -1 
        
        ###### off-diagonal terms - diffusion
        # update indices
        start = end
        end = end + self.internal_edge_count
        self.rows[start:end] = mesh[EDGES_FACE1][self.internal_edges]
        self.cols[start:end] = mesh[EDGES_FACE2][self.internal_edges]
        self.coef[start:end] = -1 * mesh[COEFFICIENT_TO_DIFFUSION_TERM][t][self.internal_edges]

        # update indices and repeat 
        start = end
        end = end + self.internal_edge_count
        self.rows[start:end] = mesh[EDGES_FACE2][self.internal_edges]
        self.cols[start:end] = mesh[EDGES_FACE1][self.internal_edges]
        self.coef[start:end] = -1 * mesh[COEFFICIENT_TO_DIFFUSION_TERM][t][self.internal_edges]    
    
class RHS:
    def __init__(
        self,
        mesh: xr.Dataset,
        input_array: np.array,
    ):
        """
        Initialize the right-hand side matrix of concentrations based on user-defined boundary conditions. 

        Args:
            mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.
            inp (np.array):      Array of shape (time x nface) with user-defined inputs of concentrations
                                    in each cell at each timestep. 
        """
        self.nreal_count = mesh.nreal + 1  # 0 indexed
        self.input_array = input_array
        self.vals = np.zeros(self.nreal_count)
        self.ghost_cells = np.where(mesh[EDGES_FACE2] > mesh.nreal)[0]

    def update_values(
        self,
        solution: np.array,
        mesh: xr.Dataset,
        t: int,
        name: str,
    ):
        """ 
        Update right hand side data based on the solution from the previous timestep
            solution: solution from solving the sparse matrix 

        Args:
            solution (np.array):    Solution of concentrations at timestep t from solving sparse matrix. 
            mesh (xr.Dataset):      UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                Timestep
            name (str):             Constituent name.
        """
        solver = np.zeros(
            len(
                mesh[name].isel(time=t)
            )
        ) 
        solver[0:self.nreal_count] = solution
        solver[self.input_array[t].nonzero()] = self.input_array[t][self.input_array[t].nonzero()] 
        self.vals[:] = self._calculate_rhs(mesh, t, solver[0:self.nreal_count])

    def _calculate_change_in_time(self, mesh: xr.Dataset, t: int):
        """Calculate the change in time.

        Args:
            mesh (xr.Dataset):      UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                Timestep  

        Returns:
            The change in time at timestep t.
        """
        return mesh[CHANGE_IN_TIME].values[t]
    
    def _calculate_volume(self, mesh: xr.Dataset, t: int):
        """Calculate the volume in real cells.

        Args:
            mesh (xr.Dataset):      UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                Timestep

        Returns:
            xr.DataArray of volume values for internal (real) cells at timestep t.
        """
        return mesh[VOLUME][t][0:self.nreal_count]
    
    def _calculate_load(self, mesh: xr.DataArray, t: int, concentrations: np.ndarray):
        """Calculate the load 

        Args:
            mesh (xr.Dataset):              UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                        Timestep
            concentrations (xr.DataArray):  Concentrations at t timestep.

        Returns:
            load (xr.DataArray):            (M/T) Calculated as volume (L3) * concentration (M/L3) / time (T).
        """
        volume = self._calculate_volume(mesh, t)
        delta_time = self._calculate_change_in_time(mesh, t)
        load = volume * concentrations / delta_time
        return load
    
    def _calculate_ghost_cell_values(self, mesh: xr.Dataset, t: int):
        """
        Determine the ghost cells that are flowing into the model mesh
            and the ghost cells that are receiving flow out of the model mesh.

        Args:
            mesh (xr.Dataset):              UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                        Timestep

        Returns:
            ghost_cells_in (np.ndarray):    Indices of ghost cells that are flowing in to the model mesh
            ghost_cells_out (np.ndarray):   Indices of ghost cells that are receiving flow out of the model mesh.
        """
        ghost_cells_in = np.zeros(self.nreal_count)
        ghost_cells_out = np.zeros(self.nreal_count)
        ghost_cells_in[:] = self._ghost_cell(mesh, t, flowing_in=True)[0:self.nreal_count]
        ghost_cells_out[:] = self._ghost_cell(mesh, t, flowing_in=False)[0:self.nreal_count]
        return ghost_cells_in, ghost_cells_out
    
    def _calculate_rhs(self, mesh: xr.Dataset, t: int, concentrations: np.ndarray):
        """
        Calculates the Right Hand Side matrix,
            including the load at the current timestep for internal (real) cells,
            and known transport terms associated with connected external (ghost) cells. 

        Args:
            mesh (xr.Dataset):              UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                        Timestep
            concentrations (xr.DataArray):  Concentrations at t timestep.
        """
        load = self._calculate_load(mesh, t, concentrations)
        ghost_cells_in, ghost_cells_out = self._calculate_ghost_cell_values(mesh, t+1)
        return load + ghost_cells_in + ghost_cells_out


    def _transport_mechanisms(self, flowing_in: bool):
        """
        Determines which transport mechanisms associated with ghost cells should
            be included in the right hand side of the matrix. This function also 
            determines a condition to help identify ghost cells that are flowing in/
            out of the mesh, based on the sign of the edge velocity. 
            1. Ghost cells flowing in to the model mesh will include both advection
            and diffusion terms under the upwind differencing scheme. Ghost cells flowing
            into the model mesh will have an edge velocity that is less than zero in RAS. 
            2. Ghost cells receiving flow from the model mesh will only include
            diffusion terms; the advection term for these cells will be on the left 
            hand side of the equation under the upwind differencing scheme. These cells
            will have an edge velocity that is greater than zero in RAS.
        
        Args:
            flowing_in (bool):              Indicator of whether the function should return values
                                                for ghost cells flowing in to the model (True) or
                                                receiving flow out of the model (false).
        Returns:
            advection (bool):               True if advection terms should be included on the RHS, otherwise False.
            diffusion (bool):               True if diffusion terms should be included on the RHS, otherwise False.
            condition (np.ufunc):           np.less for ghost cells flowing into the model,
                                                np.greater for ghost cells receiving flow out of the model. 
        """
        diffusion = True
        if flowing_in:
            advection = True
            condition = np.less
        else:
            advection = False
            condition = np.greater
        return advection, diffusion, condition
    
    def _define_arrays(self, mesh: xr.Dataset, advection: bool):
        """Initialize arrays for advection and diffusion terms associated with ghost cells.

        Args:
            mesh (xr.Dataset):              UGRID-complaint xarray Dataset with all data required for the transport equation.
            advection (bool):               Boolean indicating whether advection terms should be included on the RHS. 
        
        Returns:
            advection_edge(np.ndarray):     Empty numpy array with a length equal to the number of edges in the model.
            advection_face(np.ndarray):     Empty numpy array with a length equal to the number of faces in the model.
            diffusion_edge(np.ndarray):     Empty numpy array with a length equal to the number of edges in the model.
            diffusion_face(np.ndarray):     Empty numpy array with a length equal to the number of faces in the model.
        """
        advection_edge = None
        advection_face = None
        diffusion_edge = None
        diffusion_face = None

        if advection:
            advection_edge = np.zeros(len(mesh.nedge))
            advection_face = np.zeros(len(mesh.nface))
        diffusion_edge = np.zeros(len(mesh.nedge))
        diffusion_face = np.zeros(len(mesh.nface))
        return advection_edge, advection_face, diffusion_edge, diffusion_face
    
    def _edge_to_face(self, edge_array: np.array, face_array: np.array, mesh_array: xr.DataArray, index_list: list, internal_cell_index):
        """Transfer values associated with edges to corresponding internal face.

        Args:
            edge_array (np.ndarray):        Numpy array with a length equal to the number of edges in the model.
                                                Populated with edge values between a ghost cell and and an internal cell. 
            face_array (np.ndarray):        Empty numpy array with a length equal to the number of faces in the model.
        
        Returns:
            face_array (np.ndarray):         Numpy array with a length equal to the number of faces in the model.
                                                Populated with values previously associated with edges between a ghost and internal cell,
                                                now the values falls on the indices associated with the internal cell. 
        """    
        edge_array[index_list] = abs(mesh_array[index_list])
        values = np.where(edge_array != 0)[0]
        face_array[np.array(internal_cell_index)] = edge_array[values]
        return face_array

    def _ghost_cell(self, mesh: xr.Dataset, t: int, flowing_in: bool):
        """
        Manages terms on the right hand side of the matrix associated with ghost cells
            that are flowing in or out of the model mesh.

        Args:
            mesh (xr.Dataset):              UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (int):                        Timestep
            flowing_in (bool):              Indicator of whether the function should return values
                                                for ghost cells flowing in to the model (True) or
                                                receiving flow out of the model (false).
        Returns:
            add_to_rhs (np.ndarray):        Array of transport terms associated with ghost cells
                                                that should be added to the right hand side.
        """
        advection, diffusion, condition = self._transport_mechanisms(flowing_in)
        advection_edge, advection_face, diffusion_edge, diffusion_face = self._define_arrays(mesh, advection)

        velocity_indices = np.where(condition(mesh[EDGE_VELOCITY][t], 0))[0]
        index_list = np.intersect1d(velocity_indices, self.ghost_cells)
        internal_cell_index = mesh[EDGES_FACE1][index_list]
        external_cell_index = mesh[EDGES_FACE2][index_list]

        concentration_multipliers = np.zeros(len(mesh.nface))
        concentration_multipliers[internal_cell_index] = self.input_array[t][external_cell_index] 

        if len(index_list) != 0:
            if advection:
                advection_face[:] = self._edge_to_face(
                    advection_edge,
                    advection_face,
                    mesh[ADVECTION_COEFFICIENT][t],
                    index_list,
                    internal_cell_index
                    )
            if diffusion:
                if mesh.diffusion_coefficient !=0:
                    diffusion_face[:] = self._edge_to_face(
                        diffusion_edge,
                        diffusion_face,
                        mesh[COEFFICIENT_TO_DIFFUSION_TERM][t],
                        index_list,
                        internal_cell_index
                        )
                
        if flowing_in:
            add_to_rhs = advection_face + diffusion_face
        else:
            add_to_rhs = diffusion_face
        
        add_to_rhs = add_to_rhs * concentration_multipliers
        
        return add_to_rhs