import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from clearwater_data.variables import VariableRegistry
from clearwater_riverine.variables import(
    ADVECTION_COEFFICIENT,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    DIFFUSION_COEFFICIENT,
    EDGE_FACE_CONNECTIVITY,
    EDGES_FACE1,
    EDGES_FACE2,
    EDGE_VELOCITY,
    FACES,
    FLOW_ACROSS_FACE,
    GATE_CONNECTIVITY,
    GATE_FLOW,
    NEDGE,
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)

# matrix solver 
class LHS:
    def __init__(self, registry: VariableRegistry):
        """
        Initialize Sparse Matrix used to solve transport equation. 

        Rather than looping through every single cell at every timestep, we can instead set up a sparse 
        matrix at each timestep that will allow us to solve the entire unstructured grid all at once. 
        We will solve an implicit Advection-Diffusion (transport) equation for the fractional total-load 
        concentrations. This discretization produces a linear system of equations that can be represented by 
        a sparse-matrix problem. 

        All constituents will have the same LHS matrix, since this is populated entirely by 
        hydrodynamic and topological information from the model grid. 

        """
        edges_face1 = registry.get(EDGE_FACE_CONNECTIVITY).T[0]
        edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]
        self.real_cell_count = registry.get(NUMBER_OF_REAL_CELLS)
        self.real_cell_index = registry.get(NUMBER_OF_REAL_CELLS) - 1

        self.internal_edges = np.where(
            (edges_face1 <= self.real_cell_index) & (edges_face2 <= self.real_cell_index)
        )[0]
        self.internal_edge_count = len(self.internal_edges)
        self.real_edges_face1 = np.where(edges_face1 <= self.real_cell_index)[0]
        self.real_edges_face2 = np.where(edges_face2 <= self.real_cell_index)[0]
        self.has_gate_flow = GATE_FLOW in registry
                
    def update_values(
        self,
        registry: VariableRegistry,
        current_time: datetime,
        time_step: timedelta,

    ):
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
            rows / columns: point to the row and column of each cell
            coefficients: value in the specified row, column pair in the matrix 
        """
        # get requried variables from registry
        flow_across_face = registry.get_at_time(
            FLOW_ACROSS_FACE,
            current_time
        )
        volume = registry.get_at_time(
            VOLUME,
            current_time + time_step  # use volume at t+1 timestep
        ) 
        coefficient_to_diffusion_term = registry.get_at_time(
            COEFFICIENT_TO_DIFFUSION_TERM,
            current_time
        )

        # topology information
        faces = registry.get(VOLUME)[FACES]
        edges_face1 = registry.get(EDGE_FACE_CONNECTIVITY).T[0]
        edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]

        if self.has_gate_flow:
            gate_flow = registry.get(
                GATE_FLOW,
                current_time
            )
            flow_out_gate_indices = np.where(gate_flow > 0)
            flow_in_gate_indices = np.where(gate_flow < 0)
        else:
            flow_out_gate_indices = np.array([])
            flow_in_gate_indices = np.array([])            

        # define edges where flow is flowing in versus out at current timestep
        flow_out_indices = np.where(flow_across_face > 0)[0]
        flow_out_indices_internal = np.where(
            (flow_across_face > 0) & \
            (np.isin(flow_across_face.nedge, self.internal_edges))
        )[0]
        flow_in_indices = np.where(
            (flow_across_face < 0) & \
            (np.isin(flow_across_face.nedge, self.internal_edges))
        )

        # find empty cells at next timestep
        vol_subset = volume.isel(nface=slice(0, self.real_cell_index))
        # get indices of zero-volume cells
        empty_cells = np.where(vol_subset.values == 0)[0]

        # initialize arrays that will define the sparse matrix 
        self.start_index = 0
        self.end_index = 0
        self.__init_matrix_values(
            flow_out_indices,
            flow_in_indices,
            empty_cells,
            flow_out_gate_indices,
            flow_in_gate_indices,
        )

        # fill in matrix values
        self.__fill_empty_cells(empty_cells)
        self.__fill_load_values(volume, faces, registry.get(CHANGE_IN_TIME))
        self.__fill_diffusion_values(
            coefficient_to_diffusion_term,
            edges_face1,
            edges_face2,
        )
        self.__fill_advection_values(
            flow_across_face,
            edges_face1,
            edges_face2,
            flow_out_indices,
            flow_out_indices_internal,
            flow_in_indices,
        )
        if self.has_gate_flow:
            self.__fill_advection_values(
                flow_across_face,
                registry.get(GATE_CONNECTIVITY).T[0],
                registry.get(GATE_CONNECTIVITY).T[1],
                flow_out_gate_indices,
                flow_out_gate_indices,
                flow_in_gate_indices,
            )

    def __init_matrix_values(
        self,
        flow_out_indices,
        flow_in_indices,
        empty_cells,
        flow_out_gate_indices,
        flow_in_gate_indices,

    ):
        length_of_values = self.internal_edge_count * 2 + self.real_cell_count * 2 + \
            len(flow_out_indices)* 2  + len(flow_in_indices)*2 + len(empty_cells) + \
            len(self.real_edges_face1) + len(self.real_edges_face2) + \
            len(flow_out_gate_indices) + len(flow_in_gate_indices)
        # create empty placeholders that will be used to fill the CSR
        # rows and columns have the indices of the row and column (where values go in the matrix)
        # and coefficients are the values that go in that location within the matrix
        self.rows = np.zeros(length_of_values)
        self.columns = np.zeros(length_of_values)
        self.coefficients = np.zeros(length_of_values)
    
    
    def __fill_empty_cells(self, empty_cells):
        """
        Place dummy values in empty cell diagonals so the matrix is invertible.
        Since there is no volume, these will solve to 0 as desired. 
        Without filling dummy values on the diagonals, the matrix algebra will fail. 
        """
        # put dummy values in dry cells
        self.__fill(
            rows=empty_cells,
            columns=empty_cells,
            coefficients=np.ones(len(empty_cells)),
        )


    def __fill_load_values(self, volume, faces, change_in_time):
        """
        Load = (Volume * Concentration) / Change in Time
        Since concentration is unknown at the n+1 timestep (this is what is being solved for),
        the coefficient to this term will go on the LHS of the transport equation.      
        """
        load = volume[0:self.real_cell_count] / change_in_time
        self.__fill(
            rows=faces[0:self.real_cell_count],
            columns=faces[0:self.real_cell_count],
            coefficients=load
        )


    def __fill_diffusion_values(
        self,
        coefficient_to_diffusion_term,
        edges_face1,
        edges_face2,
    ):
        """
        Sum of coefficient to diffusion terms associated with each cell. 
        The coefficient to the diffusion term is as follows:
           (Face vertical area x diffusion coefficient) / (distance between cells)
        The coefficient to teh diffusion term is summed over all faces, and is multiplied by 
           the difference between the NEIGHBOR cell (N) and the REFERENCE cell (P);
           we therefore need to place values both on the diagonal and off-diagonal accordingly.
        Diffusion coefficients for ghost cells will get added to the RHS of the matrix.
        """
        # diagnoal terms
        self.__fill(
            rows=edges_face1[self.real_edges_face1],
            columns=edges_face1[self.real_edges_face1],
            coefficients=coefficient_to_diffusion_term[self.real_edges_face1]
        )
        self.__fill(
            rows = edges_face2[self.real_edges_face2],
            columns = edges_face2[self.real_edges_face2],
            coefficients=coefficient_to_diffusion_term[self.real_edges_face2]  
        )

        # off-diagonal terms
        self.__fill(
            rows=edges_face1[self.internal_edges],
            columns=edges_face2[self.internal_edges],
            coefficients=coefficient_to_diffusion_term[self.internal_edges] * -1
        )

        self.__fill(
            rows=edges_face2[self.internal_edges],
            columns=edges_face1[self.internal_edges], 
            coefficients=coefficient_to_diffusion_term[self.internal_edges] * -1
        )


    def __fill_advection_values(
        self,
        flow_across_face,
        edges_face1,
        edges_face2,
        flow_out_indices,
        flow_out_indices_internal,
        flow_in_indices,
    ):
        """
        Advection coefficient values.
        The advection term coefficient is the flow across the face multiplied by the Concentration across the face (C_f)
          Where the flow across the face is positive, the concentration across the face (C_f)
            is the reference cell (C_f = C_P)
          Where the flow across the face is negative, the concentration across the face (C_f)
            is the neighbor cell (C_f = C_N)
        This function places those values in the matrix accordingly and subtracts the value from 
          the corresponding partner cell (i.e., if C_f = C_P, then the value is placed on the diagonal for C_P
          and then that same value is subtracted from the neighbor cell (C_N) for mass balance -- or vice versa).
        """
        if len(flow_out_indices) > 0:
            # where face flow is positive, the concentration across the face will be the REFERENCE CELL 
            # so the the coefficient will go in the diagonal - both row and column will equal diag_cell
            # Advection coefficient for timestep t is the flow across the face going from t to t+1 
            self.__fill(
                rows = edges_face1[flow_out_indices],
                columns = edges_face1[flow_out_indices],
                coefficients=flow_across_face[flow_out_indices]
            )

            # subtract from corresponding neighbor cell (off-diagonal)
            # for internal cells only
            self.__fill(
                rows = edges_face2[flow_out_indices_internal],
                columns = edges_face1[flow_out_indices_internal],
                coefficients=flow_across_face[flow_out_indices_internal] * -1
        )

        if len(flow_in_indices) > 0:
            ## where face flopw is negative, the concentration across the face will be the neighbor cell ("N")
            ## so the coefficient will be off-diagonal
            ## This is internal cells only; external cells will be handled on the RHS
            self.__fill(
                rows=edges_face1[flow_in_indices],
                columns=edges_face2[flow_in_indices],
                coefficients=flow_across_face[flow_in_indices]
            )

            ## do the opposite on the corresponding diagonal 
            self.__fill(
                rows=edges_face2[flow_in_indices],
                columns=edges_face2[flow_in_indices],
                coefficients=flow_across_face[flow_in_indices] * -1,
            )


    def __fill(self, rows, columns, coefficients):
        """This function handles the placement of values in the matrix"""
        self.start_index = self.end_index
        self.end_index = self.start_index + len(coefficients)
        self.rows[self.start_index:self.end_index] = rows
        self.columns[self.start_index:self.end_index] = columns
        if self.rows.max() > self.real_cell_count:
            print('here')
        self.coefficients[self.start_index: self.end_index] = coefficients

    
class RHS:
    def __init__(
        self,
        registry: VariableRegistry,
    ):
        """
        Initialize the right-hand side matrix of concentrations based on user-defined boundary conditions. 
        """
        self.real_cell_index = registry.get(NUMBER_OF_REAL_CELLS) - 1
        self.real_cell_count = registry.get(NUMBER_OF_REAL_CELLS)
        edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]

        self.values = np.zeros(self.real_cell_count)
        self.ghost_cells = np.where(edges_face2 > self.real_cell_index)[0]

    def update_values(
        self,
        # solution: np.array,
        registry: VariableRegistry,
        current_time: datetime,
        time_step: timedelta,
        constituent_name: str,
    ):
        """ 
        Update right hand side data based on the solution from the previous timestep
            solution: solution from solving the sparse matrix 
        """
        constituent_data = registry.get_at_time(
            constituent_name,
            current_time
        )

        self.concentrations = constituent_data.fillna(0.0)[0:self.real_cell_count]
        # solver.values[0:]
        
        # solver = np.zeros(
        #     len(
        #         mesh[name].isel(time=t)
        #     )
        # ) 
        # solver[0:self.nreal_count] = solution
        # solver[self.input_array[t].nonzero()] = self.input_array[t][self.input_array[t].nonzero()] 
        self.values[:] = self.__calculate_rhs(registry, current_time, time_step, constituent_name)

    def _calculate_load(self, registry: VariableRegistry, current_time: datetime):
        """Calculate the load 
        Returns:
            load (xr.DataArray):            (M/T) Calculated as volume (L3) * concentration (M/L3) / time (T).
        """
        volume = registry.get_at_time(
            VOLUME,
            current_time
        )[0:self.real_cell_count]
        delta_time = registry.get(CHANGE_IN_TIME)
        load = volume * self.concentrations / delta_time
        return load
    
    def _calculate_ghost_cell_values(
            self,
            registry: VariableRegistry,
            current_time: datetime,
            time_step: timedelta,
            constituent_name: str,
        ):
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
        ghost_cells_in = np.zeros(self.real_cell_count)
        ghost_cells_out = np.zeros(self.real_cell_count)
        ghost_cells_in[:] = self._ghost_cell(
            registry,
            current_time,
            time_step,
            constituent_name,
            flowing_in=True
        )[0:self.real_cell_count]
        ghost_cells_out[:] = self._ghost_cell(
            registry=registry,
            current_time=current_time,
            time_step=time_step,
            constituent_name=constituent_name,
            flowing_in=False
        )[0:self.real_cell_count]

        return ghost_cells_in, ghost_cells_out
    
    def __calculate_rhs(
            self,
            registry: VariableRegistry,
            current_time: datetime,
            time_step: timedelta,
            constituent_name: str,
        ):
        """
        Calculates the Right Hand Side matrix,
            including the load at the current timestep for internal (real) cells,
            and known transport terms associated with connected external (ghost) cells.
        """
        load = self._calculate_load(registry, current_time)
        ghost_cells_in, ghost_cells_out = self._calculate_ghost_cell_values(registry, current_time, time_step, constituent_name)
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
    
    def _define_arrays(self, registry: VariableRegistry, advection: bool):
        """Initialize arrays for advection and diffusion terms associated with ghost cells.
        
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

        nface = registry.get(NFACE)
        nedge = registry.get(NEDGE) 

        if advection:
            advection_edge = np.zeros(nedge)
            advection_face = np.zeros(nface)
        diffusion_edge = np.zeros(nedge)
        diffusion_face = np.zeros(nface)
        return advection_edge, advection_face, diffusion_edge, diffusion_face
    
    def _edge_to_face(
            self,
            edge_array: np.array,
            face_array: np.array,
            mesh_array: xr.DataArray,
            index_list: list,
            internal_cell_index
        ):
        """Transfer values associated with edges to corresponding internal face.
````
        Args:
            edge_array (np.ndarray):        Numpy array with a length equal to the number of edges in the model.
                                                Populated with edge values between a ghost cell and and an internal cell. 
            face_array (np.ndarray):        Empty numpy array with a length equal to the number of faces in the model.
            mesh_array (xr.DataArray):      Values associated with an edge for a model mesh.
            index_list (list):              List of ghost cells meeting flowing in or out condition
            internal_cell_index:            Internal cell for a ghost cell edge.
        
        Returns:
            face_array (np.ndarray):         Numpy array with a length equal to the number of faces in the model.
                                                Populated with values previously associated with edges between a ghost and internal cell,
                                                now the values falls on the indices associated with the internal cell. 
        """    
        edge_array[index_list] = abs(mesh_array[index_list])
        values = np.where(edge_array != 0)[0]
        face_array[np.array(internal_cell_index)] = edge_array[values]
        return face_array

    def _ghost_cell(
            self,
            registry: VariableRegistry,
            current_time: datetime,
            time_step: timedelta,
            constituent_name: str,
            flowing_in: bool
        ):
        """
        Manages terms on the right hand side of the matrix associated with ghost cells
            that are flowing in or out of the model mesh.
        Returns:
            add_to_rhs (np.ndarray):        Array of transport terms associated with ghost cells
                                                that should be added to the right hand side.
        """
        advection, diffusion, condition = self._transport_mechanisms(flowing_in)
        advection_edge, advection_face, diffusion_edge, diffusion_face = self._define_arrays(registry, advection)

        edge_velocity = registry.get_at_time(
            EDGE_VELOCITY,
            current_time,
        )
        edges_face1 = registry.get(EDGE_FACE_CONNECTIVITY).T[0]
        edges_face2 = registry.get(EDGE_FACE_CONNECTIVITY).T[1]

        velocity_indices = np.where(condition(edge_velocity, 0))[0]
        advection_coefficient = registry.get_at_time(FLOW_ACROSS_FACE, current_time)
        diffusion_coefficient = registry.get_at_time(DIFFUSION_COEFFICIENT, current_time)
        diffusion_term = registry.get_at_time(COEFFICIENT_TO_DIFFUSION_TERM, current_time)
        index_list = np.intersect1d(velocity_indices, self.ghost_cells)
        internal_cell_index = edges_face1[index_list]
        external_cell_index = edges_face2[index_list]

        concentration_multipliers = np.zeros(registry.get(NFACE))
        concentration_multipliers[internal_cell_index] = registry.get_at_time(
            constituent_name,
            current_time + time_step
        )[external_cell_index]

        if len(index_list) != 0:
            if advection:
                advection_face[:] = self._edge_to_face(
                    advection_edge,
                    advection_face,
                    advection_coefficient,
                    index_list,
                    internal_cell_index
                    )
            if diffusion and diffusion_coefficient !=0:
                diffusion_face[:] = self._edge_to_face(
                    diffusion_edge,
                    diffusion_face,
                    diffusion_term,
                    index_list,
                    internal_cell_index
                    )
                
        if flowing_in:
            add_to_rhs = advection_face + diffusion_face
        else:
            add_to_rhs = diffusion_face
        
        add_to_rhs = add_to_rhs * concentration_multipliers
        
        return add_to_rhs