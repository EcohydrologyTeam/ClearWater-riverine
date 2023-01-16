import numpy as np
import xarray as xr

from clearwater_riverine import variables

# matrix solver 
class LHS:
    """ Initialize Sparse Matrix used to solve transport equation. 

    Rather than looping through every single cell at every timestep, we can instead set up a sparse 
    matrix at each timestep that will allow us to solve the entire unstructured grid all at once. 
    We will solve an implicit Advection-Diffusion (transport) equation for the fractional total-load 
    concentrations. This discretization produces a linear system of equations that can be represented by 
    a sparse-matrix problem. 

    """
                
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
        flow_out_indices = np.where(mesh[variables.ADVECTION_COEFFICIENT][t+1] > 0)[0]
        flow_in_indices = np.where(mesh[variables.ADVECTION_COEFFICIENT][t+1] < 0)[0]
        empty_cells = np.where(mesh[variables.VOLUME][t+1] == 0)[0]

        # initialize arrays that will define the sparse matrix 
        len_val = len(mesh['nedge']) * 2 + len(mesh['nface']) * 2 + len(flow_out_indices)* 2  + len(flow_in_indices)*2 + len(empty_cells)
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
        end = end + len(mesh['nface'])
        self.rows[start:end] = mesh['nface']
        self.cols[start:end] = mesh['nface']
        seconds = mesh[variables.CHANGE_IN_TIME].values[t] # / np.timedelta64(1, 's'))
        self.coef[start:end] = mesh[variables.VOLUME][t+1] / seconds + mesh[variables.SUM_OF_COEFFICIENTS_TO_DIFFUSION_TERM][t+1] 

        # add ghost cell volumes to diagonals: based on flow across face into ghost cell
        # note: these values are 0 for cell that is not a ghost cell
        # note: also 0 for any ghost cell that is not RECEIVING flow 

        start = end
        end = end + len(mesh['nface'])
        self.rows[start:end] = mesh['nface']
        self.cols[start:end] = mesh['nface']
        self.coef[start:end] = mesh[variables.GHOST_CELL_VOLUMES_OUT][t+1] / seconds 
             
        ###### advection
        # if statement to prevent errors if flow_out_indices or flow_in_indices have length of 0
        if len(flow_out_indices) > 0:
            start = end
            end = end + len(flow_out_indices)

            # where advection coefficient is positive, the concentration across the face will be the REFERENCE CELL 
            # so the the coefficient will go in the diagonal - both row and column will equal diag_cell
            self.rows[start:end] = mesh['edge_face_connectivity'].T[0][flow_out_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[0][flow_out_indices]
            self.coef[start:end] = mesh[variables.ADVECTION_COEFFICIENT][t+1][flow_out_indices]  

            # subtract from corresponding neighbor cell (off-diagonal)
            start = end
            end = end + len(flow_out_indices)
            self.rows[start:end] = mesh['edge_face_connectivity'].T[1][flow_out_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[0][flow_out_indices]
            self.coef[start:end] = mesh[variables.ADVECTION_COEFFICIENT][t+1][flow_out_indices] * -1  

        if len(flow_in_indices) > 0:
            # update indices
            start = end
            end = end + len(flow_in_indices)

            ## where it is negative, the concentration across the face will be the neighbor cell ("N")
            ## so the coefficient will be off-diagonal 
            self.rows[start:end] = mesh['edge_face_connectivity'].T[0][flow_in_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[1][flow_in_indices]
            self.coef[start:end] = mesh[variables.ADVECTION_COEFFICIENT][t+1][flow_in_indices] 

            ## update indices 
            start = end
            end = end + len(flow_in_indices)
            ## do the opposite on the corresponding diagonal 
            self.rows[start:end] = mesh['edge_face_connectivity'].T[1][flow_in_indices]
            self.cols[start:end] = mesh['edge_face_connectivity'].T[1][flow_in_indices]
            self.coef[start:end] = mesh[variables.ADVECTION_COEFFICIENT][t+1][flow_in_indices]  * -1 
        
        ###### off-diagonal terms - diffusion
        # update indices
        start = end
        end = end + len(mesh['nedge'])
        self.rows[start:end] = mesh['edges_face1']
        self.cols[start:end] = mesh['edges_face2']
        self.coef[start:end] = -1 * mesh[variables.COEFFICIENT_TO_DIFFUSION_TERM][t+1]

        # update indices and repeat 
        start = end
        end = end + len(mesh['nedge'])
        self.rows[start:end] = mesh['edges_face2']
        self.cols[start:end] = mesh['edges_face1']
        self.coef[start:end] = -1 * mesh[variables.COEFFICIENT_TO_DIFFUSION_TERM][t+1] 

class RHS:
    def __init__(self, mesh: xr.Dataset, t: float, inp: np.array):
        """
        Initialize the right-hand side matrix of concentrations based on user-defined boundary conditions. 

        Args:
            mesh (xr.Dataset):   UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (float):           Timestep
            inp (np.array):      Array of shape (time x nface) with user-defined inputs of concentrations
                                    in each cell at each timestep. 

        Notes:
            Need to consider how ghost volumes / cells will be handled. 
            Need to consider how we will format the user-defined inputs 
                - An Excel file?
                - A modifiable table in a Jupyter notebook?
                - Alternatives?
        """
        self.conc = np.zeros(len(mesh['nface']))
        self.conc = inp[t] 
        self.vals = np.zeros(len(mesh['nface']))
        seconds = mesh[variables.CHANGE_IN_TIME].values[t] 
        # SHOULD GHOST VOLUMES BE INCLUDED?
        vol = mesh[variables.VOLUME][t] + mesh[variables.GHOST_CELL_VOLUMES_IN][t]
        self.vals[:] = vol / seconds * self.conc 
        # self.vals[:] = mesh['volume'][t] / seconds * self.conc 

    def update_values(self, solution: np.array, mesh: xr.Dataset, t: float, inp: np.array):
        """ 
        Update right hand side data based on the solution from the previous timestep
            solution: solution from solving the sparse matrix 
            inp: array of shape (time x nface) with user defined inputs of concentrations
                in each cell at each timestep 

        Args:
            solution (np.array):    Solution of concentrations at timestep t from solving sparse matrix. 
            mesh (xr.Dataset):      UGRID-complaint xarray Dataset with all data required for the transport equation.
            t (float):              Timestep
            inp (np.array):         Array of shape (time x nface) with user-defined inputs of concentrations
                                        in each cell at each timestep [boundary conditions]
        """
        seconds = mesh[variables.CHANGE_IN_TIME].values[t] 
        solution[inp[t].nonzero()] = inp[t][inp[t].nonzero()] 
        vol = mesh[variables.VOLUME][t] + mesh[variables.GHOST_CELL_VOLUMES_IN][t]
        self.vals[:] = solution * vol / seconds