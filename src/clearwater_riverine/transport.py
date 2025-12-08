
from scipy.sparse import csr_matrix, linalg

from clearwater_riverine.linalg import LHS
from clearwater_riverine.variables import (
    NUMBER_OF_REAL_CELLS
)


class TransportEngine:
    def __init__(self, registry):
        # initialize left hand side of transport equation
        self.lhs = LHS(registry)

    def run(self, registry, time_step, constituents):
        """Run the transport engine."""
        # update the left hand side of the matrix
        self.lhs.update_values(
            registry,
            time_step
        )

        # define compressed sparse row matrix for LHS
        real_cell_count = registry.get(NUMBER_OF_REAL_CELLS) + 1
        A = csr_matrix(
            (self.lhs.coefficients, (self.lhs.rows, self.lhs.columns)),
            shape = (real_cell_count, real_cell_count)
        )

        # loop through all constituents
        for constituent_name, constituent in constituents.items():
            constituent_value = self.registry.get_at_time(constituent_name, self.__current_time)
            # update right hand side of the matrix
            constituent.rhs.update_values(
                registry=registry,
                time_step=time_step,
                name=constituent_name,
            )
        
            # solve
            x = linalg.spsolve(A, constituent.b.vals)

            # update the value in the registry

            # optionally: calculate mass flux