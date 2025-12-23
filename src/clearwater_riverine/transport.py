import numpy as np
from scipy.sparse import csr_matrix, linalg
from datetime import datetime, timedelta

from clearwater_riverine.linalg import LHS
from clearwater_riverine.variables import (
    NUMBER_OF_REAL_CELLS
)
from clearwater_riverine.constituents import Constituent


class TransportEngine:
    def __init__(self, registry):
        # initialize left hand side of transport equation
        self.lhs = LHS(registry)

    def run(
        self,
        registry: VariableRegistry,
        current_time: datetime,
        time_step: timedelta,
        constituents: dict[str, Constituent]
    ):
        """Run the transport engine."""
        # update the left hand side of the matrix
        self.lhs.update_values(
            registry,
            current_time,
            time_step,
        )

        # define compressed sparse row matrix for LHS
        real_cell_count = registry.get(NUMBER_OF_REAL_CELLS) + 1
        A = csr_matrix(
            (self.lhs.coefficients, (self.lhs.rows, self.lhs.columns)),
            shape = (real_cell_count, real_cell_count)
        )

        # loop through all constituents
        for constituent_name, constituent in constituents.items():
            constituent_value = registry.get_at_time(constituent_name, current_time)
            # update right hand side of the matrix
            constituent.rhs.update_values(
                registry=registry,
                current_time=current_time,
                time_step=time_step,
                constituent_name=constituent_name,
            )
        
            # solve
            x = linalg.spsolve(A, constituent.rhs.values)

            # update the value in the registry
            mask = np.isnan(constituent_value)
            constituent_value[mask] = x[mask]

            # optionally: calculate mass flux