from suitesparse.spqr.spqr_solver_base_INT32_t_FLOAT64_t cimport SPQRSolverBase_INT32_t_FLOAT64_t
from suitesparse.common_types.suitesparse_types cimport *

from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT64_t cimport CSCSparseMatrix_INT32_t_FLOAT64_t

# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t

cdef class SPQRCysparseSolver_INT32_t_FLOAT64_t(SPQRSolverBase_INT32_t_FLOAT64_t):
    cdef:
        # Matrix A in CSC format
        CSCSparseMatrix_INT32_t_FLOAT64_t csc_mat

        # no copy of internal CSC C arrays?
        bint __no_copy

        float __matrix_transform_time

