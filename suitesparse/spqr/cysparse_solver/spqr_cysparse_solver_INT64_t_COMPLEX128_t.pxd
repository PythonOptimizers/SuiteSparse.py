from suitesparse.spqr.spqr_solver_base_INT64_t_COMPLEX128_t cimport SPQRSolverBase_INT64_t_COMPLEX128_t
from suitesparse.common_types.suitesparse_types cimport *

from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t cimport CSCSparseMatrix_INT64_t_COMPLEX128_t

# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t

cdef class SPQRCysparseSolver_INT64_t_COMPLEX128_t(SPQRSolverBase_INT64_t_COMPLEX128_t):
    cdef:
        # Matrix A in CSC format
        CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat

        # no copy of internal CSC C arrays?
        bint __no_copy

        float __matrix_transform_time


        # we keep internally two arrays for the complex numbers: this is required by CHOLMOD...
        FLOAT64_t * csc_rval
        FLOAT64_t * csc_ival

