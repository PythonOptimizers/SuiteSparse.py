from suitesparse.cholmod.cholmod_solver_base_INT32_t_COMPLEX128_t cimport CholmodSolverBase_INT32_t_COMPLEX128_t

from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_COMPLEX128_t cimport CSCSparseMatrix_INT32_t_COMPLEX128_t

# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t


cdef class CholmodCysparseSolver_INT32_t_COMPLEX128_t(CholmodSolverBase_INT32_t_COMPLEX128_t):
    cdef:
        # Matrix A in CSC format
        CSCSparseMatrix_INT32_t_COMPLEX128_t csc_mat

        float __matrix_transform_time