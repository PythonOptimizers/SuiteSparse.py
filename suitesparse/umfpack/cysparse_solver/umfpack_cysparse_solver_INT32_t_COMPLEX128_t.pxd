from suitesparse.umfpack.umfpack_solver_base_INT32_t_COMPLEX128_t cimport UmfpackSolverBase_INT32_t_COMPLEX128_t

from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_COMPLEX128_t cimport CSCSparseMatrix_INT32_t_COMPLEX128_t

cdef class UmfpackCysparseSolver_INT32_t_COMPLEX128_t(UmfpackSolverBase_INT32_t_COMPLEX128_t):
    cdef:
        # Matrix A in CSC format
        CSCSparseMatrix_INT32_t_COMPLEX128_t csc_mat

        float __matrix_transform_time