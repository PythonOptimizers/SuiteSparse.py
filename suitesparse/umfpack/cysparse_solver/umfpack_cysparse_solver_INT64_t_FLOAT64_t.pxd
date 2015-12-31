from suitesparse.umfpack.umfpack_solver_base_INT64_t_FLOAT64_t cimport UmfpackSolverBase_INT64_t_FLOAT64_t

from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT64_t cimport CSCSparseMatrix_INT64_t_FLOAT64_t

cdef class UmfpackCysparseSolver_INT64_t_FLOAT64_t(UmfpackSolverBase_INT64_t_FLOAT64_t):
    cdef:
        # Matrix A in CSC format
        CSCSparseMatrix_INT64_t_FLOAT64_t csc_mat
