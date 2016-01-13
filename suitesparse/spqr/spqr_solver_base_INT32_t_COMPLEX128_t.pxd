from suitesparse.solver_INT32_t_COMPLEX128_t cimport Solver_INT32_t_COMPLEX128_t

from suitesparse.common_types.suitesparse_types cimport *

from  suitesparse.spqr.spqr_common cimport *


# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t

cdef extern from "cholmod.h":
    # COMMON STRUCT
    ctypedef struct cholmod_common:
        pass

        # SPARSE MATRIX
    ctypedef struct cholmod_sparse:
        pass

cdef extern from  "SuiteSparseQR_C.h":

    ####################################################################################################################
    # EXPERT MODE
    ####################################################################################################################
    # A real or complex QR factorization, computed by SuiteSparseQR_C_factorize
    ctypedef struct SuiteSparseQR_C_factorization:
        int xtype                  # CHOLMOD_REAL or CHOLMOD_COMPLEX
        void *factors              # from SuiteSparseQR_factorize <double> or SuiteSparseQR_factorize <Complex>

cdef class SPQRSolverBase_INT32_t_COMPLEX128_t(Solver_INT32_t_COMPLEX128_t):
    cdef:

        cholmod_common * common_struct
        cholmod_sparse * sparse_struct

        SuiteSparseQR_C_factorization * factor_struct

    cpdef bint check_common(self)
    cpdef bint check_matrix(self)
    cpdef bint check_factor(self)