from suitesparse.solver_INT32_t_FLOAT64_t cimport Solver_INT32_t_FLOAT64_t

from suitesparse.common_types.suitesparse_types cimport *

cdef extern from "umfpack.h":

    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO


cdef class UmfpackSolverBase_INT32_t_FLOAT64_t(Solver_INT32_t_FLOAT64_t):
    cdef:
        INT32_t nrow
        INT32_t ncol
        INT32_t nnz

        # UMFPACK takes a C CSC matrix object
        INT32_t * ind
        bint own_ind_memory  # does ind belong to the class or it is external memory?
        INT32_t * row
        bint own_row_memory  # does row belong to the class or it is external memory?

        FLOAT64_t * val
        bint own_val_memory


        # UMFPACK opaque objects
        void * symbolic

        void * numeric

        # Control and Info arrays
        public double info[UMFPACK_INFO]
        public double control[UMFPACK_CONTROL]


    cdef check_matrix(self)