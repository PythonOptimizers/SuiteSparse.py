from suitesparse.solver_INT64_t_COMPLEX128_t cimport Solver_INT64_t_COMPLEX128_t

from suitesparse.common_types.suitesparse_types cimport *

cdef extern from "umfpack.h":

    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO


cdef class UmfpackSolverBase_INT64_t_COMPLEX128_t(Solver_INT64_t_COMPLEX128_t):
    cdef:
        INT64_t nrow
        INT64_t ncol
        INT64_t nnz

        # UMFPACK takes a C CSC matrix object
        INT64_t * ind
        bint own_ind_memory  # does ind belong to the class or it is external memory?
        INT64_t * row
        bint own_row_memory  # does row belong to the class or it is external memory?

        FLOAT64_t * rval
        bint own_rval_memory  # does rval belong to the class or it is external memory?
        FLOAT64_t * ival
        bint own_ival_memory  # does ival belong to the class or it is external memory?

        bint internal_real_arrays_computed


        # UMFPACK opaque objects
        void * symbolic

        void * numeric

        # Control and Info arrays
        public double info[UMFPACK_INFO]
        public double control[UMFPACK_CONTROL]


    cdef check_matrix(self)