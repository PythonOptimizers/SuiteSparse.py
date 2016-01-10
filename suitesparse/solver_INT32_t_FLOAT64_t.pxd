"""
Common solver interface in Cython.
"""
# cimport the right types here for your library
from suitesparse.common_types.suitesparse_types cimport *

cdef class Solver_INT32_t_FLOAT64_t:
    cdef:

        # common internal attributes
        INT32_t nrow
        INT32_t ncol
        INT32_t nnz

        str __solver_name
        str __solver_version
        object __A
        bint __verbose

        bint __analyzed
        bint __factorized

        float __solve_time
        float __analyze_time
        float __factorize_time
        float __specialized_solver_time

    cdef check_common_attributes(self)