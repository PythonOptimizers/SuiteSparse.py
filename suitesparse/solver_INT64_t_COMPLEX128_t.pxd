"""
Common solver interface in Cython.
"""
# cimport the right types here for your library
from suitesparse.common_types.suitesparse_types cimport *

cdef class Solver_INT64_t_COMPLEX128_t:
    cdef:

        # common internal attributes
        INT64_t nrow
        INT64_t ncol
        INT64_t nnz

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

    cdef bint check_common_attributes(self)