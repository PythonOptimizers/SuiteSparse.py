"""
Common solver interface in Cython.
"""

cdef class Solver_INT64_t_COMPLEX128_t:
    cdef:
        str __solver_name
        object __A
        bint __verbose

        bint __analyzed
        bint __factorized

        float __solve_time
        float __analyze_time
        float __factorize_time