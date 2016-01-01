"""
Common solver interface in Cython.
"""

cdef class Solver_INT64_t_FLOAT64_t:
    cdef:
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