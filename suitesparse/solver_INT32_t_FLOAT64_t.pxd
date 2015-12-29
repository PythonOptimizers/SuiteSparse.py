"""
Common solver interface in Cython.
"""

cdef class Solver_INT32_t_FLOAT64_t:
    cdef:
        str __solver_name
        object __A
        bint __verbose

        bint __analyzed
        bint __factorized