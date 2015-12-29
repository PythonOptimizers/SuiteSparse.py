"""
Common solver interface in Cython.
"""

cdef class Solver_INT32_t_COMPLEX128_t:
    cdef:
        str __solver_name
        object __A