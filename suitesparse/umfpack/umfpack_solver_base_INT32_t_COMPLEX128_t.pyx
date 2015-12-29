from suitesparse.solver_INT32_t_COMPLEX128_t cimport Solver_INT32_t_COMPLEX128_t

cdef class UmfpackSolverBase_INT32_t_COMPLEX128_t(Solver_INT32_t_COMPLEX128_t):
    def __cinit__(self, A, **kwargs):
        self.__solver_name = 'UMFPACK'