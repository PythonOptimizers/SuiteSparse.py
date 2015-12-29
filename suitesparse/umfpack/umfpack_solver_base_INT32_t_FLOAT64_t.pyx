from suitesparse.solver_INT32_t_FLOAT64_t cimport Solver_INT32_t_FLOAT64_t

cdef class UmfpackSolverBase_INT32_t_FLOAT64_t(Solver_INT32_t_FLOAT64_t):
    def __cinit__(self, A, **kwargs):
        self.__solver_name = 'UMFPACK'