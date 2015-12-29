from suitesparse.umfpack.umfpack_solver_base_INT64_t_FLOAT64_t cimport UmfpackSolverBase_INT64_t_FLOAT64_t

cdef class UmfpackCysparseSolver_INT64_t_FLOAT64_t(UmfpackSolverBase_INT64_t_FLOAT64_t):
    def __cinit__(self, A, **kwargs):
        if self.__verbose:
            print "I'm talking a lot!"

    def _solve(self, b):
        print "Calling real solve with ",
        print b

    def _factorize(self, **kwargs):
        print "Called real factorize"

    def _analyze(self, **kwargs):
        print "Called real analyze"