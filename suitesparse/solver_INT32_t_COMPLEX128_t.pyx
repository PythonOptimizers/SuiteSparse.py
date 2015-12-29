

cdef class Solver_INT32_t_COMPLEX128_t:
    def __cinit__(self, A, **kwargs):
        self.__A = A
        self.__verbose = kwargs.get('verbose', False)

    
    @property
    def solver_name(self):
        return self.__solver_name

    
    @property
    def A(self):
        return self.__A

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def factorize(self, *args, **kwargs):
        raise NotImplementedError()