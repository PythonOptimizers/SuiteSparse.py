

cdef class Solver_INT32_t_FLOAT64_t:
    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def factorize(self, *args, **kwargs):
        raise NotImplementedError()