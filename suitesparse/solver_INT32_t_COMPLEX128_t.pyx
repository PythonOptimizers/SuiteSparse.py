

cdef class Solver_INT32_t_COMPLEX128_t:
    def __cinit__(self, A, **kwargs):
        self.__A = A
        self.__verbose = kwargs.get('verbose', False)

        self.__analyzed = False
        self.__factorized = False

    
    @property
    def solver_name(self):
        return self.__solver_name

    
    @property
    def A(self):
        return self.__A

    ####################################################################################################################
    # Common functions
    ####################################################################################################################
    def solve(self, *args, **kwargs):
        # TODO: modify this
        return self._solve(*args, **kwargs)

    def analyze(self, *args, **kwargs):
        force_analyze = kwargs.get('force_analyze', False)

        if force_analyze or not self.__analyzed:
            self._analyze(*args, **kwargs)
            self.__analyzed = True

    def factorize(self, *args, **kwargs):
        force_factorize = kwargs.get('force_factorize', False)

        self.analyze(*args, **kwargs)

        if force_factorize or not self.__factorized:
            self._factorize(*args, **kwargs)
            self.__factorized = True

    def __call__(self, B):
        return self.solve(B)

    def __mul__(self, B):
        return self.solve(B)

    ####################################################################################################################
    # Callbacks
    ####################################################################################################################
    def _solve(self, *args, **kwargs):
        raise NotImplementedError()

    def _analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def _factorize(self, *args, **kwargs):
        raise NotImplementedError()