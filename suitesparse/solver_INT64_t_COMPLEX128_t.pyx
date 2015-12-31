from __future__ import print_function

from cpython cimport Py_INCREF, Py_DECREF

import time

# TODO: add stats
# TODO: add timing

cdef class Solver_INT64_t_COMPLEX128_t:
    def __cinit__(self, A, **kwargs):
        self.__A = A
        Py_INCREF(self.__A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.__verbose = kwargs.get('verbose', False)

        self.__analyzed = False
        self.__factorized = False

        ###  Stats  ###
        # performance
        # time.clock() -> in seconds
        self.__solve_time = 0.0
        self.__analyze_time = 0.0
        self.__factorize_time = 0.0

    
    @property
    def solver_name(self):
        return self.__solver_name

    
    @property
    def solver_version(self):
        return self.__solver_version

    
    @property
    def A(self):
        return self.__A

    def __dealloc__(self):
        """

        """
        Py_DECREF(self.__A) # release ref

    ####################################################################################################################
    # Common functions
    ####################################################################################################################
    def solve(self, *args, **kwargs):

        start_time = time.clock()
        # b could be a sparse/denses matrix/vector
        b = self._solve(*args, **kwargs)
        self.__solve_time = time.clock() - start_time

        return b

    def analyze(self, *args, **kwargs):
        force_analyze = kwargs.get('force_analyze', False)
        if self.__verbose and force_analyze:
            print("Force analyze.")

        if force_analyze or not self.__analyzed:
            start_time = time.clock()
            self._analyze(*args, **kwargs)
            self.__analyze_time = time.clock() - start_time
            self.__analyzed = True

    def factorize(self, *args, **kwargs):
        force_factorize = kwargs.get('force_factorize', False)

        if self.__verbose and force_factorize:
            print("Force factorize.")

        self.analyze(*args, **kwargs)

        if force_factorize or not self.__factorized:
            start_time = time.clock()
            self._factorize(*args, **kwargs)
            self.__factorize_time = time.clock() - start_time
            self.__factorized = True

    ####################################################################################################################
    # Special functions
    ####################################################################################################################
    def __call__(self, B):
        return self.solve(B)

    def __mul__(self, B):
        return self.solve(B)

    ####################################################################################################################
    # Common statistics
    ####################################################################################################################
    def stats(self, *args, **kwargs):
        """
        General statistics about the last factorization.

        Args:
            OUT: output stream.
        """
        lines = []
        lines.append("General statistics")
        lines.append("==================")
        lines.append("")
        lines.append("Solver: %s" % self.__solver_name)
        lines.append("")
        lines.append("Performance:")
        lines.append("------------")
        lines.append("")
        lines.append("Analyze: %f" % self.__analyze_time)
        lines.append("Factorize: %f" % self.__factorize_time)
        lines.append("Solve: %f" % self.__solve_time)
        lines.append("")
        lines.append("Specialized statistics")
        lines.append("======================")
        lines.append("")
        lines.append(self._stats(*args, **kwargs))

        return '\n'.join(lines)

    ####################################################################################################################
    # Callbacks
    ####################################################################################################################
    def _solve(self, *args, **kwargs):
        raise NotImplementedError()

    def _analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def _factorize(self, *args, **kwargs):
        raise NotImplementedError()

    def _stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        raise NotImplementedError()