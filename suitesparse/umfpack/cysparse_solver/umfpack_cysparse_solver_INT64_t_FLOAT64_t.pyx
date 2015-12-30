from __future__ import print_function

from suitesparse.umfpack.umfpack_solver_base_INT64_t_FLOAT64_t cimport UmfpackSolverBase_INT64_t_FLOAT64_t

from cysparse.sparse.s_mat cimport PyLLSparseMatrix_Check, PyCSCSparseMatrix_Check, PyCSRSparseMatrix_Check

#from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT64_t cimport PyCSCSparseMatrix_Check
#from cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_FLOAT64_t cimport PyCSRSparseMatrix_Check


cdef class UmfpackCysparseSolver_INT64_t_FLOAT64_t(UmfpackSolverBase_INT64_t_FLOAT64_t):
    def __cinit__(self, A, **kwargs):

        if PyLLSparseMatrix_Check(self.__A):
            pass
        elif PyCSCSparseMatrix_Check(self.__A):
            pass
        elif PyCSRSparseMatrix_Check(self.__A):
            pass
        else:
            matrix_type = "unknown"
            try:
                matrix_type = self.__A.base_type_str
            except:
                matrix_type = type(self.__A)

            raise NotImplementedError("CySparse matrix type '%s' not recognized" % matrix_type)

        if self.__verbose:
            print("I'm talking a lot!")

    def _solve(self, b):
        print("Calling real solve with ",)
        print(b)

    def _factorize(self, **kwargs):
        print("Called real factorize")

    def _analyze(self, **kwargs):
        print("Called real analyze")

    def _specialized_stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        lines = []

        lines.append("CySparse matrix type: %s" % self.__A.base_type_str)

        return '\n'.join(lines)