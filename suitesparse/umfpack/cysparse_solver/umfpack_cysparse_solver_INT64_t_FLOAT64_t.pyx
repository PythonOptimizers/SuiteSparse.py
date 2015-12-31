from __future__ import print_function

from suitesparse.umfpack.umfpack_solver_base_INT64_t_FLOAT64_t cimport UmfpackSolverBase_INT64_t_FLOAT64_t
from suitesparse.common_types.suitesparse_types cimport *


from cysparse.sparse.s_mat cimport PySparseMatrix_Check, PyLLSparseMatrix_Check, PyCSCSparseMatrix_Check, PyCSRSparseMatrix_Check

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class UmfpackCysparseSolver_INT64_t_FLOAT64_t(UmfpackSolverBase_INT64_t_FLOAT64_t):
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, A, **kwargs):

        assert PySparseMatrix_Check(A), "Matrix A is not recognized as a CySparse sparse matrix"

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz


        if PyLLSparseMatrix_Check(self.__A):
            # transfrom matrix into CSC
            self.csc_mat = self.__A.to_csc()

        elif PyCSCSparseMatrix_Check(self.__A):
            self.csc_mat = self.__A
        elif PyCSRSparseMatrix_Check(self.__A):
            # transfrom matrix into CSC
            self.csc_mat = self.__A.to_csc()
        else:
            matrix_type = "unknown"
            try:
                matrix_type = self.__A.base_type_str
            except:
                matrix_type = type(self.__A)

            raise NotImplementedError("CySparse matrix type '%s' not recognized" % matrix_type)

        # take internal arrays
        self.ind = <INT64_t *> self.csc_mat.ind

        self.row = <INT64_t *> self.csc_mat.row


        self.val = <FLOAT64_t *> self.csc_mat.val


        # Control the matrix is fine
        self.check_matrix()

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):

        # numeric and symbolic UMFPACK objects are being taken care by parent class
        # self.csc_mat will be deleted with this object if it was created internally


        pass


    ####################################################################################################################
    # CALLBACKS
    ####################################################################################################################
    def _specialized_stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization with a :program:`CySparse` sparse matrix.
        """
        lines = []

        lines.append("CySparse matrix type: %s" % self.__A.base_type_str)
        lines.append("(nrow, ncol) = (%d, %d)" % (self.nrow, self.ncol) )
        lines.append("nnz = %s" % self.nnz)

        return '\n'.join(lines)