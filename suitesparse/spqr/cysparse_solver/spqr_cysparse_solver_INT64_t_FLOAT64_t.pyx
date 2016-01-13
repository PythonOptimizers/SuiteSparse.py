from __future__ import print_function

from suitesparse.spqr.spqr_solver_base_INT64_t_FLOAT64_t cimport SPQRSolverBase_INT64_t_FLOAT64_t
from suitesparse.common_types.suitesparse_types cimport *

from suitesparse.cholmod.cysparse_solver.cholmod_cysparse_solver_INT64_t_FLOAT64_t cimport populate1_cholmod_sparse_struct_with_CSCSparseMatrix, populate2_cholmod_sparse_struct_with_CSCSparseMatrix


from suitesparse.spqr.spqr_common import SPQR_SYS_DICT

from cysparse.sparse.s_mat cimport PySparseMatrix_Check, PyLLSparseMatrix_Check, PyCSCSparseMatrix_Check, PyCSRSparseMatrix_Check
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT64_t cimport CSCSparseMatrix_INT64_t_FLOAT64_t, MakeCSCSparseMatrix_INT64_t_FLOAT64_t
from cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_FLOAT64_t cimport CSRSparseMatrix_INT64_t_FLOAT64_t, MakeCSRSparseMatrix_INT64_t_FLOAT64_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time


cdef class SPQRCysparseSolver_INT64_t_FLOAT64_t(SPQRSolverBase_INT64_t_FLOAT64_t):
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, A, **kwargs):
        assert PySparseMatrix_Check(A), "Matrix A is not recognized as a CySparse sparse matrix"

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz

        self.__matrix_transform_time = 0.0

        start_time = time.clock()

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

        # We don't allow internal copies of CSC C arrays
        self.__no_copy = True

        # transform CSC mat into CHOLMOD sparse_struct
        # common attributes for real and complex matrices
        populate1_cholmod_sparse_struct_with_CSCSparseMatrix(self.sparse_struct, self.csc_mat, no_copy=self.__no_copy)


        populate2_cholmod_sparse_struct_with_CSCSparseMatrix(self.sparse_struct, self.csc_mat, no_copy=self.__no_copy)


        self.__matrix_transform_time += (time.clock() - start_time)

        # this is for the main stats from the Solver class
        self.__specialized_solver_time += self.__matrix_transform_time

        # Internal test:
        # common attributes
        assert self.check_common_attributes(), "Internal problem: Common attributes are wrong"
        # Control the matrix is fine
        assert self.check_matrix(), "Internal problem: matrix has some problems"

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        assert self.__no_copy, "Internal error: the code only works with non copied internal arrays"

        # numeric and symbolic UMFPACK objects are being taken care by parent class
        # self.csc_mat will be deleted with this object if it was created internally

        # common_struct and factor_strcut are taken care in parent class


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
        lines.append("nrow, ncol = (%d, %d)" % (self.nrow, self.ncol) )
        lines.append("nnz = %s" % self.nnz)
        lines.append("Matrix transformation: %f secs" % self.__matrix_transform_time)

        return '\n'.join(lines)