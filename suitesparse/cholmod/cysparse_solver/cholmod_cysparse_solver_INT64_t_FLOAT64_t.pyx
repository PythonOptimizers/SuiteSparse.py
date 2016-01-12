from __future__ import print_function

from suitesparse.cholmod.cholmod_solver_base_INT64_t_FLOAT64_t cimport CHOLMODSolverBase_INT64_t_FLOAT64_t
from suitesparse.common_types.suitesparse_types cimport *

from suitesparse.cholmod.cholmod_common import CHOLMOD_SYS_DICT

from cysparse.sparse.s_mat cimport PySparseMatrix_Check, PyLLSparseMatrix_Check, PyCSCSparseMatrix_Check, PyCSRSparseMatrix_Check
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT64_t cimport CSCSparseMatrix_INT64_t_FLOAT64_t, MakeCSCSparseMatrix_INT64_t_FLOAT64_t
from cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_FLOAT64_t cimport CSRSparseMatrix_INT64_t_FLOAT64_t, MakeCSRSparseMatrix_INT64_t_FLOAT64_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time

cdef extern from "cholmod.h":
    # we only use REAL and ZOMPLEX
    cdef enum:
        CHOLMOD_PATTERN  	# pattern only, no numerical values
        CHOLMOD_REAL		# a real matrix
        CHOLMOD_COMPLEX     # a complex matrix (ANSI C99 compatible)
        CHOLMOD_ZOMPLEX     # a complex matrix (MATLAB compatible)

    # itype: we only use INT and LONG
    cdef enum:
        CHOLMOD_INT         # all integer arrays are int
        CHOLMOD_INTLONG     # most are int, some are SuiteSparse_long
        CHOLMOD_LONG        # all integer arrays are SuiteSparse_long

    # dtype: float or double
    # dtype: float or double
    cdef enum:
        CHOLMOD_DOUBLE      # all numerical values are double
        CHOLMOD_SINGLE



    int cholmod_l_start(cholmod_common *Common)
    int cholmod_l_finish(cholmod_common *Common)

    int cholmod_l_defaults(cholmod_common *Common)

    # Common struct
    int cholmod_l_check_common(cholmod_common *Common)
    int cholmod_l_print_common(const char *name, cholmod_common *Common)

    # Sparse struct
    int cholmod_l_check_sparse(cholmod_sparse *A, cholmod_common *Common)
    int cholmod_l_print_sparse(cholmod_sparse *A, const char *name, cholmod_common *Common)
    int cholmod_l_free_sparse(cholmod_sparse **A, cholmod_common *Common)

    # Dense struct
    int cholmod_l_free_dense(cholmod_dense **X, cholmod_common *Common)

    # Factor struct
    int cholmod_l_check_factor(cholmod_factor *L, cholmod_common *Common)
    int cholmod_l_print_factor(cholmod_factor *L, const char *name, cholmod_common *Common)
    #int cholmod_l_free_factor()
    # factor_to_sparse

    # Memory management
    void * cholmod_l_free(size_t n, size_t size,	void *p,  cholmod_common *Common)



########################################################################################################################
# CHOLMOD HELPERS
########################################################################################################################
##################################################################
# FROM CSCSparseMatrix -> cholmod_sparse
##################################################################
# Populating a sparse matrix in CHOLMOD is done in two steps:
# - first (populate1), we give the common attributes and
# - second (populate2), we split the values array in two if needed (complex case) and give the values (real or complex).

cdef populate1_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct, CSCSparseMatrix_INT64_t_FLOAT64_t csc_mat, bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_FLOAT64_t` matrix.

    First part: common attributes for both real and complex matrices.

    Note:
        We only use the ``cholmod_sparse`` **packed** and **sorted** version.
    """
    assert no_copy, "The version with copy is not implemented yet..."

    assert(csc_mat.are_row_indices_sorted()), "We only use CSC matrices with internal row indices sorted. The non sorted version is not implemented yet."

    sparse_struct.nrow = csc_mat.nrow
    sparse_struct.ncol = csc_mat.ncol
    sparse_struct.nzmax = csc_mat.nnz

    sparse_struct.p = csc_mat.ind
    sparse_struct.i = csc_mat.row

    # TODO: change this when we'll accept symmetric matrices **without** symmetric storage scheme
    if csc_mat.is_symmetric:
        sparse_struct.stype = -1
    else:
        sparse_struct.stype = 0

    # itype: can be CHOLMOD_INT or CHOLMOD_LONG: we don't use the mixed version CHOLMOD_INTLONG

    sparse_struct.itype = CHOLMOD_LONG


    sparse_struct.sorted = 1                                 # TRUE if columns are sorted, FALSE otherwise
    sparse_struct.packed = 1                                 # We use the packed CSC version: **no** need to construct
                                                             # the nz (array with number of non zeros by column)



cdef populate2_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct, CSCSparseMatrix_INT64_t_FLOAT64_t csc_mat, bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_FLOAT64_t` matrix.

    Second part: Non common attributes for complex matrices.


    """
    assert no_copy, "The version with copy is not implemented yet..."

    sparse_struct.x = csc_mat.val

    sparse_struct.xtype = CHOLMOD_REAL                    # CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    sparse_struct.dtype = CHOLMOD_DOUBLE



##################################################################
# FROM cholmod_sparse -> CSCSparseMatrix
##################################################################
cdef CSCSparseMatrix_INT64_t_FLOAT64_t cholmod_sparse_to_CSCSparseMatrix_INT64_t_FLOAT64_t(cholmod_sparse * sparse_struct, bint no_copy=False):
    """
    Convert a ``cholmod`` sparse struct to a :class:`CSCSparseMatrix_INT64_t_FLOAT64_t`.

    """
    # TODO: generalize to any cholmod sparse structure, with or without copy
    # TODO: generalize to complex case
    # TODO: remove asserts
    assert sparse_struct.sorted == 1, "We only accept cholmod_sparse matrices with sorted indices"
    assert sparse_struct.packed == 1, "We only accept cholmod_sparse matrices with packed indices"




    cdef:
        CSCSparseMatrix_INT64_t_FLOAT64_t csc_mat
        INT64_t nrow
        INT64_t ncol
        INT64_t nnz
        bint store_symmetric = False

        # internal arrays of the CSC matrix
        INT64_t * ind
        INT64_t * row

        # internal arrays of the cholmod sparse matrix
        INT64_t * ind_cholmod
        INT64_t * row_cholmod

        # internal array for the CSC matrix
        FLOAT64_t * val

        # internal array for the cholmod sparse matrix
        FLOAT64_t * val_cholmod


        INT64_t j, k

    nrow = sparse_struct.nrow
    ncol = sparse_struct.ncol
    nnz = sparse_struct.nzmax

    if sparse_struct.stype == 0:
        store_symmetric = False
    elif sparse_struct.stype < 0:
        store_symmetric == True
    else:
        raise NotImplementedError('We do not accept cholmod square symmetric sparse matrix with upper triangular part filled in.')

    ##################################### NO COPY ######################################################################
    if no_copy:
        ind = <INT64_t *> sparse_struct.p
        row = <INT64_t *> sparse_struct.i

        val = <FLOAT64_t *> sparse_struct.x

    ##################################### WITH COPY ####################################################################
    else:   # we do a copy

        ind_cholmod = <INT64_t * > sparse_struct.p
        row_cholmod = <INT64_t * > sparse_struct.i

        ind = <INT64_t *> PyMem_Malloc((ncol + 1) * sizeof(INT64_t))

        if not ind:
            raise MemoryError()

        row = <INT64_t *> PyMem_Malloc(nnz * sizeof(INT64_t))

        if not row:
            PyMem_Free(ind)
            PyMem_Free(row)

            raise MemoryError()


        for j from 0 <= j <= ncol:
            ind[j] = ind_cholmod[j]

        for k from 0 <= k < nnz:
            row[k] = row_cholmod[k]


        val_cholmod = <FLOAT64_t * > sparse_struct.x

        val = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))

        if not val:
            PyMem_Free(ind)
            PyMem_Free(row)
            PyMem_Free(val)

            raise MemoryError()

        for k from 0 <= k < nnz:
            val[k] = val_cholmod[k]



    csc_mat = MakeCSCSparseMatrix_INT64_t_FLOAT64_t(nrow=nrow, ncol=ncol, nnz=nnz, ind=ind, row=row, val=val, store_symmetric=store_symmetric, store_zero=False)


    return csc_mat


cdef class CHOLMODCysparseSolver_INT64_t_FLOAT64_t(CHOLMODSolverBase_INT64_t_FLOAT64_t):
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, A, **kwargs):
        assert PySparseMatrix_Check(A), "Matrix A is not recognized as a CySparse sparse matrix"

        self.nrow = A.nrow
        self.ncol = A.ncol

        # test if we can use CHOLMOD
        assert self.nrow == self.ncol, "Only square matrices are handled in CHOLMOD"
        # TODO: change this. This assumption is too strong
        assert self.A.store_symmetric, "Only symmetric matrices (using the symmetric storage scheme) are handled in CHOLMOD"

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