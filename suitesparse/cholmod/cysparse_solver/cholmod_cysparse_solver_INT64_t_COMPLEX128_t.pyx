from __future__ import print_function

from suitesparse.cholmod.cholmod_solver_base_INT64_t_COMPLEX128_t cimport CholmodSolverBase_INT64_t_COMPLEX128_t
from suitesparse.common_types.suitesparse_types cimport *

from suitesparse.common_types.suitesparse_generic_types cimport split_array_complex_values_kernel_INT64_t_COMPLEX128_t, join_array_complex_values_kernel_INT64_t_COMPLEX128_t

from suitesparse.cholmod.cholmod_common import CHOLMOD_SYS_DICT

from cysparse.sparse.s_mat cimport PySparseMatrix_Check, PyLLSparseMatrix_Check, PyCSCSparseMatrix_Check, PyCSRSparseMatrix_Check
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t cimport CSCSparseMatrix_INT64_t_COMPLEX128_t, MakeCSCSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_COMPLEX128_t cimport CSRSparseMatrix_INT64_t_COMPLEX128_t, MakeCSRSparseMatrix_INT64_t_COMPLEX128_t

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


    # COMMON STRUCT
    ctypedef struct cholmod_common:
        #######################################################
        # parameters for symbolic/numeric factorization and update/downdate
        #######################################################
        double dbound
        double grow0
        double grow1
        size_t grow2
        size_t maxrank
        double supernodal_switch
        int supernodal
        int final_asis
        int final_super
        int final_ll
        int final_pack
        int final_monotonic
        int final_resymbol
        double zrelax [3]
        size_t nrelax [3]
        int prefer_zomplex
        int prefer_upper
        int quick_return_if_not_posdef
        int prefer_binary

        #######################################################
        # printing and error handling options
        #######################################################
        int print_ "print"
        int precise
        int try_catch

        #######################################################
        # workspace
        #######################################################
        size_t nrow
        SuiteSparse_long mark

        #######################################################
        # Statistics
        #######################################################

        int status 	        # error code
        double fl 		    # LL' flop count from most recent analysis
        double lnz 	        # fundamental nz in L
        double anz 	        # nonzeros in tril(A) if A is symmetric/lower,
                            # triu(A) if symmetric/upper, or tril(A*A') if
                            # unsymmetric, in last call to cholmod_analyze.
        double modfl 	    # flop count from most recent update/downdate/
                            # rowadd/rowdel (excluding flops to modify the
                            # solution to Lx=b, if computed)
        size_t malloc_count    # # of objects malloc'ed minus the # free'd
        size_t memory_usage    # peak memory usage in bytes
        size_t memory_inuse    # current memory usage in bytes

        double nrealloc_col    # of column reallocations
        double nrealloc_factor # of factor reallocations due to col. reallocs
        double ndbounds_hit    # of times diagonal modified by dbound

        double rowfacfl 	    # of flops in last call to cholmod_rowfac
        double aatfl 	        # of flops to compute A(:,f)*A(:,f)'

        int called_nd 	    # TRUE if the last call to
                            # cholmod_analyze called NESDIS or METIS.
        int blas_ok         # FALSE if BLAS int overflow TRUE otherwise

        # SuiteSparseQR control parameters:

        double SPQR_grain       # task size is >= max (total flops / grain)
        double SPQR_small       # task size is >= small
        int SPQR_shrink         # controls stack realloc method
        int SPQR_nthreads       # number of TBB threads, 0 = auto

        # SuiteSparseQR statistics

        # was other1 [0:3]
        double SPQR_flopcount          # flop count for SPQR
        double SPQR_analyze_time       # analysis time in seconds for SPQR
        double SPQR_factorize_time     # factorize time in seconds for SPQR
        double SPQR_solve_time         # backsolve time in seconds

        # was SPQR_xstat [0:3]
        double SPQR_flopcount_bound    # upper bound on flop count
        double SPQR_tol_used           # tolerance used
        double SPQR_norm_E_fro         # Frobenius norm of dropped entries

        # was SPQR_istat [0:9]
        SuiteSparse_long SPQR_istat [10]

        #######################################################
        # GPU configuration
        #######################################################
        int useGPU
        size_t maxGpuMemBytes
        double maxGpuMemFraction



    # SPARSE MATRIX
    ctypedef struct cholmod_sparse:
        size_t nrow                    # the matrix is nrow-by-ncol
        size_t ncol
        size_t nzmax                   # maximum number of entries in the matrix

        # pointers to int or SuiteSparse_long
        void *p                        # p [0..ncol], the column pointers
        void *i                        # i [0..nzmax-1], the row indices

        # we only use the packed form
        #void *nz                      # nz [0..ncol-1], the # of nonzeros in each col.  In
			                           # packed form, the nonzero pattern of column j is in
	                                   # A->i [A->p [j] ... A->p [j+1]-1].  In unpacked form, column j is in
	                                   # A->i [A->p [j] ... A->p [j]+A->nz[j]-1] instead.  In both cases, the
	                                   # numerical values (if present) are in the corresponding locations in
	                                   # the array x (or z if A->xtype is CHOLMOD_ZOMPLEX).

        # pointers to double or float
        void *x                        #  size nzmax or 2*nzmax, if present
        void *z                        #  size nzmax, if present

        int stype                      #  Describes what parts of the matrix are considered
		                               #
                                       # 0:  matrix is "unsymmetric": use both upper and lower triangular parts
                                       #     (the matrix may actually be symmetric in pattern and value, but
                                       #     both parts are explicitly stored and used).  May be square or
                                       #     rectangular.
                                       # >0: matrix is square and symmetric, use upper triangular part.
                                       #     Entries in the lower triangular part are ignored.
                                       # <0: matrix is square and symmetric, use lower triangular part.
                                       #     Entries in the upper triangular part are ignored.

                                       # Note that stype>0 and stype<0 are different for cholmod_sparse and
                                       # cholmod_triplet.  See the cholmod_triplet data structure for more
                                       # details.

        int itype                      # CHOLMOD_INT: p, i, and nz are int.
			                           # CHOLMOD_INTLONG: p is SuiteSparse_long,
                                       #                  i and nz are int.
			                           # CHOLMOD_LONG:    p, i, and nz are SuiteSparse_long

        int xtype                      # pattern, real, complex, or zomplex
        int dtype 		               # x and z are double or float
        int sorted                     # TRUE if columns are sorted, FALSE otherwise
        int packed                     # TRUE if packed (nz ignored), FALSE if unpacked
			                           # (nz is required)

    # DENSE MATRIX
    ctypedef struct cholmod_dense:
        size_t nrow                    # the matrix is nrow-by-ncol
        size_t ncol
        size_t nzmax                   # maximum number of entries in the matrix
        size_t d                       # leading dimension (d >= nrow must hold)
        void *x                        # size nzmax or 2*nzmax, if present
        void *z                        # size nzmax, if present
        int xtype                      # pattern, real, complex, or zomplex
        int dtype                      # x and z double or float

    # FACTOR
    ctypedef struct cholmod_factor:
        pass

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

cdef populate1_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct, CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat, bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t` matrix.

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



cdef populate2_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct,
                                                              CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat,
                                                              FLOAT64_t * csc_mat_rval,
                                                              FLOAT64_t * csc_mat_ival,
                                                              bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t` matrix.

    Second part: Non common attributes for complex matrices.

    Note:
        We only use the ``cholmod_sparse`` **packed** version.
    """
    assert no_copy, "The version with copy is not implemented yet..."


    sparse_struct.x = csc_mat_rval
    sparse_struct.z = csc_mat_ival

    sparse_struct.xtype = CHOLMOD_ZOMPLEX                    # CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    sparse_struct.dtype = CHOLMOD_DOUBLE




##################################################################
# FROM cholmod_sparse -> CSCSparseMatrix
##################################################################
cdef CSCSparseMatrix_INT64_t_COMPLEX128_t cholmod_sparse_to_CSCSparseMatrix_INT64_t_COMPLEX128_t(cholmod_sparse * sparse_struct, bint no_copy=False):
    """
    Convert a ``cholmod`` sparse struct to a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t`.

    """
    # TODO: generalize to any cholmod sparse structure, with or without copy
    # TODO: generalize to complex case
    # TODO: remove asserts
    assert sparse_struct.sorted == 1, "We only accept cholmod_sparse matrices with sorted indices"
    assert sparse_struct.packed == 1, "We only accept cholmod_sparse matrices with packed indices"

    assert sparse_struct.xtype == CHOLMOD_ZOMPLEX, "We only accept cholmod_sparse matrices with zomplex"


    cdef:
        CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat
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

        # internal arrays for the CSC matrix
        FLOAT64_t * valx
        FLOAT64_t * valz

        COMPLEX128_t * val_complex

        # internal arrays for the cholmod sparse matrix
        FLOAT64_t * valx_cholmod
        FLOAT64_t * valz_cholmod



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

        valx = <FLOAT64_t *> sparse_struct.x
        valz = <FLOAT64_t *> sparse_struct.z

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



        valx_cholmod = <FLOAT64_t *> sparse_struct.x
        valz_cholmod = <FLOAT64_t *> sparse_struct.z

        valx = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))

        if not valx:
            PyMem_Free(ind)
            PyMem_Free(row)
            PyMem_Free(valx)

            raise MemoryError()

        valz = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))

        if not valz:
            PyMem_Free(ind)
            PyMem_Free(row)
            PyMem_Free(valx)
            PyMem_Free(valz)

            raise MemoryError()

        for k from 0 <= k < nnz:
            valx[k] = valx_cholmod[k]
            valz[k] = valz_cholmod[k]



    raise NotImplementedError('Complex case not implemented yet...')


    return csc_mat


cdef class CholmodCysparseSolver_INT64_t_COMPLEX128_t(CholmodSolverBase_INT64_t_COMPLEX128_t):
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

        # take internal arrays
        self.ind = <INT64_t *> self.csc_mat.ind

        self.row = <INT64_t *> self.csc_mat.row


        # OPTIMIZATION: we could delay the creation of both arrays until
        # analyze, factorize or solve
        cdef:
            FLOAT64_t * rval
            FLOAT64_t * ival

        rval = <FLOAT64_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT64_t))
        if not rval:
            raise MemoryError()
        self.rval = rval

        ival = <FLOAT64_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT64_t))
        if not ival:
            PyMem_Free(rval)
            raise MemoryError()
        self.ival = ival

        split_array_complex_values_kernel_INT64_t_COMPLEX128_t(self.csc_mat.val, self.nnz,
                                                                       self.rval, self.nnz,
                                                                       self.ival, self.nnz)


        self.__matrix_transform_time += (time.clock() - start_time)

        # this is for the main stats from the Solver class
        self.__specialized_solver_time += self.__matrix_transform_time

        # Control the matrix is fine
        self.check_common_attributes()
        self.check_matrix()

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):

        # numeric and symbolic UMFPACK objects are being taken care by parent class
        # self.csc_mat will be deleted with this object if it was created internally


        PyMem_Free(self.rval)
        PyMem_Free(self.ival)
