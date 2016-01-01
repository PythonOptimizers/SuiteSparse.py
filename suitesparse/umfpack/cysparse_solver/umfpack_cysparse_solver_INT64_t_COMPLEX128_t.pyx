from __future__ import print_function

from suitesparse.umfpack.umfpack_solver_base_INT64_t_COMPLEX128_t cimport UmfpackSolverBase_INT64_t_COMPLEX128_t
from suitesparse.common_types.suitesparse_types cimport *

from suitesparse.common_types.suitesparse_generic_types cimport split_array_complex_values_kernel_INT64_t_COMPLEX128_t, join_array_complex_values_kernel_INT64_t_COMPLEX128_t

from suitesparse.umfpack.umfpack_common import test_umfpack_result, UMFPACK_SYS_DICT

from cysparse.sparse.s_mat cimport PySparseMatrix_Check, PyLLSparseMatrix_Check, PyCSCSparseMatrix_Check, PyCSRSparseMatrix_Check
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t cimport CSCSparseMatrix_INT64_t_COMPLEX128_t, MakeCSCSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.csr_mat_matrices.csr_mat_INT64_t_COMPLEX128_t cimport CSRSparseMatrix_INT64_t_COMPLEX128_t, MakeCSRSparseMatrix_INT64_t_COMPLEX128_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time


cdef extern from "umfpack.h":

    char * UMFPACK_DATE
    ctypedef long SuiteSparse_long  # This is exactly CySparse's INT64_t

    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO

        UMFPACK_VERSION, UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION

        # Return codes:
        UMFPACK_OK

        UMFPACK_WARNING_singular_matrix, UMFPACK_WARNING_determinant_underflow
        UMFPACK_WARNING_determinant_overflow

        UMFPACK_ERROR_out_of_memory
        UMFPACK_ERROR_invalid_Numeric_object
        UMFPACK_ERROR_invalid_Symbolic_object
        UMFPACK_ERROR_argument_missing
        UMFPACK_ERROR_n_nonpositive
        UMFPACK_ERROR_invalid_matrix
        UMFPACK_ERROR_different_pattern
        UMFPACK_ERROR_invalid_system
        UMFPACK_ERROR_invalid_permutation
        UMFPACK_ERROR_internal_error
        UMFPACK_ERROR_file_IO

        # Control:
        # Printing routines:
        UMFPACK_PRL
        # umfpack_*_symbolic:
        UMFPACK_DENSE_ROW
        UMFPACK_DENSE_COL
        UMFPACK_BLOCK_SIZE
        UMFPACK_STRATEGY
        UMFPACK_2BY2_TOLERANCE
        UMFPACK_FIXQ
        UMFPACK_AMD_DENSE
        UMFPACK_AGGRESSIVE
        # umfpack_*_numeric:
        UMFPACK_PIVOT_TOLERANCE
        UMFPACK_ALLOC_INIT
        UMFPACK_SYM_PIVOT_TOLERANCE
        UMFPACK_SCALE
        UMFPACK_FRONT_ALLOC_INIT
        UMFPACK_DROPTOL
        # umfpack_*_solve:
        UMFPACK_IRSTEP

        # For UMFPACK_STRATEGY:
        UMFPACK_STRATEGY_AUTO
        UMFPACK_STRATEGY_UNSYMMETRIC
        UMFPACK_STRATEGY_2BY2
        UMFPACK_STRATEGY_SYMMETRIC

        # For UMFPACK_SCALE:
        UMFPACK_SCALE_NONE
        UMFPACK_SCALE_SUM
        UMFPACK_SCALE_MAX

        # for SOLVE ACTIONS
        UMFPACK_A
        UMFPACK_At
        UMFPACK_Aat
        UMFPACK_Pt_L
        UMFPACK_L
        UMFPACK_Lt_P
        UMFPACK_Lat_P
        UMFPACK_Lt
        UMFPACK_U_Qt
        UMFPACK_U
        UMFPACK_Q_Ut
        UMFPACK_Q_Uat
        UMFPACK_Ut
        UMFPACK_Uat

    # TODO: Change types for CySparse types? int -> INT32_t, double -> FLOAT64_t etc?
    #       and keep only **one** declaration?












    ####################################################################################################################
    # ZL VERSION:   complex double precision, SuiteSparse long integers
    ####################################################################################################################
    SuiteSparse_long umfpack_zl_symbolic(SuiteSparse_long n_row, SuiteSparse_long n_col,
                            SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax, double * Az,
                            void ** symbolic,
                            double * control, double * info)

    SuiteSparse_long umfpack_zl_numeric(SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax, double * Az,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_zl_free_symbolic(void ** symbolic)
    void umfpack_zl_free_numeric(void ** numeric)
    void umfpack_zl_defaults(double * control)

    SuiteSparse_long umfpack_zl_solve(SuiteSparse_long umfpack_sys, SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax,  double * Az, double * Xx, double * Xz, double * bx, double * bz, void * numeric, double * control, double * info)

    SuiteSparse_long umfpack_zl_get_lunz(SuiteSparse_long * lnz, SuiteSparse_long * unz, SuiteSparse_long * n_row, SuiteSparse_long * n_col,
                            SuiteSparse_long * nz_udiag, void * numeric)

    SuiteSparse_long umfpack_zl_get_numeric(SuiteSparse_long * Lp, SuiteSparse_long * Lj,
                               double * Lx, double * Lz,
                               SuiteSparse_long * Up, SuiteSparse_long * Ui,
                               double * Ux, double * Uz,
                               SuiteSparse_long * P, SuiteSparse_long * Q,
                               double * Dx, double * Dz,
                               SuiteSparse_long * do_recip, double * Rs,
                               void * numeric)

    void umfpack_zl_report_control(double *)
    void umfpack_zl_report_info(double *, double *)
    void umfpack_zl_report_symbolic(void *, double *)
    void umfpack_zl_report_numeric(void *, double *)




cdef class UmfpackCysparseSolver_INT64_t_COMPLEX128_t(UmfpackSolverBase_INT64_t_COMPLEX128_t):
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
        self.check_matrix()

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):

        # numeric and symbolic UMFPACK objects are being taken care by parent class
        # self.csc_mat will be deleted with this object if it was created internally


        PyMem_Free(self.rval)
        PyMem_Free(self.ival)


    ####################################################################################################################
    # LU ROUTINES
    ####################################################################################################################
    def get_LU(self, get_L=True, get_U=True, get_P=True, get_Q=True, get_D=True, get_R=True):
        """
        Return LU factorisation objects. If needed, the LU factorisation is triggered.

        Returns:
            (L, U, P, Q, D, do_recip, R)

            The original matrix A is factorized into

                L U = P R A Q

            where:
             - L is unit lower triangular,
             - U is upper triangular,
             - P and Q are permutation matrices,
             - R is a row-scaling diagonal matrix such that

                  * the i-th row of A has been multiplied by R[i] if do_recip = True,
                  * the i-th row of A has been divided by R[i] if do_recip = False.

            L and U are returned as CSRSparseMatrix and CSCSparseMatrix sparse matrices respectively.
            P, Q and R are returned as NumPy arrays.

        """
        # TODO: use properties?? we can only get matrices, not set them...
        # TODO: implement the use of L=True, U=True, P=True, Q=True, D=True, R=True
        # i.e. allow to return only parts of the arguments and not necessarily all of them...
        self.factorize()

        cdef:
            INT64_t lnz
            INT64_t unz
            INT64_t n_row
            INT64_t n_col
            INT64_t nz_udiag

            INT64_t _do_recip

        (lnz, unz, n_row, n_col, nz_udiag) = self.get_lunz()

        # L CSR matrix
        cdef INT64_t * Lp = <INT64_t *> PyMem_Malloc((n_row + 1) * sizeof(INT64_t))
        if not Lp:
            raise MemoryError()

        cdef INT64_t * Lj = <INT64_t *> PyMem_Malloc(lnz * sizeof(INT64_t))
        if not Lj:
            PyMem_Free(Lp)
            raise MemoryError()


        cdef FLOAT64_t * Lx = <FLOAT64_t *> PyMem_Malloc(lnz * sizeof(FLOAT64_t))
        if not Lx:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            raise MemoryError()

        cdef FLOAT64_t * Lz = <FLOAT64_t *> PyMem_Malloc(lnz * sizeof(FLOAT64_t))
        if not Lz:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)

            raise MemoryError()


        # U CSC matrix
        cdef INT64_t * Up = <INT64_t *> PyMem_Malloc((n_col + 1) * sizeof(INT64_t))
        if not Up:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)

            PyMem_Free(Lz)

            raise MemoryError()

        cdef INT64_t * Ui = <INT64_t *> PyMem_Malloc(unz * sizeof(INT64_t))
        if not Ui:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)

            PyMem_Free(Lz)

            PyMem_Free(Up)

            raise MemoryError()


        cdef FLOAT64_t * Ux = <FLOAT64_t *> PyMem_Malloc(unz * sizeof(FLOAT64_t))
        if not Ux:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)
            PyMem_Free(Lz)

            PyMem_Free(Ui)

            raise MemoryError()

        cdef FLOAT64_t * Uz = <FLOAT64_t *> PyMem_Malloc(unz * sizeof(FLOAT64_t))
        if not Uz:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)
            PyMem_Free(Lz)

            PyMem_Free(Ui)
            PyMem_Free(Ux)

            raise MemoryError()


        # TODO: see what type of int exactly to pass
        cdef cnp.npy_intp *dims_n_row = [n_row]
        cdef cnp.npy_intp *dims_n_col = [n_col]

        cdef cnp.npy_intp *dims_min = [min(n_row, n_col)]

        cdef INT64_t dim_D = min(n_row, n_col)

        #cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] P
        #cdef cnp.ndarray[int, ndim=1, mode='c'] P
        cdef cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] P


        #P = cnp.PyArray_EMPTY(1, dims_n_row, cnp.NPY_INT32, 0)
        P = cnp.PyArray_EMPTY(1, dims_n_row, cnp.NPY_INT64, 0)


        #cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] Q
        #cdef cnp.ndarray[int, ndim=1, mode='c'] Q
        cdef cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] Q

        #Q = cnp.PyArray_EMPTY(1, dims_n_col, cnp.NPY_INT32, 0)
        Q = cnp.PyArray_EMPTY(1, dims_n_col, cnp.NPY_INT64, 0)


        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] D
        D = cnp.PyArray_EMPTY(1, dims_min, cnp.NPY_DOUBLE, 0)


        # create Dx and Dz
        cdef:
            FLOAT64_t * Dx
            FLOAT64_t * Dz

        Dx = <FLOAT64_t *> PyMem_Malloc(dim_D * sizeof(FLOAT64_t))
        if not Dx:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)
            PyMem_Free(Lz)

            PyMem_Free(Ui)
            PyMem_Free(Ux)

            raise MemoryError()

        Dz = <FLOAT64_t *> PyMem_Malloc(dim_D * sizeof(FLOAT64_t))

        if not Dz:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)
            PyMem_Free(Lz)

            PyMem_Free(Ui)
            PyMem_Free(Ux)

            PyMem_Free(Dx)
            raise MemoryError()


        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] R
        R = cnp.PyArray_EMPTY(1, dims_n_row, cnp.NPY_DOUBLE, 0)


        cdef int status =umfpack_zl_get_numeric(Lp, Lj, Lx, Lz,
                               Up, Ui, Ux,Uz,
                               <INT64_t *> cnp.PyArray_DATA(P), <INT64_t *> cnp.PyArray_DATA(Q), Dx, Dz,
                               &_do_recip, <double *> cnp.PyArray_DATA(R),
                               self.numeric)



        if status != UMFPACK_OK:
            test_umfpack_result(status, "get_LU()")

        cdef bint do_recip = _do_recip

        cdef CSRSparseMatrix_INT64_t_COMPLEX128_t L
        cdef CSCSparseMatrix_INT64_t_COMPLEX128_t U

        cdef INT64_t size = min(n_row,n_col)


        cdef COMPLEX128_t * Lx_complex = <COMPLEX128_t *> PyMem_Malloc(lnz * sizeof(COMPLEX128_t))
        if not Lx_complex:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)
            PyMem_Free(Lz)

            PyMem_Free(Ui)
            PyMem_Free(Ux)

            PyMem_Free(Dx)
            PyMem_Free(Dz)

            raise MemoryError()

        join_array_complex_values_kernel_INT64_t_COMPLEX128_t(Lx, lnz,
                                                        Lz, lnz,
                                                        Lx_complex, lnz)

        cdef COMPLEX128_t * Ux_complex = <COMPLEX128_t *> PyMem_Malloc(unz * sizeof(COMPLEX128_t))
        if not Ux_complex:
            PyMem_Free(Lp)
            PyMem_Free(Lj)

            PyMem_Free(Lx)
            PyMem_Free(Lz)

            PyMem_Free(Ui)
            PyMem_Free(Ux)

            PyMem_Free(Dx)
            PyMem_Free(Dz)

            PyMem_Free(Lx_complex)

            raise MemoryError()

        join_array_complex_values_kernel_INT64_t_COMPLEX128_t(Ux, unz,
                                                        Uz, unz,
                                                        Ux_complex, unz)

        L = MakeCSRSparseMatrix_INT64_t_COMPLEX128_t(nrow=size, ncol=size, nnz=lnz, ind=Lp, col=Lj, val=Lx_complex, store_symmetric=False, store_zero=False)
        U = MakeCSCSparseMatrix_INT64_t_COMPLEX128_t(nrow=size, ncol=size, nnz=unz, ind=Up, row=Ui, val=Ux_complex, store_symmetric=False, store_zero=False)

        return (L, U, P, Q, D, do_recip, R)

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