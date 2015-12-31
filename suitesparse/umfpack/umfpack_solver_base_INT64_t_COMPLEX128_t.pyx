from __future__ import print_function

from suitesparse.solver_INT64_t_COMPLEX128_t cimport Solver_INT64_t_COMPLEX128_t

from suitesparse.common_types.suitesparse_types cimport *
from suitesparse.umfpack.umfpack_common import test_umfpack_result, UMFPACK_SYS_DICT

from suitesparse.common_types.suitesparse_generic_types cimport split_array_complex_values_kernel_INT64_t_COMPLEX128_t, join_array_complex_values_kernel_INT64_t_COMPLEX128_t


from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import numpy as np
cimport numpy as cnp

cnp.import_array()


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





cdef class UmfpackSolverBase_INT64_t_COMPLEX128_t(Solver_INT64_t_COMPLEX128_t):
    """

    We follow the common use of Umfpack. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Umfpack.
    """
    UMFPACK_VERSION = "%s.%s.%s (%s)" % (UMFPACK_MAIN_VERSION,
                                     UMFPACK_SUB_VERSION,
                                     UMFPACK_SUBSUB_VERSION,
                                     UMFPACK_DATE)

    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, A, **kwargs):
        self.__solver_name = 'UMFPACK'
        self.__solver_version = UmfpackSolverBase_INT64_t_COMPLEX128_t.UMFPACK_VERSION


    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """
        self.free()

    def free_symbolic(self):
        """
        Free symbolic object if needed.

        """
        if self.__analyzed:
            umfpack_zl_free_symbolic(&self.symbolic)
            if self.__verbose:
                print("Symbolic object freed")

    def free_numeric(self):
        """
        Free numeric object if needed.

        """
        if self.__factorized:
            umfpack_zl_free_numeric(&self.numeric)
            if self.__verbose:
                print("Numeric object freed")

    def free(self):
        """
        Free symbolic and/or numeric objects if needed.

        """
        self.free_numeric()
        self.free_symbolic()

    ####################################################################################################################
    # Helpers
    ####################################################################################################################
    cdef check_matrix(self):
        """
        Call this method in the child class to assert all is well with the matrix.
        """
        # test if we can use UMFPACK
        assert self.nrow == self.ncol, "Only square matrices are handled in UMFPACK"

        # check if internal arrays are empty or not
        assert self.ind != NULL, "Internal ind array is not defined"
        assert self.row != NULL, "Internal row array is not defined"

        assert self.rval != NULL, "Internal rval array is not defined"
        assert self.ival != NULL, "Internal ival array is not defined"


    ####################################################################################################################
    # Callbacks
    ####################################################################################################################
    def _analyze(self, *args, **kwargs):
        """
        Create the symbolic object.

        """

        if self.__analyzed:
            self.free_symbolic()

        cdef int status


        status= umfpack_zl_symbolic(self.nrow, self.ncol, self.ind, self.row, self.rval, self.ival, &self.symbolic, self.control, self.info)


        if status != UMFPACK_OK:
            self.free_symbolic()
            test_umfpack_result(status, "analyze()", print_on_screen=self.__verbose)


    def _factorize(self, *args, **kwargs):

        if self.__factorized:
            self.free_numeric()

        cdef int status


        status = umfpack_zl_numeric(self.ind, self.row,
                   self.rval, self.ival,
                   self.symbolic,
                   &self.numeric,
                   self.control, self.info)



        if status != UMFPACK_OK:
            self.free_numeric()
            test_umfpack_result(status, "factorize()", print_on_screen=self.__verbose)

    def _solve(self, cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] b, umfpack_sys='UMFPACK_A', irsteps=2):
        """
        Solve the linear system  ``A * x = b``.

        Args:
           b: a Numpy vector of appropriate dimension.
           umfpack_sys: specifies the type of system being solved:

                    +-------------------+--------------------------------------+
                    |``"UMFPACK_A"``    | :math:`\mathbf{A} x = b` (default)   |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_At"``   | :math:`\mathbf{A}^T x = b`           |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Pt_L"`` | :math:`\mathbf{P}^T \mathbf{L} x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_L"``    | :math:`\mathbf{L} x = b`             |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Lt_P"`` | :math:`\mathbf{L}^T \mathbf{P} x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Lt"``   | :math:`\mathbf{L}^T x = b`           |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_U_Qt"`` | :math:`\mathbf{U} \mathbf{Q}^T x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_U"``    | :math:`\mathbf{U} x = b`             |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Q_Ut"`` | :math:`\mathbf{Q} \mathbf{U}^T x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Ut"``   | :math:`\mathbf{U}^T x = b`           |
                    +-------------------+--------------------------------------+

           irsteps: number of iterative refinement steps to attempt. Default: 2

        Returns:
            ``sol``: The solution of ``A*x=b`` if everything went well.

        Raises:
            AssertionError: When vector ``b`` is not a :program:`NumPy` vector.
            AttributeError: When vector ``b`` doesn't have a ``shape`` attribute.
            AssertionError: When vector ``b`` doesn't have the right first dimension.
            MemoryError: When there is not enought memory to create solution.
            RuntimeError: Whenever ``UMFPACK`` returned status is not ``UMFPACK_OK`` and is an error.

        Notes:
            The opaque objects ``symbolic`` and ``numeric`` are automatically created if necessary.

            You can ask for a report of what happened by calling :meth:`report_info()`.



        """
        # TODO: add other umfpack_sys arguments to the docstring.
        # TODO: allow other types of b and test better argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")
        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        if umfpack_sys not in UMFPACK_SYS_DICT.keys():
            raise ValueError('umfpack_sys must be in' % UMFPACK_SYS_DICT.keys())

        self.control[UMFPACK_IRSTEP] = irsteps

        self.factorize()

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        #cdef INT64_t * ind = <INT64_t *> self.ind
        #cdef INT64_t * row = <INT64_t *> self.row


        # access b
        cdef COMPLEX128_t * b_data = <COMPLEX128_t *> cnp.PyArray_DATA(b)

        # create bx and bz
        cdef:
            FLOAT64_t * bx
            FLOAT64_t * bz

        bx = <FLOAT64_t *> PyMem_Malloc(dim_b * sizeof(FLOAT64_t))
        if not bx:
            raise MemoryError()

        bz = <FLOAT64_t *> PyMem_Malloc(dim_b * sizeof(FLOAT64_t))

        if not bz:
            PyMem_Free(bx)
            raise MemoryError()

        split_array_complex_values_kernel_INT64_t_COMPLEX128_t(b_data, dim_b,
                                                         bx, dim_b,
                                                         bz, dim_b)

        # create solx and solz
        cdef:
            FLOAT64_t * solx
            FLOAT64_t * solz

        solx = <FLOAT64_t *> PyMem_Malloc(self.ncol * sizeof(FLOAT64_t))
        if not solx:
            PyMem_Free(bx)
            PyMem_Free(bz)

            raise MemoryError()

        solz = <FLOAT64_t *> PyMem_Malloc(self.ncol * sizeof(FLOAT64_t))

        if not solz:
            PyMem_Free(bx)
            PyMem_Free(bz)

            PyMem_Free(solx)
            raise MemoryError()



        cdef int status =  umfpack_zl_solve(UMFPACK_SYS_DICT[umfpack_sys], self.ind, self.row, self.rval, self.ival, solx, solz, bx, bz, self.numeric, self.control, self.info)

        if status != UMFPACK_OK:
            test_umfpack_result(status, "solve()")


        # join solx and solz
        cdef COMPLEX128_t * sol_data = <COMPLEX128_t *> cnp.PyArray_DATA(sol)

        join_array_complex_values_kernel_INT64_t_COMPLEX128_t(solx, self.ncol,
                                                        solz, self.ncol,
                                                        sol_data, self.ncol)

        # Free temp arrays
        PyMem_Free(bx)
        PyMem_Free(bz)
        PyMem_Free(solx)
        PyMem_Free(solz)


        return sol

    ####################################################################################################################
    # Statistics Callbacks
    ####################################################################################################################
    def _stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        return self._specialized_stats(*args, **kwargs)

    def _specialized_stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        raise NotImplementedError("You have to add some specialized stats for every type of supported matrices")