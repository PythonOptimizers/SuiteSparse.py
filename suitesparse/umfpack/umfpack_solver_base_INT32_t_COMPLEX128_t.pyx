from __future__ import print_function

from suitesparse.solver_INT32_t_COMPLEX128_t cimport Solver_INT32_t_COMPLEX128_t

from suitesparse.common_types.suitesparse_types cimport *
from suitesparse.umfpack.umfpack_common import test_umfpack_result, UMFPACK_SYS_DICT

from suitesparse.common_types.suitesparse_generic_types cimport split_array_complex_values_kernel_INT32_t_COMPLEX128_t, join_array_complex_values_kernel_INT32_t_COMPLEX128_t

from suitesparse.utils.stdout_redirect import stdout_redirected

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import numpy as np
cimport numpy as cnp

cnp.import_array()

from cStringIO import StringIO
import sys


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
    # ZI VERSION:   complex double precision, int integers
    ####################################################################################################################
    int umfpack_zi_symbolic(int n_row, int n_col,
                            int * Ap, int * Ai, double * Ax, double * Az,
                            void ** symbolic,
                            double * control, double * info)

    int umfpack_zi_numeric(int * Ap, int * Ai, double * Ax, double * Az,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_zi_free_symbolic(void ** symbolic)
    void umfpack_zi_free_numeric(void ** numeric)
    void umfpack_zi_defaults(double * control)

    int umfpack_zi_solve(int umfpack_sys, int * Ap, int * Ai, double * Ax,  double * Az,
                         double * Xx, double * Xz, double * bx, double * bz, void * numeric, double * control, double * info)

    int umfpack_zi_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, void * numeric)

    int umfpack_zi_get_numeric(int * Lp, int * Lj,
                               double * Lx, double * Lz,
                               int * Up, int * Ui,
                               double * Ux, double * Uz,
                               int * P, int * Q,
                               double * Dx, double * Dz,
                               int * do_recip, double * Rs,
                               void * numeric)

    void umfpack_zi_report_control(double *)
    void umfpack_zi_report_info(double *, double *)
    void umfpack_zi_report_symbolic(void *, double *)
    void umfpack_zi_report_numeric(void *, double *)









cdef class UmfpackSolverBase_INT32_t_COMPLEX128_t(Solver_INT32_t_COMPLEX128_t):
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
        self.__solver_version = UmfpackSolverBase_INT32_t_COMPLEX128_t.UMFPACK_VERSION

        if self.__verbose:
            self.set_verbosity(3)
        else:
            self.set_verbosity(0)


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
            umfpack_zi_free_symbolic(&self.symbolic)
            if self.__verbose:
                print("Symbolic object freed")

    def free_numeric(self):
        """
        Free numeric object if needed.

        """
        if self.__factorized:
            umfpack_zi_free_numeric(&self.numeric)
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


        status= umfpack_zi_symbolic(self.nrow, self.ncol, self.ind, self.row, self.rval, self.ival, &self.symbolic, self.control, self.info)


        if status != UMFPACK_OK:
            self.free_symbolic()
            test_umfpack_result(status, "analyze()", print_on_screen=self.__verbose)


    def _factorize(self, *args, **kwargs):

        if self.__factorized:
            self.free_numeric()

        cdef int status


        status = umfpack_zi_numeric(self.ind, self.row,
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
            TypeError: When vector ``b`` is not a :program:`NumPy` vector.
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

        #cdef INT32_t * ind = <INT32_t *> self.ind
        #cdef INT32_t * row = <INT32_t *> self.row


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

        split_array_complex_values_kernel_INT32_t_COMPLEX128_t(b_data, dim_b,
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



        cdef int status =  umfpack_zi_solve(UMFPACK_SYS_DICT[umfpack_sys], self.ind, self.row, self.rval, self.ival, solx, solz, bx, bz, self.numeric, self.control, self.info)

        if status != UMFPACK_OK:
            test_umfpack_result(status, "solve()")


        # join solx and solz
        cdef COMPLEX128_t * sol_data = <COMPLEX128_t *> cnp.PyArray_DATA(sol)

        join_array_complex_values_kernel_INT32_t_COMPLEX128_t(solx, self.ncol,
                                                        solz, self.ncol,
                                                        sol_data, self.ncol)

        # Free temp arrays
        PyMem_Free(bx)
        PyMem_Free(bz)
        PyMem_Free(solx)
        PyMem_Free(solz)


        return sol

    ####################################################################################################################
    # LU ROUTINES
    ####################################################################################################################
    def get_lunz(self):
        """
        Determine the size and number of non zeros in the LU factors held by the opaque ``Numeric`` object.

        Returns:
            (lnz, unz, n_row, n_col, nz_udiag):

            lnz: The number of nonzeros in ``L``, including the diagonal (which is all one's)
            unz: The number of nonzeros in ``U``, including the diagonal.
            n_row, n_col: The order of the ``L`` and ``U`` matrices. ``L`` is ``n_row`` -by- ``min(n_row,n_col)``
                and ``U`` is ``min(n_row,n_col)`` -by- ``n_col``.
            nz_udiag: The number of numerically nonzero values on the diagonal of ``U``. The
                matrix is singular if ``nz_diag < min(n_row,n_col)``. A ``divide-by-zero``
                will occur if ``nz_diag < n_row == n_col`` when solving a sparse system
                involving the matrix ``U`` in ``solve()``.

        Raises:
            RuntimeError: When ``UMFPACK`` return status is not ``UMFPACK_OK`` and is an error.



        """
        self.factorize()

        cdef:
            INT32_t lnz
            INT32_t unz
            INT32_t n_row
            INT32_t n_col
            INT32_t nz_udiag

        cdef status = umfpack_zi_get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag, self.numeric)

        if status != UMFPACK_OK:
            test_umfpack_result(status, "get_lunz()")

        return (lnz, unz, n_row, n_col, nz_udiag)

    def get_LU(self, get_L=True, get_U=True, get_P=True, get_Q=True, get_D=True, get_R=True):
        # This is really dependent on the matrices library we use
        raise NotImplementedError()

    ####################################################################################################################
    # REPORTING ROUTINES
    ####################################################################################################################
    def set_verbosity(self, level):
        """
        Set UMFPACK verbosity level.

        Args:
            level (int): Verbosity level (default: 1).

        """
        self.control[UMFPACK_PRL] = level

    def get_verbosity(self):
        """
        Return UMFPACK verbosity level.

        Returns:
            verbosity_level (int): The verbosity level set.

        """
        return self.control[UMFPACK_PRL]

    def get_report_control(self):
        """
        Return control values into string.

        """

        umfpack_zi_report_control(self.control)


    def report_info(self):
        """
        Print all status information.

        Use **after** calling :meth:`create_symbolic()`, :meth:`create_numeric()`, :meth:`factorize()` or :meth:`solve()`.

        """
        umfpack_zi_report_info(self.control, self.info)


    def report_symbolic(self):
        """
        Print information about the opaque ``symbolic`` object.

        """
        if not self.symbolic_computed:
            print("No opaque symbolic object has been computed")
            return

        umfpack_zi_report_symbolic(self.symbolic, self.control)


    def report_numeric(self):
        """
        Print information about the opaque ``numeric`` object.

        """
        umfpack_zi_report_numeric(self.numeric, self.control)


    ####################################################################################################################
    # Statistics Callbacks
    ####################################################################################################################
    def _stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        lines = []
        lines.append("Matrix library:")
        lines.append("===============")
        lines.append(self._specialized_stats(*args, **kwargs))

        lines.append("Report control:")
        lines.append("---------------")
        #lines.append(self.get_report_control())
        lines.append("")

        lines.append("Report info:")
        lines.append("---------------")
        #lines.append(self.report_info())
        lines.append("")

        lines.append("Report symbolic object:")
        lines.append("-----------------------")
        #lines.append(self.report_symbolic())
        lines.append("")

        lines.append("Report numeric object:")
        lines.append("-----------------------")
        #lines.append(self.report_numeric())
        lines.append("")

        return '\n'.join(lines)

    def _specialized_stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        raise NotImplementedError("You have to add some specialized stats for every type of supported matrices")