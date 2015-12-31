from __future__ import print_function

from suitesparse.solver_INT32_t_FLOAT64_t cimport Solver_INT32_t_FLOAT64_t

from suitesparse.common_types.suitesparse_types cimport *

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
    # DI VERSION:  real double precision, int integers
    ####################################################################################################################
    int umfpack_di_symbolic(int n_row, int n_col,
                            int * Ap, int * Ai, double * Ax,
                            void ** symbolic,
                            double * control, double * info)

    int umfpack_di_numeric(int * Ap, int * Ai, double * Ax,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_di_free_symbolic(void ** symbolic)
    void umfpack_di_free_numeric(void ** numeric)
    void umfpack_di_defaults(double * control)

    int umfpack_di_solve(int umfpack_sys, int * Ap, int * Ai, double * Ax, double * x, double * b, void * numeric, double * control, double * info)

    int umfpack_di_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, void * numeric)

    int umfpack_di_get_numeric(int * Lp, int * Lj, double * Lx,
                               int * Up, int * Ui, double * Ux,
                               int * P, int * Q, double * Dx,
                               int * do_recip, double * Rs,
                               void * numeric)

    void umfpack_di_report_control(double *)
    void umfpack_di_report_info(double *, double *)
    void umfpack_di_report_symbolic(void *, double *)
    void umfpack_di_report_numeric(void *, double *)













cdef class UmfpackSolverBase_INT32_t_FLOAT64_t(Solver_INT32_t_FLOAT64_t):
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
        self.__solver_version = UmfpackSolverBase_INT32_t_FLOAT64_t.UMFPACK_VERSION

        # this should be adpated in the child classes
        # by default, this solver owns all memory
        self.own_ind_memory = True
        self.own_row_memory = True

        self.own_val_memory = True




    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """
        self.free()

        if self.own_ind_memory:
            PyMem_Free(self.ind)
            if self.__verbose:
                print("Internal ind array cleaned")
        if self.own_row_memory:
            PyMem_Free(self.row)
            if self.__verbose:
                print("Internal row array cleaned")


        if self.own_val_memory:
            PyMem_Free(self.val)
            if self.__verbose:
                print("Internal val array cleaned")



    def free_symbolic(self):
        """
        Free symbolic object if needed.

        """
        if self.__analyzed:
            umfpack_di_free_symbolic(&self.symbolic)
            if self.__verbose:
                print("Symbolic object freed")

    def free_numeric(self):
        """
        Free numeric object if needed.

        """
        if self.__factorized:
            umfpack_di_free_numeric(&self.numeric)
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

    ####################################################################################################################
    # Callbacks
    ####################################################################################################################
    def _analyze(self, *args, **kwargs):
        """
        Create the symbolic object.

        """

        if self.__analyzed:
            self.free_symbolic()

        #cdef INT32_t * ind = <INT32_t *> self.ind
        #cdef INT32_t * row = <INT32_t *> self.row


        #cdef FLOAT64_t * val = <FLOAT64_t *> self.val

        cdef int status


        status= umfpack_di_symbolic(self.nrow, self.ncol, self.ind, self.row, self.val, &self.symbolic, self.control, self.info)


        return status

    def _factorize(self, *args, **kwargs):
        raise NotImplementedError()

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