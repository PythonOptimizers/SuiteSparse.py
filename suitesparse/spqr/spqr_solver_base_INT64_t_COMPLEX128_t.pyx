from suitesparse.spqr.spqr_common import SPQR_SYS_DICT, spqr_version, spqr_detailed_version, ORDERING_METHOD_LIST, ORDERING_METHOD_DICT

from suitesparse.cholmod.cholmod_solver_base_INT64_t_COMPLEX128_t cimport numpy_ndarray_to_cholmod_dense

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef extern from "cholmod.h":

    cdef:
        char * CHOLMOD_DATE

    cdef enum:
        CHOLMOD_MAIN_VERSION
        CHOLMOD_SUB_VERSION
        CHOLMOD_SUBSUB_VERSION
        CHOLMOD_VERSION


    cdef enum:
        # Five objects
        CHOLMOD_COMMON
        CHOLMOD_SPARSE
        CHOLMOD_FACTOR
        CHOLMOD_DENSE
        CHOLMOD_TRIPLET

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
    cdef enum:
        CHOLMOD_DOUBLE      # all numerical values are double
        CHOLMOD_SINGLE

    cdef enum:
        CHOLMOD_A    		# solve Ax=b
        CHOLMOD_LDLt        # solve LDL'x=b
        CHOLMOD_LD          # solve LDx=b
        CHOLMOD_DLt  	    # solve DL'x=b
        CHOLMOD_L    	    # solve Lx=b
        CHOLMOD_Lt   	    # solve L'x=b
        CHOLMOD_D    	    # solve Dx=b
        CHOLMOD_P    	    # permute x=Px
        CHOLMOD_Pt   	    # permute x=P'x
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


cdef extern from  "SuiteSparseQR_C.h":
    # returns rank(A) estimate, (-1) if failure
    cdef SuiteSparse_long SuiteSparseQR_C(
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        SuiteSparse_long econ,      # e = max(min(m,econ),rank(A))
        int getCTX,                 # 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k)
        cholmod_sparse *A,          # m-by-n sparse matrix to factorize
        cholmod_sparse *Bsparse,    # sparse m-by-k B
        cholmod_dense  *Bdense,     # dense  m-by-k B
        # outputs:
        cholmod_sparse **Zsparse,   # sparse Z
        cholmod_dense  **Zdense,    # dense Z
        cholmod_sparse **R,         # e-by-n sparse matrix
        SuiteSparse_long **E,       # size n column perm, NULL if identity
        cholmod_sparse **H,         # m-by-nh Householder vectors
        SuiteSparse_long **HPinv,   # size m row permutation
        cholmod_dense **HTau,       # 1-by-nh Householder coefficients
        cholmod_common *cc          # workspace and parameters
        )


    # [Q,R,E] = qr(A), returning Q as a sparse matrix
    # returns rank(A) est., (-1) if failure
    cdef SuiteSparse_long SuiteSparseQR_C_QR (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        SuiteSparse_long econ,      # e = max(min(m,econ),rank(A))
        cholmod_sparse *A,          # m-by-n sparse matrix to factorize
        # outputs:
        cholmod_sparse **Q,         # m-by-e sparse matrix
        cholmod_sparse **R,         # e-by-n sparse matrix
        SuiteSparse_long **E,       # size n column perm, NULL if identity
        cholmod_common *cc          # workspace and parameters
        )

    # X = A\B where B is dense
    # returns X, NULL if failure
    cdef cholmod_dense *SuiteSparseQR_C_backslash (
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_dense  *B,          # m-by-k
        cholmod_common *cc          # workspace and parameters
    )

    # X = A\B where B is dense, using default ordering and tol
    # returns X, NULL if failure
    cdef cholmod_dense *SuiteSparseQR_C_backslash_default (
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_dense  *B,          # m-by-k
        cholmod_common *cc          # workspace and parameters
    )

    # TODO: implement this version?
    # X = A\B where B is sparse
    # returns X, or NULL
    cdef cholmod_sparse *SuiteSparseQR_C_backslash_sparse (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_sparse *B,          # m-by-k
        cholmod_common *cc          # workspace and parameters
    )

    ####################################################################################################################
    # EXPERT MODE
    ####################################################################################################################
    cdef SuiteSparseQR_C_factorization *SuiteSparseQR_C_factorize (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_common *cc          # workspace and parameters
    )

    cdef SuiteSparseQR_C_factorization *SuiteSparseQR_C_symbolic (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        int allow_tol,              # if TRUE allow tol for rank detection
        cholmod_sparse *A,          # m-by-n sparse matrix, A->x ignored
        cholmod_common *cc          # workspace and parameters
    )

    cdef int SuiteSparseQR_C_numeric (
        # inputs:
        double tol,                 # treat columns with 2-norm <= tol as zero
        cholmod_sparse *A,          # sparse matrix to factorize
        # input/output:
        SuiteSparseQR_C_factorization *QR,
        cholmod_common *cc          # workspace and parameters
    )

    # Free the QR factors computed by SuiteSparseQR_C_factorize
    # returns TRUE (1) if OK, FALSE (0) otherwise
    cdef int SuiteSparseQR_C_free (
        SuiteSparseQR_C_factorization **QR,
        cholmod_common *cc          # workspace and parameters
    )

    # returnx X, or NULL if failure
    cdef cholmod_dense* SuiteSparseQR_C_solve (
        int system,                 # which system to solve
        SuiteSparseQR_C_factorization *QR,  # of an m-by-n sparse matrix A
        cholmod_dense *B,           # right-hand-side, m-by-k or n-by-k
        cholmod_common *cc          # workspace and parameters
    )

    # Applies Q in Householder form (as stored in the QR factorization object
    # returned by SuiteSparseQR_C_factorize) to a dense matrix X.
    #
    # method SPQR_QTX (0): Y = Q'*X
    # method SPQR_QX  (1): Y = Q*X
    # method SPQR_XQT (2): Y = X*Q'
    # method SPQR_XQ  (3): Y = X*Q

    # returns Y, or NULL on failure
    cdef cholmod_dense *SuiteSparseQR_C_qmult (
        # inputs:
        int method,                 # 0,1,2,3
        SuiteSparseQR_C_factorization *QR,  # of an m-by-n sparse matrix A
        cholmod_dense *X,           # size m-by-n with leading dimension ldx
        cholmod_common *cc          # workspace and parameters
    )



cdef class SPQRSolverBase_INT64_t_COMPLEX128_t(Solver_INT64_t_COMPLEX128_t):

    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, A, **kwargs):
        self.__solver_name = 'SPQR'
        self.__solver_version = spqr_detailed_version()

        if self.__verbose:
            self.set_verbosity(3)
        else:
            self.set_verbosity(0)

        # CHOLMOD
        self.common_struct = <cholmod_common *> malloc(sizeof(cholmod_common))

        # TODO test if malloc succeeded

        cholmod_l_start(self.common_struct)

        # All internal memory allocation is done by the specialized Solvers!!!
        # Specialized solvers are also responsible to deallocate this memory!!!
        # This is an internal hack for efficiency when possible
        self.sparse_struct = <cholmod_sparse *> malloc(sizeof(cholmod_sparse))

        # TODO test if malloc succeeded

        # by default, we expect the user to use the expert mode
        # only certain solve_XXX() methods cancel this
        self.need_to_use_factor_explicitely = True

        # set default parameters for control
        self.reset_default_parameters()

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """


        if self.__analyzed:
            SuiteSparseQR_C_free(&self.factor_struct, self.common_struct)

        cholmod_l_finish(self.common_struct)

    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################
    def reset_default_parameters(self):
        cholmod_l_defaults(self.common_struct)

    ####################################################################################################################
    # CALLBACKS
    ####################################################################################################################
    def _analyze(self, *args, **kwargs):
        if self.need_to_use_factor_explicitely:
            pass

    def _factorize(self, *args, **kwargs):
        if self.need_to_use_factor_explicitely:
            pass

    def _solve(self, cnp.ndarray[cnp.npy_complex128, mode="c"] b, str ordering='SPQR_ORDERING_BEST', double drop_tol=0):
        """
        Solve `A*x = b` with `b` dense (case `X = A\B`).

        Args:
            ordering (str):
            drop_tol (double): Treat columns with 2-norm <= drop_tol as zero.

        """
        # no expert mode: SPQR is responsible for the factor object
        self.need_to_use_factor_explicitely = False

        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        # TODO: change this (see TODO below)
        if b.ndim != 1:
            raise NotImplementedError('Matrices for right member will be implemented soon...')

        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        # TODO: does it use multidimension (matrix and not vector)
        # Currently ONLY support vectors...
        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol

        cholmod_sol = SuiteSparseQR_C_backslash(ORDERING_METHOD_DICT[ordering], drop_tol, self.sparse_struct, &B, self.common_struct)

        # test solution
        if cholmod_sol == NULL:
            # no solution was found
            raise RuntimeError('No solution found')

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT64_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_l_free_dense(&cholmod_sol, self.common_struct)

        return sol

    #############################################################
    # SPECIALIZED ROUTINES
    #############################################################
    def solve_default(self, cnp.ndarray[cnp.npy_complex128, mode="c"] b):
        """
        Solve `A*x = b` with `b` dense (case `X = A\B`).

        Args:
            b

        """
        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        # TODO: change this (see TODO below)
        if b.ndim != 1:
            raise NotImplementedError('Matrices for right member will be implemented soon...')

        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        # TODO: does it use multidimension (matrix and not vector)
        # Currently ONLY support vectors...
        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol

        cholmod_sol = SuiteSparseQR_C_backslash_default(self.sparse_struct, &B, self.common_struct)

        # test solution
        if cholmod_sol == NULL:
            # no solution was found
            raise RuntimeError('No solution found')

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT64_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_l_free_dense(&cholmod_sol, self.common_struct)

        return sol

    #############################################################
    # CHECKING ROUTINES
    #############################################################
    cpdef bint check_common(self):
        return cholmod_l_check_common(self.common_struct)

    cpdef bint check_factor(self):
        pass


    cpdef bint check_matrix(self):
        """
        Check if internal CSC matrix is OK.

        Returns:
            ``True`` if everything is OK, ``False`` otherwise. Depending on the verbosity, some error messages can
            be displayed on ``sys.stdout``.
        """
        return cholmod_l_check_sparse(self.sparse_struct, self.common_struct), "Internal CSC matrix ill formatted"

    #############################################################
    # CHOLMOD PRINTING ROUTINES
    #############################################################
    def set_verbosity(self, verbosity_level):
        # TODO: change this!
        pass

    ####################################################################################################################
    # STATISTICS
    ####################################################################################################################
    def SPQR_orderning(self):
        """
        Returns the chosen ordering.
        """
        return ORDERING_METHOD_LIST[self.common_struct.SPQR_istat[7]]

    def SPQR_drop_tol_used(self):
        """
        Return `drop_tol` (`double`). columns with 2-norm <= tol treated as 0
        """
        return self.common_struct.SPQR_tol_used

    cdef _SPQR_istat(self):
        """
        Main statistic method for SPQR, :program:`Cython` version.
        """
        s = ''

        # ordering
        s += 'ORDERING USED: %s' % self.SPQR_orderning()

        return s

    def spqr_statistics(self):
        """
        Main statistic for SPQR.

        """
        # TODO: todo
        return self. _SPQR_istat()