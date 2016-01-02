
from suitesparse.cholmod.cholmod_common import CHOLMOD_SYS_DICT

cdef extern from "cholmod.h":
    cdef:
        char * CHOLMOD_DATE
    #ctypedef long SuiteSparse_long # doesn't work... why?
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

    # Factor struct
    int cholmod_l_check_factor(cholmod_factor *L, cholmod_common *Common)
    int cholmod_l_print_factor(cholmod_factor *L, const char *name, cholmod_common *Common)
    #int cholmod_l_free_factor()
    # factor_to_sparse

    # Memory management
    void * cholmod_l_free(size_t n, size_t size,	void *p,  cholmod_common *Common)



cdef class CholmodSolverBase_INT64_t_COMPLEX128_t(Solver_INT64_t_COMPLEX128_t):
    """

    We follow the common use of Umfpack. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Umfpack.
    """
    CHOLMOD_VERSION = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                     CHOLMOD_SUB_VERSION,
                                     CHOLMOD_SUBSUB_VERSION,
                                     CHOLMOD_DATE)

    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, A, **kwargs):
        self.__solver_name = 'UMFPACK'
        self.__solver_version = CholmodSolverBase_INT64_t_COMPLEX128_t.CHOLMOD_VERSION

        if self.__verbose:
            self.set_verbosity(3)
        else:
            self.set_verbosity(0)

        # set default parameters for control
        self.reset_default_parameters()

    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################
    def reset_default_parameters(self):
        cholmod_l_defaults(&self.common_struct)

    cdef check_matrix(self):
        pass

    cdef check_factor(self):
        pass

    def set_verbosity(self, verbosity_level):
        pass