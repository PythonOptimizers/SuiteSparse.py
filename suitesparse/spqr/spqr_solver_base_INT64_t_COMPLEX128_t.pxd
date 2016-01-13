from suitesparse.solver_INT64_t_COMPLEX128_t cimport Solver_INT64_t_COMPLEX128_t

from suitesparse.common_types.suitesparse_types cimport *

from  suitesparse.spqr.spqr_common cimport *


# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t

cdef extern from "cholmod.h":
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
        pass

cdef extern from  "SuiteSparseQR_C.h":

    ####################################################################################################################
    # EXPERT MODE
    ####################################################################################################################
    # A real or complex QR factorization, computed by SuiteSparseQR_C_factorize
    ctypedef struct SuiteSparseQR_C_factorization:
        int xtype                  # CHOLMOD_REAL or CHOLMOD_COMPLEX
        void *factors              # from SuiteSparseQR_factorize <double> or SuiteSparseQR_factorize <Complex>

cdef class SPQRSolverBase_INT64_t_COMPLEX128_t(Solver_INT64_t_COMPLEX128_t):
    cdef:

        cholmod_common * common_struct
        cholmod_sparse * sparse_struct

        SuiteSparseQR_C_factorization * factor_struct

        # expert mode
        bint need_to_use_factor_explicitely

    cpdef bint check_common(self)
    cpdef bint check_matrix(self)
    cpdef bint check_factor(self)

    cdef _SPQR_istat(self)