
from suitesparse.cholmod.cholmod_common import CHOLMOD_SYS_DICT, CHOLMOD_version, CHOLMOD_detailed_version

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp

cnp.import_array()

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

    int cholmod_start(cholmod_common *Common)
    int cholmod_finish(cholmod_common *Common)

    int cholmod_defaults(cholmod_common *Common)

    # Common struct
    int cholmod_check_common(cholmod_common *Common)
    int cholmod_print_common(const char *name, cholmod_common *Common)

    # Sparse struct
    int cholmod_check_sparse(cholmod_sparse *A, cholmod_common *Common)
    int cholmod_print_sparse(cholmod_sparse *A, const char *name, cholmod_common *Common)
    int cholmod_free_sparse(cholmod_sparse **A, cholmod_common *Common)

    # Dense struct
    int cholmod_free_dense(cholmod_dense **X, cholmod_common *Common)

    # Factor struct
    int cholmod_check_factor(cholmod_factor *L, cholmod_common *Common)
    int cholmod_print_factor(cholmod_factor *L, const char *name, cholmod_common *Common)
    int cholmod_free_factor(cholmod_factor *L, cholmod_common *Common)
    # factor_to_sparse

    # Memory management
    void * cholmod_free(size_t n, size_t size,	void *p,  cholmod_common *Common)

    # ANALYZE
    cholmod_factor * cholmod_analyze(cholmod_sparse *A,cholmod_common *Common)

    # FACTORIZE
    int cholmod_factorize(cholmod_sparse *, cholmod_factor *, cholmod_common *)

    # SOLVE
    cholmod_dense * cholmod_solve (int, cholmod_factor *, cholmod_dense *, cholmod_common *)
    cholmod_sparse * cholmod_spsolve (int, cholmod_factor *, cholmod_sparse *,
    cholmod_common *)

########################################################################################################################
# CHOLMOD HELPERS
########################################################################################################################
##################################################################
# FROM NumPy ndarray -> cholmod_dense
##################################################################
cdef cholmod_dense numpy_ndarray_to_cholmod_dense(cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] b):
    """
    Convert a :program:`NumPy` one dimensionnal array to the corresponding ``cholmod_dense`` matrix.
    """
    # access b
    cdef COMPLEX128_t * b_data = <COMPLEX128_t *> cnp.PyArray_DATA(b)

    # Creation of CHOLMOD DENSE MATRIX
    cdef cholmod_dense B
    B = cholmod_dense()

    B.nrow = b.shape[0]
    B.ncol = 1

    B.nzmax = b.shape[0]

    B.d = b.shape[0]


    # TODO: to be done!
    raise NotImplementedError("Not yet...")

    B.xtype = CHOLMOD_ZOMPLEX                    # CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    B.dtype = CHOLMOD_DOUBLE


    return B

##################################################################
# FROM cholmod_dense -> NumPy ndarray
##################################################################
cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] cholmod_dense_to_numpy_ndarray(cholmod_dense * b):
    raise NotImplementedError()

########################################################################################################################
# Base functionnalities
########################################################################################################################


cdef class CHOLMODSolverBase_INT32_t_COMPLEX128_t(Solver_INT32_t_COMPLEX128_t):
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
        self.__solver_name = 'CHOLMOD'
        self.__solver_version = CHOLMODSolverBase_INT32_t_COMPLEX128_t.CHOLMOD_VERSION

        if self.__verbose:
            self.set_verbosity(3)
        else:
            self.set_verbosity(0)

        # CHOLMOD
        self.common_struct = <cholmod_common *> malloc(sizeof(cholmod_common))

        # TODO test if malloc succeeded

        cholmod_start(self.common_struct)

        # All internal memory allocation is done by the specialized Solvers!!!
        # Specialized solvers are also responsible to deallocate this memory!!!
        # This is an internal hack for efficiency when possible
        self.sparse_struct = <cholmod_sparse *> malloc(sizeof(cholmod_sparse))

        # TODO test if malloc succeeded

        # set default parameters for control
        self.reset_default_parameters()

    ####################################################################################################################
    # Properties
    ####################################################################################################################
    # Propreties that bear the same name as a reserved Python keyword, are prefixed by 'c_'.
    ######################################### COMMON STRUCT Properties #################################################
    # Printing
    property c_print:
        def __get__(self): return self.common_struct.print_
        def __set__(self, value): self.common_struct.print_ = value

    property precise:
        def __get__(self): return self.common_struct.precise
        def __set__(self, value): self.common_struct.precise = value

    property try_catch:
        def __get__(self): return self.common_struct.try_catch
        def __set__(self, value): self.common_struct.try_catch = value

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):


        # TODO: Solve this bug!!! See #16
        #if self.__factorized:
        #    cholmod_free_factor(self.factor_struct, self.common_struct)

        # specialized child Solvers **must** deallocate the internal memory used!!!
        # DO NOT CALL CHOLMOD function to free the sparse matrix
        free(self.sparse_struct)

        cholmod_finish(self.common_struct)
        free(self.common_struct)



    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################
    def reset_default_parameters(self):
        cholmod_defaults(self.common_struct)

    #############################################################
    # CHECKING ROUTINES
    #############################################################
    cpdef bint check_common(self):
        return cholmod_check_common(self.common_struct)

    cpdef bint check_factor(self):
        if self.__analyzed:
            return cholmod_check_factor(self.factor_struct, self.common_struct)

        return False

    def set_verbosity(self, verbosity_level):
        # TODO: change this!
        pass

    cpdef bint check_matrix(self):
        """
        Check if internal CSC matrix is OK.

        Returns:
            ``True`` if everything is OK, ``False`` otherwise. Depending on the verbosity, some error messages can
            be displayed on ``sys.stdout``.
        """
        return cholmod_check_sparse(self.sparse_struct, self.common_struct), "Internal CSC matrix ill formatted"

    #############################################################
    # CHOLMOD PRINTING ROUTINES
    #############################################################
    def print_sparse_matrix(self):
        return cholmod_print_sparse(self.sparse_struct, "Internal CSC CHOLMOD respresentation of sparse matrix", self.common_struct)

    def print_factor(self):
        return cholmod_print_factor(self.factor_struct, "Internal CHOLMOD factor struct", self.common_struct)

    def print_common(self):
        cholmod_print_common("cholmod_common_struct", self.common_struct)

    ####################################################################################################################
    # GPU
    ####################################################################################################################
    def request_GPU(self):
        """
        GPU-acceleration is requested.

        If GPU processing is requested but there is no GPU present, CHOLMOD will continue using the CPU only.
        Consequently it is **always safe** to request GPU processing.

        """
        self.common_struct.useGPU = 1

    def prohibit_GPU(self):
        """
        GPU-acceleration is explicitely prohibited.

        """
        self.common_struct.useGPU = 0

    ####################################################################################################################
    # Callbacks
    ####################################################################################################################
    def _analyze(self, *args, **kwargs):
        self.factor_struct = <cholmod_factor *> cholmod_analyze(self.sparse_struct,self.common_struct)

    def _factorize(self, *args, **kwargs):
        cholmod_factorize(self.sparse_struct, self.factor_struct, self.common_struct)

    def _solve(self, cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] b, cholmod_sys='CHOLMOD_A'):

        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        if cholmod_sys not in CHOLMOD_SYS_DICT.keys():
            raise ValueError("Argument 'cholmod_sys' must be in " % CHOLMOD_SYS_DICT.keys())

        # if needed
        self.factorize()

        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol
        cholmod_sol = cholmod_solve(CHOLMOD_SYS_DICT[cholmod_sys], self.factor_struct, &B, self.common_struct)

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT32_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_free_dense(&cholmod_sol, self.common_struct)

        return sol

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



        return '\n'.join(lines)

    def _specialized_stats(self, *args, **kwargs):
        """
        Returns a string with specialized statistics about the factorization.
        """
        raise NotImplementedError("You have to add some specialized stats for every type of supported matrices")