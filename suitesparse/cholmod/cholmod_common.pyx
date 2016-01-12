

cdef extern from "cholmod.h":

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


CHOLMOD_SYS_DICT = {
        'CHOLMOD_A'     : CHOLMOD_A,
        'CHOLMOD_LDLt'  : CHOLMOD_LDLt,
        'CHOLMOD_LD'    : CHOLMOD_LD,
        'CHOLMOD_DLt'  	: CHOLMOD_DLt,
        'CHOLMOD_L'    	: CHOLMOD_L,
        'CHOLMOD_Lt'   	: CHOLMOD_Lt,
        'CHOLMOD_D'    	: CHOLMOD_D,
        'CHOLMOD_P'    	: CHOLMOD_P,
        'CHOLMOD_Pt'   	: CHOLMOD_Pt
    }


def CHOLMOD_version():
    version_string = "CHOLMOD version %s" % CHOLMOD_VERSION

    return version_string

def CHOLMOD_detailed_version():
    version_string = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                         CHOLMOD_SUB_VERSION,
                                         CHOLMOD_SUBSUB_VERSION,
                                         CHOLMOD_DATE)
    return version_string