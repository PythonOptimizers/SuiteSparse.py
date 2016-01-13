from collections import OrderedDict

cdef extern from "SuiteSparseQR_definitions.h":
    # ordering options
    cdef enum:
        SPQR_ORDERING_FIXED = 0
        SPQR_ORDERING_NATURAL = 1
        SPQR_ORDERING_COLAMD = 2
        SPQR_ORDERING_GIVEN = 3       # only used for C/C++ interface
        SPQR_ORDERING_CHOLMOD = 4     # CHOLMOD best-effort (COLAMD, METIS,...)
        SPQR_ORDERING_AMD = 5         # AMD(A'*A)
        SPQR_ORDERING_METIS = 6       # metis(A'*A)
        SPQR_ORDERING_DEFAULT = 7     # SuiteSparseQR default ordering
        SPQR_ORDERING_BEST = 8        # try COLAMD, AMD, and METIS; pick best
        SPQR_ORDERING_BESTAMD = 9     # try COLAMD and AMD; pick best

    # tol options
    cdef enum:
        SPQR_DEFAULT_TOL = -2       # if tol <= -2, the default tol is used
        SPQR_NO_TOL = -1            # if -2 < tol < 0, then no tol is used

    # for qmult, method can be 0,1,2,3:
    cdef enum:
        SPQR_QTX = 0
        SPQR_QX  = 1
        SPQR_XQT = 2
        SPQR_XQ  = 3

    # system can be 0,1,2,3:  Given Q*R=A*E from SuiteSparseQR_factorize:
    cdef enum:
        SPQR_RX_EQUALS_B =  0       # solve R*X=B      or X = R\B
        SPQR_RETX_EQUALS_B = 1      # solve R*E'*X=B   or X = E*(R\B)
        SPQR_RTX_EQUALS_B = 2       # solve R'*X=B     or X = R'\B
        SPQR_RTX_EQUALS_ETB = 3     # solve R'*X=E'*B  or X = R'\(E'*B)

    #VERSION
    char * SPQR_DATE

    cdef enum:
        SPQR_MAIN_VERSION
        SPQR_SUB_VERSION
        SPQR_SUBSUB_VERSION
        SPQR_VERSION

ORDERING_METHOD_DICT = OrderedDict()
ORDERING_METHOD_DICT['SPQR_ORDERING_FIXED'] = SPQR_ORDERING_FIXED
ORDERING_METHOD_DICT['SPQR_ORDERING_NATURAL'] = SPQR_ORDERING_NATURAL
ORDERING_METHOD_DICT['SPQR_ORDERING_COLAMD'] = SPQR_ORDERING_COLAMD
ORDERING_METHOD_DICT['SPQR_ORDERING_GIVEN'] = SPQR_ORDERING_GIVEN
ORDERING_METHOD_DICT['SPQR_ORDERING_CHOLMOD'] = SPQR_ORDERING_CHOLMOD
ORDERING_METHOD_DICT['SPQR_ORDERING_AMD'] = SPQR_ORDERING_AMD
ORDERING_METHOD_DICT['SPQR_ORDERING_METIS'] = SPQR_ORDERING_METIS
ORDERING_METHOD_DICT['SPQR_ORDERING_DEFAULT'] = SPQR_ORDERING_DEFAULT
ORDERING_METHOD_DICT['SPQR_ORDERING_BEST'] = SPQR_ORDERING_BEST
ORDERING_METHOD_DICT['SPQR_ORDERING_BESTAMD'] = SPQR_ORDERING_BESTAMD

ORDERING_METHOD_LIST = ORDERING_METHOD_DICT.keys()

SPQR_SYS_DICT = {
        'SPQR_RX_EQUALS_B'     : SPQR_RX_EQUALS_B,
        'SPQR_RETX_EQUALS_B'   : SPQR_RETX_EQUALS_B,
        'SPQR_RTX_EQUALS_B'    : SPQR_RTX_EQUALS_B,
        'SPQR_RTX_EQUALS_ETB'  : SPQR_RTX_EQUALS_ETB
    }


def spqr_version():
    version_string = "SPQR version %s" % SPQR_VERSION

    return version_string

def spqr_detailed_version():
    version_string = "%s.%s.%s (%s)" % (SPQR_MAIN_VERSION,
                                         SPQR_SUB_VERSION,
                                         SPQR_SUBSUB_VERSION,
                                         SPQR_DATE)
    return version_string