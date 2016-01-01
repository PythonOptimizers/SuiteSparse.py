

cpdef enum CySparseType:
    INT32_T = 0
    UINT32_T = 1
    INT64_T = 2
    UINT64_T = 3
    FLOAT32_T = 4
    FLOAT64_T = 5
    FLOAT128_T = 6
    COMPLEX64_T = 7
    COMPLEX128_T = 8
    COMPLEX256_T = 9

ctypedef int INT32_t
ctypedef unsigned int UINT32_t
ctypedef long INT64_t
ctypedef unsigned long UINT64_t

ctypedef float FLOAT32_t
ctypedef double FLOAT64_t
ctypedef long double FLOAT128_t

ctypedef float complex COMPLEX64_t
ctypedef double complex COMPLEX128_t
ctypedef long double complex COMPLEX256_t