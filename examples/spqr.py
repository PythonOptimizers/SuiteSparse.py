from suitesparse.spqr.spqr_solver import *
from cysparse.sparse.ll_mat import *

import numpy as np

import time

size = 5
itype = INT32_T #INT64_T
dtype = FLOAT64_T #COMPLEX128_T

np_dtype = np.float64 #complex128

start_time = time.clock()
A = LinearFillLLSparseMatrix(ncol=size, nrow=size, dtype=dtype, itype=itype, store_symmetric=True)

print "construction time for matrix A : %f" % (time.clock() - start_time)

start_time = time.clock()
print "construction time for solver A : %f" % (time.clock() - start_time)

s = SPQRSolver(A, verbose=True)

print spqr_version()
print spqr_detailed_version()

