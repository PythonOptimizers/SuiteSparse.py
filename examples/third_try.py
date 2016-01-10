from suitesparse.cholmod.cholmod_solver import *
from cysparse.sparse.ll_mat import *

import numpy as np

import time

size = 5
itype = INT64_T
dtype = COMPLEX128_T

np_dtype = np.complex128

start_time = time.clock()
A = ArrowheadLLSparseMatrix(ncol=size, nrow=size, dtype=dtype, itype=itype)

print "construction time for matrix A : %f" % (time.clock() - start_time)

start_time = time.clock()
print "construction time for solver A : %f" % (time.clock() - start_time)

s = CholmodSolver(A, verbose=True)



print s.CHOLMOD_VERSION

print cholmod_version()
print cholmod_detailed_version()

