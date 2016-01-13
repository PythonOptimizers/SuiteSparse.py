from suitesparse.spqr.spqr_solver import *
from cysparse.sparse.ll_mat import *

import numpy as np

import time

size = 5
nrow = 5
ncol = 5

itype = INT32_T #INT64_T
dtype = FLOAT64_T #COMPLEX128_T

np_dtype = np.float64 #complex128

start_time = time.clock()
A = ArrowheadLLSparseMatrix(ncol=ncol, nrow=nrow, dtype=dtype, itype=itype)

print A

AA = A.to_ndarray()

print "Determinant = %f" % np.linalg.det(AA)


print "construction time for matrix A : %f" % (time.clock() - start_time)

start_time = time.clock()
s = SPQRSolver(A, verbose=True)

print "construction time for solver A : %f" % (time.clock() - start_time)



b = np.ones(nrow, np_dtype)

x = s.solve_default(b)

print x
print A * x

print spqr_version()
print spqr_detailed_version()

