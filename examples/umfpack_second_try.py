from suitesparse.umfpack.umfpack_solver import *
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

s = UmfpackSolver(A, verbose=True)

print "construction time for solver A : %f" % (time.clock() - start_time)


start_time = time.clock()
b = np.ones(size, dtype=np_dtype)
print "construction time for np vector : %f" % (time.clock() - start_time)


start_time = time.clock()
sol = s.solve(b)
print "solve time  : %f" % (time.clock() - start_time)

print sol

print A * sol

print s.solver_name
print s.solver_version

print umfpack_version()
print umfpack_detailed_version()

print s.get_lunz()

print "&" * 80

print s.get_lunz()

(L, U, P, Q, D, do_recip, R) = s.get_LU()

print s.stats()

print s.report_info()

#import sys
#sys.exit(0)

print L
print U
print P
print Q
print D
print do_recip
print R

print "=" * 80

print L
print L.to_ndarray()

print U
print U.to_ndarray()

lhs = L * U

print "test L*U"
print lhs
print np.dot(L.to_ndarray(), U.to_ndarray())

#sys.exit(-1)

P_mat = PermutationLLSparseMatrix(P=P, size=size, dtype=dtype, itype=itype)


print P_mat

Q_mat = PermutationLLSparseMatrix(P=Q, size=size, dtype=dtype, itype=itype)

print Q_mat

print "z" * 80
print R
print R.dtype

if do_recip:
    #R_mat = NewBandLLSparseMatrix(diag_coeff=[0], numpy_arrays=[R], size=3, dtype=dtype, itype=itype)
    R_mat = LLSparseMatrix(size=size, dtype=dtype, itype=itype)
    for i in xrange(size):
        R_mat[i, i] = R[i]
else:
    R_mat = LLSparseMatrix(size=size, dtype=dtype, itype=itype)
    for i in xrange(size):
        R_mat[i, i] = 1/R[i]

print R_mat

print "T" * 80
print "lhs = "
print lhs
rhs = P_mat * R_mat * A * Q_mat
print "rhs = "
print rhs





