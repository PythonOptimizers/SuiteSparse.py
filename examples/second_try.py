from suitesparse.umfpack.umfpack_solver import *
from cysparse.sparse.ll_mat import *

import numpy as np

A = ArrowheadLLSparseMatrix(ncol=4, nrow=4, dtype=COMPLEX128_T)

s = UmfpackSolver(A, verbose=True)

b = np.ones(4, dtype=np.complex128)

sol = s.solve(b)

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

import sys
sys.exit(0)

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

P_mat = NewPermutationLLSparseMatrix(P=P, size=3, dtype=dtype, itype=itype)


print P_mat

Q_mat = NewPermutationLLSparseMatrix(P=Q, size=3, dtype=dtype, itype=itype)

print Q_mat

print "z" * 80
print R
print R.dtype

if do_recip:
    #R_mat = NewBandLLSparseMatrix(diag_coeff=[0], numpy_arrays=[R], size=3, dtype=dtype, itype=itype)
    R_mat = NewLLSparseMatrix(size=3, dtype=dtype, itype=itype)
    for i in xrange(3):
        R_mat[i, i] = R[i]
else:
    R_mat = NewLLSparseMatrix(size=3, dtype=dtype, itype=itype)
    for i in xrange(3):
        R_mat[i, i] = 1/R[i]

print R_mat

print "T" * 80
print "lhs = "
print lhs
rhs = P_mat * R_mat * A * Q_mat
print "rhs = "
print rhs





