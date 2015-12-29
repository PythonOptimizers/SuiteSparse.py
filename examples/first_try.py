from suitesparse.umfpack.umfpack_solver import UmfpackSolver
from cysparse.sparse.ll_mat import *

A = LLSparseMatrix(ncol=3, nrow=4)

s = UmfpackSolver(A)

print s.solver_name

b = 3

x = s * b

print "call factorize for the first time"
s.factorize()

print "=" * 80

print "recall factorize without effect..."
s.factorize()

print "=" * 80
print "recall factorize with force_factorize..."
s.factorize(force_factorize=True)

print "=" * 80
print "recall factorize with force_analyze..."
s.factorize(force_analyze=True)

print "=" * 80
print "recall factorize with force_analyze and force_factorize..."
s.factorize(force_analyze=True, force_factorize=True)
