from suitesparse.umfpack.umfpack_solver import *
from cysparse.sparse.ll_mat import *

A = LLSparseMatrix(ncol=4, nrow=4)

import numpy as np

b = np.ones(4)

s = UmfpackSolver(A)



print s.solver_name
print s.solver_version



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

print "@" * 80
print s.stats()

print "/" * 80

print umfpack_version()
print umfpack_detailed_version()



