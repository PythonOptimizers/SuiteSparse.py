from suitesparse.cholmod.cholmod_solver import *
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

s = CholmodSolver(A, verbose=True)



print cholmod_version()
print cholmod_detailed_version()

print s.check_common()
print s.check_factor()
print s.check_matrix()

s.print_sparse_matrix()
print A


print s.solver_version

print cholmod_version()
print cholmod_detailed_version()

print "=" * 50

s.analyze()
print s.check_factor()
s.print_factor()

s.factorize()
print s.check_factor()
s.print_factor()

print "=" * 80

b = np.ones(size, dtype=np_dtype)

x = s.solve(b)

print x

print A * x

print s.print_common()