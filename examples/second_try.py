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



