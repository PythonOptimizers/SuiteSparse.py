from suitesparse.umfpack.umfpack_solver import UmfpackSolver
from cysparse.sparse.ll_mat import *

A = LLSparseMatrix(ncol=3, nrow=4)

s = UmfpackSolver(A)

print s.solver_name
