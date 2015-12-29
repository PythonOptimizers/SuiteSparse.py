"""
This module generates a factory method to construct UMFPACK solvers.


"""

from cysparse.sparse.ll_mat import *


    
from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT32_t_FLOAT64_t import UmfpackCysparseSolver_INT32_t_FLOAT64_t
    
from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT32_t_COMPLEX128_t import UmfpackCysparseSolver_INT32_t_COMPLEX128_t
    

    
from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT64_t_FLOAT64_t import UmfpackCysparseSolver_INT64_t_FLOAT64_t
    
from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT64_t_COMPLEX128_t import UmfpackCysparseSolver_INT64_t_COMPLEX128_t
    



def UmfpackSolver(A, verbose=False):
    """
    Factory method that dispatchs the right type of UMFPACK solver.

    Args:
        A: Matrix so solve.
    """


    # Use optimized code for CySparse sparse matrices
    if PyLLSparseMatrix_Check(A):
        itype = A.itype
        dtype = A.dtype


    
        if itype == INT32_T:
    
        
            if dtype == FLOAT64_T:
        
                return UmfpackCysparseSolver_INT32_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                return UmfpackCysparseSolver_INT32_t_COMPLEX128_t(A, verbose=verbose)
    
    

    
        elif itype == INT64_T:
    
        
            if dtype == FLOAT64_T:
        
                return UmfpackCysparseSolver_INT64_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                return UmfpackCysparseSolver_INT64_t_COMPLEX128_t(A, verbose=verbose)
    
    




    raise NotImplementedError('This matrix type is not recognized/implemented...')