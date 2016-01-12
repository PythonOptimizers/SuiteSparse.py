"""
This module generates a factory method to construct SPQR solvers.

It is also the main and only entry for SPQR import by a Python user.

"""

from cysparse.sparse.ll_mat import *


from suitesparse.spqr.spqr_common import *


def SPQRSolver(A, verbose=False):
    """
    Factory method that dispatchs the right type of SPQR solver.

    Args:
        A: Matrix so solve.
    """


    # Use optimized code for CySparse sparse matrices
    if PySparseMatrix_Check(A):
        itype = A.itype
        dtype = A.dtype


    
        if itype == INT32_T:
    
        
            if dtype == FLOAT64_T:
        
                from suitesparse.spqr.cysparse_solver.spqr_cysparse_solver_INT32_t_FLOAT64_t import SPQRCysparseSolver_INT32_t_FLOAT64_t
                return SPQRCysparseSolver_INT32_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                from suitesparse.spqr.cysparse_solver.spqr_cysparse_solver_INT32_t_COMPLEX128_t import SPQRCysparseSolver_INT32_t_COMPLEX128_t
                return SPQRCysparseSolver_INT32_t_COMPLEX128_t(A, verbose=verbose)
    
    


    
        elif itype == INT64_T:
    
        
            if dtype == FLOAT64_T:
        
                from suitesparse.spqr.cysparse_solver.spqr_cysparse_solver_INT64_t_FLOAT64_t import SPQRCysparseSolver_INT64_t_FLOAT64_t
                return SPQRCysparseSolver_INT64_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                from suitesparse.spqr.cysparse_solver.spqr_cysparse_solver_INT64_t_COMPLEX128_t import SPQRCysparseSolver_INT64_t_COMPLEX128_t
                return SPQRCysparseSolver_INT64_t_COMPLEX128_t(A, verbose=verbose)
    
    


        raise TypeError('CySparse matrix has an element type that is incompatible with SPQR')


    raise NotImplementedError('This matrix type is not recognized/implemented...')