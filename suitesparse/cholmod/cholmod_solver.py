"""
This module generates a factory method to construct CHOLMOD solvers.

It is also the main and only entry for CHOLMOD import by a Python user.

"""

from cysparse.sparse.ll_mat import *


from suitesparse.cholmod.cholmod_common import *


def CholmodSolver(A, verbose=False):
    """
    Factory method that dispatchs the right type of CHOLMOD solver.

    Args:
        A: Matrix so solve.
    """


    # Use optimized code for CySparse sparse matrices
    if PySparseMatrix_Check(A):
        itype = A.itype
        dtype = A.dtype


    
        if itype == INT32_T:
    
        
            if dtype == FLOAT64_T:
        
                from suitesparse.cholmod.cysparse_solver.cholmod_cysparse_solver_INT32_t_FLOAT64_t import CholmodCysparseSolver_INT32_t_FLOAT64_t
                return CholmodCysparseSolver_INT32_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                from suitesparse.cholmod.cysparse_solver.cholmod_cysparse_solver_INT32_t_COMPLEX128_t import CholmodCysparseSolver_INT32_t_COMPLEX128_t
                return CholmodCysparseSolver_INT32_t_COMPLEX128_t(A, verbose=verbose)
    
    


    
        elif itype == INT64_T:
    
        
            if dtype == FLOAT64_T:
        
                from suitesparse.cholmod.cysparse_solver.cholmod_cysparse_solver_INT64_t_FLOAT64_t import CholmodCysparseSolver_INT64_t_FLOAT64_t
                return CholmodCysparseSolver_INT64_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                from suitesparse.cholmod.cysparse_solver.cholmod_cysparse_solver_INT64_t_COMPLEX128_t import CholmodCysparseSolver_INT64_t_COMPLEX128_t
                return CholmodCysparseSolver_INT64_t_COMPLEX128_t(A, verbose=verbose)
    
    


        raise TypeError('CySparse matrix has an element type that is incompatible with UMFPACK')


    raise NotImplementedError('This matrix type is not recognized/implemented...')