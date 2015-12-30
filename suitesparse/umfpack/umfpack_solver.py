"""
This module generates a factory method to construct UMFPACK solvers.

It is also the main and only entry for UMFPACK import by a Python user.

"""


def UmfpackSolver(A, verbose=False):
    """
    Factory method that dispatchs the right type of UMFPACK solver.

    Args:
        A: Matrix so solve.
    """


    # Use optimized code for CySparse sparse matrices
    if PySparseMatrix_Check(A):
        itype = A.itype
        dtype = A.dtype


    
        if itype == INT32_T:
    
        
            if dtype == FLOAT64_T:
        
                from suitesparse.umfpack.umfpack_solver_base_INT32_t_FLOAT64_t import *
                from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT32_t_FLOAT64_t import UmfpackCysparseSolver_INT32_t_FLOAT64_t
                return UmfpackCysparseSolver_INT32_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                from suitesparse.umfpack.umfpack_solver_base_INT32_t_COMPLEX128_t import *
                from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT32_t_COMPLEX128_t import UmfpackCysparseSolver_INT32_t_COMPLEX128_t
                return UmfpackCysparseSolver_INT32_t_COMPLEX128_t(A, verbose=verbose)
    
    


    
        elif itype == INT64_T:
    
        
            if dtype == FLOAT64_T:
        
                from suitesparse.umfpack.umfpack_solver_base_INT64_t_FLOAT64_t import *
                from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT64_t_FLOAT64_t import UmfpackCysparseSolver_INT64_t_FLOAT64_t
                return UmfpackCysparseSolver_INT64_t_FLOAT64_t(A, verbose=verbose)
    
        
            elif dtype == COMPLEX128_T:
        
                from suitesparse.umfpack.umfpack_solver_base_INT64_t_COMPLEX128_t import *
                from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT64_t_COMPLEX128_t import UmfpackCysparseSolver_INT64_t_COMPLEX128_t
                return UmfpackCysparseSolver_INT64_t_COMPLEX128_t(A, verbose=verbose)
    
    


        raise TypeError('CySparse matrix has an element type that is incompatible with UMFPACK')


    raise NotImplementedError('This matrix type is not recognized/implemented...')