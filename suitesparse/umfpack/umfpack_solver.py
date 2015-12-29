"""
This module generates a factory method to construct UMFPACK solvers.


"""


def UmfpackSolver(A):
    """
    Factory method that dispatchs the right type of UMFPACK solver.

    Args:
        A: Matrix so solve.
    """
    pass


    # Use optimized code for CySparse sparse matrices
    if True:
        print "hello you"

