"""
This module generates a factory method to construct UMFPACK solvers.

It is also the main and only entry for UMFPACK import by a Python user.

"""
from cysparse.sparse.ll_mat import *

from suitesparse.umfpack.umfpack_common import *

def UmfpackSolver(A, verbose=False):
    """
    Factory method that dispatchs the right type of UMFPACK solver.

    Args:
        A: Matrix so solve.
    """

{% if use_cysparse %}
    # Use optimized code for CySparse sparse matrices
    if PySparseMatrix_Check(A):
        itype = A.itype
        dtype = A.dtype

{% for index_type in umfpack_index_list %}
    {% if index_type == umfpack_index_list |first %}
        if itype == @index_type|type2enum@:
    {% for element_type in umfpack_type_list %}
        {% if element_type == umfpack_type_list |first %}
            if dtype == @element_type|type2enum@:
        {% else %}
            elif dtype == @element_type|type2enum@:
        {% endif %}
                from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_@index_type@_@element_type@ import UmfpackCysparseSolver_@index_type@_@element_type@
                return UmfpackCysparseSolver_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% else %}
        elif itype == @index_type|type2enum@:
    {% for element_type in umfpack_type_list %}
        {% if element_type == umfpack_type_list |first %}
            if dtype == @element_type|type2enum@:
        {% else %}
            elif dtype == @element_type|type2enum@:
        {% endif %}
                from suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_@index_type@_@element_type@ import UmfpackCysparseSolver_@index_type@_@element_type@
                return UmfpackCysparseSolver_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% endif %}

{% endfor %}
        raise TypeError('CySparse matrix has an element type that is incompatible with UMFPACK')
{% endif %}

    raise NotImplementedError('This matrix type is not recognized/implemented...')