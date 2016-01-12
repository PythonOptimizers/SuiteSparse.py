"""
This module generates a factory method to construct SPQR solvers.

It is also the main and only entry for SPQR import by a Python user.

"""
{% if use_cysparse %}
from cysparse.sparse.ll_mat import *
{% endif %}

from suitesparse.spqr.spqr_common import *


def SPQRSolver(A, verbose=False):
    """
    Factory method that dispatchs the right type of SPQR solver.

    Args:
        A: Matrix so solve.
    """

{% if use_cysparse %}
    # Use optimized code for CySparse sparse matrices
    if PySparseMatrix_Check(A):
        itype = A.itype
        dtype = A.dtype

{% for index_type in cholmod_index_list %}
    {% if index_type == cholmod_index_list |first %}
        if itype == @index_type|type2enum@:
    {% for element_type in cholmod_type_list %}
        {% if element_type == cholmod_type_list |first %}
            if dtype == @element_type|type2enum@:
        {% else %}
            elif dtype == @element_type|type2enum@:
        {% endif %}
                from suitesparse.spqr.cysparse_solver.spqr_cysparse_solver_@index_type@_@element_type@ import SPQRCysparseSolver_@index_type@_@element_type@
                return SPQRCysparseSolver_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% else %}
        elif itype == @index_type|type2enum@:
    {% for element_type in cholmod_type_list %}
        {% if element_type == cholmod_type_list |first %}
            if dtype == @element_type|type2enum@:
        {% else %}
            elif dtype == @element_type|type2enum@:
        {% endif %}
                from suitesparse.spqr.cysparse_solver.spqr_cysparse_solver_@index_type@_@element_type@ import SPQRCysparseSolver_@index_type@_@element_type@
                return SPQRCysparseSolver_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% endif %}

{% endfor %}
        raise TypeError('CySparse matrix has an element type that is incompatible with SPQR')
{% endif %}

    raise NotImplementedError('This matrix type is not recognized/implemented...')