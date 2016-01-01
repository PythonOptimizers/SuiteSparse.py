#!/usr/bin/env python

"""
This file tests UMFPACK ``get_LU`` for all :program:`CySparse` matrices objects.

We verify the equality ``L * U = P * R * A * Q``.

"""

import unittest
from cysparse.sparse.ll_mat import *

from suitesparse.umfpack.umfpack_solver import UmfpackSolver


########################################################################################################################
# Tests
########################################################################################################################


SIZE = 10
EPS = 1e-12

#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseUmfpackget_LUNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
       

        self.A = LinearFillLLSparseMatrix(size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):
                self.assertTrue(lhs[i, j] - rhs[i, j] < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))

#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseUmfpackget_LUWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):
                self.assertTrue(lhs[i, j] - rhs[i, j] < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseUmfpackget_LUNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.A = LinearFillLLSparseMatrix(size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):
                self.assertTrue(lhs[i, j] - rhs[i, j] < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseUmfpackget_LUWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=@type|type2enum@, itype=@index|type2enum@)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):
                self.assertTrue(lhs[i, j] - rhs[i, j] < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


if __name__ == '__main__':
    unittest.main()

