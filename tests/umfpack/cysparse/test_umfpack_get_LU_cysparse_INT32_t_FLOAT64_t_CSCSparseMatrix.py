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
class CySparseUmfpackget_LUNoSymmetryNoZero_CSCSparseMatrix_INT32_t_FLOAT64_t_TestCase(unittest.TestCase):
    def setUp(self):
       

        self.A = LinearFillLLSparseMatrix(size=SIZE, dtype=FLOAT64_T, itype=INT32_T)


        self.C = self.A.to_csc()



        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):

                self.assertTrue(lhs[i, j] - rhs[i, j] < EPS, "lhs[%d, %d] =? %f , rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseUmfpackget_LUWithSymmetryNoZero_CSCSparseMatrix_INT32_t_FLOAT64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT64_T, itype=INT32_T, store_symmetry=True)


        self.C = self.A.to_csc()



        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):

                self.assertTrue(abs(lhs[i, j] - rhs[i, j]) < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseUmfpackget_LUNoSymmetrySWithZero_CSCSparseMatrix_INT32_t_FLOAT64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.A = LinearFillLLSparseMatrix(size=SIZE, dtype=FLOAT64_T, itype=INT32_T, store_zero=True)


        self.C = self.A.to_csc()



        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):


                self.assertTrue(abs(lhs[i, j] - rhs[i, j]) < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseUmfpackget_LUWithSymmetrySWithZero_CSCSparseMatrix_INT32_t_FLOAT64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT64_T, itype=INT32_T, store_symmetry=True, store_zero=True)


        self.C = self.A.to_csc()



        self.solver = UmfpackSolver(self.C)

    def test_factorization_element_by_element(self):
        """
        Verify the equality ``L * U = P * R * A * Q``
        """
        (L, U, P, Q, D, do_recip, R) = self.solver.get_LU()

        lhs = L * U

        P_mat = PermutationLLSparseMatrix(P=P, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
        Q_mat = PermutationLLSparseMatrix(P=Q, size=SIZE, dtype=FLOAT64_T, itype=INT32_T)

        R_mat = None
        if do_recip:
            R_mat = LLSparseMatrix(size=SIZE, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = R[i]
        else:
            R_mat = LLSparseMatrix(size=size, dtype=FLOAT64_T, itype=INT32_T)
            for i in xrange(SIZE):
                R_mat[i, i] = 1/R[i]

        rhs = P_mat * R_mat * self.C * Q_mat

        for i in xrange(SIZE):
            for j in xrange(SIZE):


                self.assertTrue(abs(lhs[i, j] - rhs[i, j]) < EPS, "lhs[%d, %d] =? %f, rhs[%d, %d] = %f" % (i,j, lhs[i, j], i, j, rhs[i, j]))


if __name__ == '__main__':
    unittest.main()
