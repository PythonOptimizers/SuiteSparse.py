..  _base_solver:

=========================================================
Base solver
=========================================================

All solver interfaces available on `PythonOptimizers <https://github.com/PythonOptimizers>`_ use a common *base solver* interface. This allow you to easily switch from one solver to another
without modifying your code. We describe this common interface here.

The big three
==============

``analyze()``
-------------

``factorize()``
----------------

``solve()``
------------

Statistics
============


Syntactic sugar
===============

``LU * A`` and ``LU(b)``

