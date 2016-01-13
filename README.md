# SuiteSparse.py

Cython/Python interface to [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html).

You can use any `Python/Cython` libraries that implement sparse matrices but `SuiteSparse.py` is really well 
integrated with [CySparse](https://github.com/PythonOptimizers/cysparse).

[SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) is a **huge** project written mainly in `C` and `C++`. We try to adhere 
as much as we can to the original libraries and provide both the common and the advanced usages of each sub libraries. `Python` not being `C` nor `C++`,
some choices had to be made. See the documentation on how to use this `Cython/Python` wrapper. 

Our whish: if you know `Python` and know
the `SuiteSparse` library, you'll very quickly be able to use `SuiteSparse.py` to its fullest.

## Announcements

Working on `UMFPACK`, `CHOLMOD` and `SPQR`. This **only** works for `CySparse` sparse matrices for the moment.

Nikolaj

## Dependencies

You need to install `SuiteSparse`. Pay attention to the fact that you **need** to install the dynamic libraries, **not** the static ones.

Althought `SuiteSparse.py` has been optimized to use sparse matrices from [CySparse](https://github.com/PythonOptimizers/cysparse), 
you don't have to install `CySparse`! It's strongly recommended but not mandatory.

- NumPy

## Installation

### Python version

### Cython version

## Run tests

## Run performance tests

## What is implemented?

- [x] UMFPACK Versions 5.7.1 (Oct 10, 2014)
- [x] CHOLMOD Versions 3.0.3 (Oct 23, 2014)
- [x] SPQR Versions 2.0.0 (Oct 1, 2014)
- [ ] KLU
- [ ] BTF
- [ ] AMD
- [ ] CAMD
- [ ] COLAMD
- [ ] CCOLAMD

## Release history

- Version 0.1.0 released on Jan 10, 2016
  
  Added tests (48) for basic UMFPACK get_LU().
  Introduced CHOLMOD.
  
- Version 0.0.1 released on Dec 31, 2015

  Added basic UMFPACK for CySparse LLSparseMatrix matrices.