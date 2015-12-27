# SuiteSparse.py

Cython/Python interface to [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html).

You can use any `Python/Cython` libraries that implement sparse matrices but `SuiteSparse.py` is really well 
integrated with [CySparse](https://github.com/PythonOptimizers/cysparse).

[SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) is a **huge** project written mainly in `C` and `C++`. We try to adhere 
as much as we can to the original libraries and provide both the common and the advanced usage of each sub libraries. `Python` not being `C` nor `C++`,
some choices had to be made. See the documentation on how to use this `Cython/Python` wrapper. As a rule of thumb: if you know `Python` and know
the `SuiteSparse` library, you'll very quickly be able to use `SuiteSparse.py` to its fullest.

## Announcements

Working on `UMFPACK`. Not urgent but "easy" and a good start!

Nikolaj

## Dependencies

You need to install `SuiteSparse`. Pay attention to the fact that you **need** to install the dynamic libraries, **not** the static ones.

Althought `SuiteSparse.py` has been optimized to use sparse matrices from [CySparse](https://github.com/PythonOptimizers/cysparse), 
you don't have to install `CySparse`! It's strongly recommended but not mandatory.

## Installation

### Python version

### Cython version

## Run tests

## Run performance tests

## What is implemented?

- [ ] UMFPACK
- [ ] CHOLMOD
- [ ] SPQR
- [ ] KLU
- [ ] BTF
- [ ] AMD
- [ ] CAMD
- [ ] COLAMD
- [ ] CCOLAMD

## Release history