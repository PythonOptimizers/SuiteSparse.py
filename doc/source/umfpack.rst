..  _umfpack:

===================================
:program:`UmfPack`
===================================

..  note::

    Readers not familiar with the :program:`UMFPACK` library might consider reading its excellent user manual, in particular
    section 5: *Using UMFPACK in a C program*.
    
We provide an interface to the  :program:`C`-callable :program:`UMFPACK` library. This library has 32 user-callable routines of which XX have been implemented in :program:`SuiteSparse.py`.
All but three of the routines come in four versions in the orginal library, with different sizes of integers and for real or complex
floating-point numbers:

- umfpack di : real double precision, int integers.
- umfpack dl : real double precision, SuiteSparse long integers.
- umfpack zi : complex double precision, int integers.
- umfpack zl : complex double precision, SuiteSparse long integers.

:program:`SuiteSparse.py` takes care transparently of these different types.


 

