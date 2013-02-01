**********
Extensions
**********

.. highlight:: cython

Philosophy
==========

Use judgement when to use C extensions. Debugging them potentially
has a time cost of t=infinity, they can segfault, and may not be
debugabble by anyone else. Simply put 90% of the time its not worth it
and construction of extensions be must informed by scientific profiling.
Listen to your inner Knuth.

Writing C Extensions
====================

Caveats aside, C Extensions can be in two forms:

- C
- Cython

Cython is a superset of Python which compiles into C. The code it
produces is generally not human readable.

Reference: http://docs.cython.org/

C is well, C. You manage your own memory and interface with Python.h .
If you need raw performance or need to interface with other C libraries
this is often the best approach. Of course this requires that you
be very careful to tend to memory and and Python's internal garbage
collection.

Reference: http://docs.python.org/c-api/

One can write C++ extensions, but please don't.

One could also embed Assembly in C and thus in Python, but again please
don't.

Compilers
=========

Compatibility

    - Do not use Clang
    - Do not use GCC-LLVM

Use standard GCC >= 4.6 from gnu.org, otherwise extensions will have
undefined behavior and will not be portable.

Also make sure to code against Python 2.7 and numpy 1.6.1 header
files. If using Cython have it auto figure out the paths to ensurable
portability.

Pure C
======

.. highlight:: c

::

    #include "Python.h"

Releasing the GIL
=================

::

    from libc.stdio cimport printf

    with nogil:
        # in here you allowed to do whatever you like so long as
        # you do not touch Python objects. This really should
        # only be used to interface with other C libraries.

        printf("hello, world\n");

Debugging
=========

Compile with debug symbols and use gdb and valgrind. It sucks but its
really the only way.

Vim
===

.. highlight:: vim

For syntax highlighting in Vim::

    :set syntax=pyrex
