"""
Functions used to test metaprogramming tools' interactions with Cython.
"""
cimport cython

from zipline.utils.preprocess import preprocess


@cython.embedsignature(True)
def function_notypes(a, b, c):
    return (a, b, c)

@cython.embedsignature(True)
def function_partial_types(int a, b, str c):
    return (a, b, c)


@cython.embedsignature(True)
def function_all_types(int a, int b, str c):
    return (a, b, c)


@cython.embedsignature(False)
def function_no_docstring(a, b, c):
    return (a, b, c)


@cython.embedsignature(False)
def function_non_signature_docstring(a, b, c):
    "Random docstring"
    return (a, b, c)


@cython.embedsignature(True)
def function_with_defaults(int a, b=3, int c=4):
    pass


def add1(x):
    return x + 1


def sub1(x):
    return x - 1


cdef class PreprocessedCythonClass:
    """
    This class verifies that we can apply preprocessors to methods of a Cython
    class if they're marked with embedsignature.
    """

    @preprocess(a=add1, b=sub1)
    @cython.embedsignature(True)
    def method_notypes(self, a, b, c):
        return (a, b, c)

    @preprocess(a=add1, b=sub1)
    @cython.embedsignature(True)
    def method_partial_types(self, int a, b, str c):
        return (a, b, c)

    @preprocess(a=add1, b=sub1)
    @cython.embedsignature(True)
    def method_all_types(self, int a, int b, str c):
        return (a, b, c)

    @cython.embedsignature(False)
    def method_no_signature(self, a, b, c):
        return (a, b, c)
