"""
float specialization of AdjustedArrayWindow
"""
from numpy cimport float64_t as ctype
cdef str dtype = 'float64'

include "_windowtemplate.pxi"
