"""
bool specialization of AdjustedArrayWindow
"""
from numpy cimport uint8_t as ctype
cdef str dtype = 'bool'

include "_windowtemplate.pxi"
