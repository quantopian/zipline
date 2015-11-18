"""
datetime specialization of AdjustedArrayWindow
"""
from numpy cimport int64_t as ctype
cdef str dtype = 'datetime64[ns]'

include "_windowtemplate.pxi"
