"""
float specialization of AdjustedArrayWindow
"""
from numpy cimport float64_t
ctypedef float64_t[:, :] databuffer

include "_windowtemplate.pxi"
