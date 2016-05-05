"""
bool specialization of AdjustedArrayWindow
"""
from numpy cimport uint8_t

ctypedef uint8_t[:, :] databuffer

include "_windowtemplate.pxi"
