"""
datetime specialization of AdjustedArrayWindow
"""
from numpy cimport int64_t

ctypedef int64_t[:, :] databuffer

include "_windowtemplate.pxi"
