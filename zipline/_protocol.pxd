cdef class BarData:
    cdef object data_portal
    cdef object simulator
    cdef object data_frequency
    cdef dict _views

cdef class SidView:
    cdef object asset
    cdef object data_portal
    cdef object simulator
    cdef object data_frequency
