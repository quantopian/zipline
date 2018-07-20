cimport numpy as np

cdef class Asset:

    cdef readonly np.int64_t sid
    cdef readonly object exchange_info

    cdef readonly object symbol
    cdef readonly object asset_name

    cdef readonly object start_date
    cdef readonly object end_date
    cdef public object first_traded
    cdef readonly object auto_close_date

    cdef readonly object tick_size
    cdef readonly float price_multiplier

    cpdef __reduce__(self)
    cpdef to_dict(self)


cdef class Equity(Asset):
    pass


cdef class Future(Asset):
    cdef readonly object root_symbol
    cdef readonly object notice_date
    cdef readonly object expiration_date

    cpdef __reduce__(self)
    cpdef to_dict(self)
