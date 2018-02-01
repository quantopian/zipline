cdef class Asset:

    cdef readonly int sid
    # Cached hash of self.sid
    cdef int sid_hash

    cdef readonly object symbol
    cdef readonly object asset_name

    cdef readonly object start_date
    cdef readonly object end_date
    cdef public object first_traded
    cdef readonly object auto_close_date

    cdef readonly object exchange
    cdef readonly object exchange_full

    cpdef __reduce__(self)
    cpdef to_dict(self)


cdef class Equity(Asset):
    pass


cdef class Future(Asset):
    cdef readonly object root_symbol
    cdef readonly object notice_date
    cdef readonly object expiration_date
    cdef readonly object tick_size
    cdef readonly float multiplier

    cpdef __reduce__(self)
    cpdef to_dict(self)
