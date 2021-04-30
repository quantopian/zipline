cimport numpy as np

from zipline.assets._assets cimport Asset


cdef class InnerPosition:
    """The real values of a position.

    This exists to be owned by both a
    :class:`zipline.finance.position.Position` and a
    :class:`zipline.protocol.Position` at the same time without a cycle.
    """
    cdef readonly Asset asset
    cdef public np.int64_t amount
    cdef public np.float64_t cost_basis
    cdef public np.float64_t last_sale_price
    cdef public object last_sale_date
