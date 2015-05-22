from numpy cimport float64_t
# Purely for readability. There aren't C-level declarations for these types.
ctypedef object Int64Index_t
ctypedef object DatetimeIndex_t
ctypedef object Timestamp_t


cpdef tuple get_adjustment_locs(DatetimeIndex_t dates_index,
                                Int64Index_t assets_index,
                                Timestamp_t start_date,
                                Timestamp_t end_date,
                                int asset_id):
    """
    Compute indices suitable for passing to an Adjustment constructor.

    If the specified dates aren't in dates_index, we return the index of the
    first date **BEFORE** the supplied date.

    Example:

    >>> from pandas import date_range, Int64Index, Timestamp
    >>> dates = date_range('2014-01-01', '2014-01-07')
    >>> assets = Int64Index(range(10))
    >>> get_adjustment_locs(
    ...     dates,
    ...     assets,
    ...     Timestamp('2014-01-03'),
    ...     Timestamp('2014-01-05'),
    ...     3,
    ... )
    (2, 4, 3)
    """
    cdef int start_date_loc
    if start_date is None:
        start_date_loc = 0
    else:
        start_date_loc = dates_index.get_loc(start_date, method='ffill')

    return (
        # start_date is allowed to be None, indicating "everything
        # before the end_date"
        start_date_loc,
        dates_index.get_loc(end_date, method='ffill'),
        assets_index.get_loc(asset_id),  # Must be exact match.
    )


cdef class Float64Adjustment:
    """
    Base class for adjustments that operate on Float64 buffers.
    """
    cdef:
        readonly Py_ssize_t col, first_row, last_row
        readonly float64_t value

    def __cinit__(self,
                  Py_ssize_t first_row,
                  Py_ssize_t last_row,
                  Py_ssize_t col,
                  float64_t value):

        self.first_row = first_row
        self.last_row = last_row
        self.col = col
        self.value = value

    def __repr__(self):
        return "%s(first_row=%d, last_row=%d, col=%d, value=%f)" % (
            type(self).__name__,
            self.first_row,
            self.last_row,
            self.col,
            self.value,
        )


cdef class Float64Multiply(Float64Adjustment):
    """
    An adjustment that multiplies by a scalar.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(9, dtype=float).reshape(3, 3)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])

    >>> adj = Float64Multiply(first_row=1, last_row=2, col=1, value=4.0)
    >>> adj.mutate(arr)
    >>> arr
    array([[  0.,   1.,   2.],
           [  3.,  16.,   5.],
           [  6.,  28.,   8.]])
    """

    cpdef mutate(self, float64_t[:, :] data):
        cdef Py_ssize_t row, col
        col = self.col

        # last_row + 1 because last_row is the last index that **should be**
        # affected.
        for row in range(self.first_row, self.last_row + 1):
            data[row, col] *= self.value


cdef class Float64Overwrite(Float64Adjustment):
    """
    An adjustment that overwrites with a scalar.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(9, dtype=float).reshape(3, 3)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])

    >>> adj = Float64Overwrite(first_row=1, last_row=2, col=1, value=0.0)
    >>> adj.mutate(arr)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  0.,  5.],
           [ 6.,  0.,  8.]])
    """

    cpdef mutate(self, float64_t[:, :] data):
        cdef Py_ssize_t row, col
        col = self.col

        # last_row + 1 because last_row is the last index that **should be**
        # affected.
        for row in range(self.first_row, self.last_row + 1):
            data[row, col] = self.value


cpdef Float64Multiply from_assets_and_dates_f64mul(DatetimeIndex_t dates_index,
                                                   Int64Index_t assets_index,
                                                   Timestamp_t start_date,
                                                   Timestamp_t end_date,
                                                   int asset_id,
                                                   float64_t value):
    """
    Helper for constructing a Float64Multiply instance from coordinates in
    assets/dates indices.

    Example
    -------

    >>> from pandas import date_range, Int64Index, Timestamp
    >>> dates = date_range('2014-01-01', '2014-01-07')
    >>> assets = Int64Index(range(10))
    >>> from_assets_and_dates_f64mul(
    ...     dates,
    ...     assets,
    ...     Timestamp('2014-01-03'),
    ...     Timestamp('2014-01-05'),
    ...     3,
    ...     0.5,
    ... )
    Float64Multiply(first_row=2, last_row=4, col=3, value=0.500000)
    """
    cdef:
        Py_ssize_t first_row, last_row, col

    first_row, last_row, col = get_adjustment_locs(
        dates_index,
        assets_index,
        start_date,
        end_date,
        asset_id,
    )
    return Float64Multiply(first_row, last_row, col, value)
