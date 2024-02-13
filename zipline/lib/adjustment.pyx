# cython: embedsignature=True
from cpython cimport Py_EQ

cimport cython
from pandas import isnull, Timestamp
cimport numpy as np
from numpy cimport float64_t, uint8_t, int64_t
from numpy import asarray, datetime64, float64, int64, bool_, uint8

from zipline.utils.compat import unicode


ADJUSTMENT_KIND_NAMES = {
    MULTIPLY: 'MULTIPLY',
    ADD: 'ADD',
    OVERWRITE: 'OVERWRITE',
}

cdef dict _float_adjustment_types = {
    ADD: Float64Add,
    MULTIPLY: Float64Multiply,
    OVERWRITE: Float64Overwrite,
}
cdef dict _datetime_adjustment_types = {
    OVERWRITE: Datetime64Overwrite,
}
cdef dict _object_adjustment_types = {
    OVERWRITE: ObjectOverwrite,
}
cdef dict _int_adjustment_types = {
    OVERWRITE: Int64Overwrite,
}
cdef dict _boolean_adjustment_types = {
    OVERWRITE: BooleanOverwrite,
}


cdef _is_float(object value):
    return isinstance(value, (float, float64))


cdef _is_datetime(object value):
    return isinstance(value, (datetime64, Timestamp))


cdef _is_int(object value):
    return isinstance(value, (int, int64))


cdef _is_obj(object value):
    return isinstance(value, (bytes, unicode, type(None)))


cdef _is_bool(object value):
    return isinstance(value, (bool, bool_, uint8))


cpdef choose_adjustment_type(AdjustmentKind adjustment_kind, object value):
    """
    Make an adjustment object of the type appropriate for the given kind and
    value.

    Parameters
    ----------
    adjustment_kind : {ADD, MULTIPLY, OVERWRITE}
        The kind of adjustment to construct.
    value : object
        The value parameter to the adjustment.  Only floating-point values and
        datetime-like values are currently supported
    """
    if adjustment_kind in (ADD, MULTIPLY):
        if not _is_float(value):
            raise TypeError(
                "Can't construct %s Adjustment with value of type %r.\n"
                "ADD and MULTIPLY adjustments are only supported for "
                "floating point data." % (
                    ADJUSTMENT_KIND_NAMES[adjustment_kind],
                    type(value),
                )
            )
        return _float_adjustment_types[adjustment_kind]

    elif adjustment_kind == OVERWRITE:
        if _is_float(value):
            return _float_adjustment_types[adjustment_kind]
        elif _is_datetime(value):
            return _datetime_adjustment_types[adjustment_kind]
        elif _is_bool(value):
            return _boolean_adjustment_types[adjustment_kind]
        elif _is_int(value):
            return _int_adjustment_types[adjustment_kind]
        elif _is_obj(value):
            return _object_adjustment_types[adjustment_kind]
        else:
            raise TypeError(
                "Don't know how to make overwrite "
                "adjustments for values of type %r." % type(value),
            )

    else:
        raise ValueError("Unknown adjustment type %d." % adjustment_kind)


cpdef Adjustment make_adjustment_from_indices(Py_ssize_t first_row,
                                              Py_ssize_t last_row,
                                              Py_ssize_t first_column,
                                              Py_ssize_t last_column,
                                              AdjustmentKind adjustment_kind,
                                              object value):
    """
    Make an Adjustment object from row/column indices into a baseline array.
    """
    cdef type type_ = choose_adjustment_type(adjustment_kind, value)
    # NOTE_SS: Cython appears to generate incorrect code here if values are
    # passed by name.  This is true even if cython.always_allow_keywords is
    # enabled.  Yay Cython.
    return type_(first_row, last_row, first_column, last_column, value)


cdef type _choose_adjustment_type(AdjustmentKind adjustment_kind,
                                  column_type value):
    if adjustment_kind in (ADD, MULTIPLY):
        if column_type is np.float64_t:
            return _float_adjustment_types[adjustment_kind]

        raise TypeError(
            "Can't construct %s Adjustment with value of type %r.\n"
            "ADD and MULTIPLY adjustments are only supported for "
            "floating point data." % (
                ADJUSTMENT_KIND_NAMES[adjustment_kind],
                type(value),
            )
        )

    elif adjustment_kind == OVERWRITE:
        if column_type is np.float64_t:
            return _float_adjustment_types[adjustment_kind]
        elif column_type is np.int64_t:
            return _int_adjustment_types[adjustment_kind]
        elif column_type is np.uint8_t:
            return _boolean_adjustment_types[adjustment_kind]
        elif column_type is object:
            return _object_adjustment_types[adjustment_kind]
        else:
            raise TypeError(
                "Don't know how to make overwrite "
                "adjustments for values of type %r." % type(value),
            )

    else:
        raise ValueError("Unknown adjustment type %d." % adjustment_kind)


cdef Adjustment make_adjustment_from_indices_fused(Py_ssize_t first_row,
                                                   Py_ssize_t last_row,
                                                   Py_ssize_t first_column,
                                                   Py_ssize_t last_column,
                                                   AdjustmentKind adjustment_kind,
                                                   column_type value):
    """
    Make an Adjustment object from row/column indices into a baseline array.
    """
    cdef type type_ = _choose_adjustment_type(adjustment_kind, value)
    # NOTE_SS: Cython appears to generate incorrect code here if values are
    # passed by name.  This is true even if cython.always_allow_keywords is
    # enabled.  Yay Cython.
    return type_(first_row, last_row, first_column, last_column, value)


cpdef make_adjustment_from_labels(DatetimeIndex_t dates_index,
                                  Int64Index_t assets_index,
                                  Timestamp_t start_date,
                                  Timestamp_t end_date,
                                  np.int64_t asset_id,
                                  AdjustmentKind adjustment_kind,
                                  object value):
    """
    Make an Adjustment object from date/asset labels into a labelled baseline
    array.
    """
    cdef type type_ = choose_adjustment_type(adjustment_kind, value)
    return type_.from_assets_and_dates(
        dates_index,
        assets_index,
        start_date,
        end_date,
        asset_id,
        value,
    )


cpdef tuple get_adjustment_locs(DatetimeIndex_t dates_index,
                                Int64Index_t assets_index,
                                Timestamp_t start_date,
                                Timestamp_t end_date,
                                np.int64_t asset_id):
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

    # None or NaT signifies "All values before the end_date".
    if isnull(start_date):
        start_date_loc = 0
    else:
        # Location of earliest date on or after start_date.
        start_date_loc = dates_index.get_loc(start_date, method='bfill')

    return (
        start_date_loc,
        # Location of latest date on or before start_date.
        dates_index.get_loc(end_date, method='ffill'),
        assets_index.get_loc(asset_id),  # Must be exact match.
    )


cpdef _from_assets_and_dates(cls,
                             DatetimeIndex_t dates_index,
                             Int64Index_t assets_index,
                             Timestamp_t start_date,
                             Timestamp_t end_date,
                             np.int64_t asset_id,
                             object value):
    """
    Helper for constructing an Adjustment instance from coordinates in
    assets/dates indices.

    Example
    -------

    >>> from pandas import date_range, Int64Index, Timestamp
    >>> dates = date_range('2014-01-01', '2014-01-07')
    >>> assets = Int64Index(range(10))
    >>> Float64Multiply.from_assets_and_dates(
    ...     dates,
    ...     assets,
    ...     Timestamp('2014-01-03'),
    ...     Timestamp('2014-01-05'),
    ...     3,
    ...     0.5,
    ... )
    Float64Multiply(first_row=2, last_row=4, first_col=3, last_col=3, value=0.500000)
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
    return cls(
        first_row=first_row,
        last_row=last_row,
        first_col=col,
        last_col=col,
        value=value,
    )


cdef class Adjustment:
    """
    Base class for Adjustments.

    Subclasses should inherit and provide a `value` attribute and a `mutate`
    method.
    """
    def __init__(self,
                 Py_ssize_t first_row,
                 Py_ssize_t last_row,
                 Py_ssize_t first_col,
                 Py_ssize_t last_col):
        if not (0 <= first_row <= last_row):
            raise ValueError(
                'first_row must be in the range [0, last_row], got:'
                ' first_row=%s last_row=%s' % (
                    first_row,
                    last_row,
                ),
            )
        if not (0 <= first_col <= last_col):
            raise ValueError(
                'first_col must be in the range [0, last_col], got:'
                ' first_col=%s last_col=%s' % (
                    first_col,
                    last_col,
                ),
            )

        self.first_col = first_col
        self.last_col = last_col
        self.first_row = first_row
        self.last_row = last_row

    from_assets_and_dates = classmethod(_from_assets_and_dates)

    def __richcmp__(self, object other, int op):
        """
        Rich comparison method.  Only Equality is defined.
        """
        if op != Py_EQ or type(self) != type(other):
            return NotImplemented

        return self._key() == other._key()

    cpdef tuple _key(self):
        """
        Comparison key
        """
        return (
            self.first_row,
            self.last_row,
            self.first_col,
            self.last_col,
            self.value,
        )

    def __reduce__(self):
        return type(self), self._key()


cdef class Float64Adjustment(Adjustment):
    """
    Base class for adjustments that operate on Float64 data.
    """
    def __init__(self,
                 Py_ssize_t first_row,
                 Py_ssize_t last_row,
                 Py_ssize_t first_col,
                 Py_ssize_t last_col,
                 float64_t value):

        super(Float64Adjustment, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        self.value = value

    from_assets_and_dates = classmethod(_from_assets_and_dates)

    def __repr__(self):
        return (
            "%s(first_row=%d, last_row=%d,"
            " first_col=%d, last_col=%d, value=%f)" % (
                type(self).__name__,
                self.first_row,
                self.last_row,
                self.first_col,
                self.last_col,
                self.value,
            )
        )


cdef class Float64Multiply(Float64Adjustment):
    """
    An adjustment that multiplies by a float.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(9, dtype=float).reshape(3, 3)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])

    >>> adj = Float64Multiply(
    ...     first_row=1,
    ...     last_row=2,
    ...     first_col=1,
    ...     last_col=2,
    ...     value=4.0,
    ... )
    >>> adj.mutate(arr)
    >>> arr
    array([[  0.,   1.,   2.],
           [  3.,  16.,  20.],
           [  6.,  28.,  32.]])
    """

    cpdef mutate(self, float64_t[:, :] data):
        cdef Py_ssize_t row, col
        cdef float64_t value = self.value

        # last_col + 1 because last_col should also be affected.
        for col in range(self.first_col, self.last_col + 1):
            # last_row + 1 because last_row should also be affected.
            for row in range(self.first_row, self.last_row + 1):
                data[row, col] *= value


cdef class Float64Overwrite(Float64Adjustment):
    """
    An adjustment that overwrites with a float.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(9, dtype=float).reshape(3, 3)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])

    >>> adj = Float64Overwrite(
    ...     first_row=1,
    ...     last_row=2,
    ...     first_col=1,
    ...     last_col=2,
    ...     value=0.0,
    ... )
    >>> adj.mutate(arr)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  0.,  0.],
           [ 6.,  0.,  0.]])
    """

    cpdef mutate(self, float64_t[:, :] data):
        cdef Py_ssize_t row, col
        cdef float64_t value = self.value

        # last_col + 1 because last_col should also be affected.
        for col in range(self.first_col, self.last_col + 1):
            # last_row + 1 because last_row should also be affected.
            for row in range(self.first_row, self.last_row + 1):
                data[row, col] = value


cdef class ArrayAdjustment(Adjustment):
    """
    Base class for ArrayAdjustments.

    Subclasses should inherit and provide a `values` attribute and a `mutate`
    method.
    """
    def __repr__(self):
            return (
                "%s(first_row=%d, last_row=%d,"
                " first_col=%d, last_col=%d, values=%s)" % (
                    type(self).__name__,
                    self.first_row,
                    self.last_row,
                    self.first_col,
                    self.last_col,
                    asarray(self.values),
                )
            )

cdef class Float641DArrayOverwrite(ArrayAdjustment):
    """
    An adjustment that overwrites subarrays with a value for each subarray.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(25, dtype=float).reshape(5, 5)
    >>> arr
    array([[  0.,   1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.],
           [ 15.,  16.,  17.,  18.,  19.],
           [ 20.,  21.,  22.,  23.,  24.]])
    >>> adj = Float641DArrayOverwrite(
    ...     row_start=0,
    ...     row_end=3,
    ...     column_start=0,
    ...     column_end=0,
    ...     values=np.array([1, 2, 3, 4]),
    )
    >>> adj.mutate(arr)
    >>> arr
    array([[  1.,   1.,   2.,   3.,   4.],
           [  2.,   6.,   7.,   8.,   9.],
           [ 3.,  11.,  12.,  13.,  14.],
           [ 4.,  16.,  17.,  18.,  19.],
           [ 20.,  21.,  22.,  23.,  24.]])
    """
    def __init__(self,
                 int64_t first_row,
                 int64_t last_row,
                 int64_t first_col,
                 int64_t last_col,
                 float64_t[:] values):
        super(Float641DArrayOverwrite, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        if last_row + 1 - first_row != len(values):
            raise ValueError(
                "Mismatch: got %d values for rows starting at index %d and "
                "ending at index %d." % (len(values), first_row, last_row)
            )
        self.values = values

    cpdef mutate(self, float64_t[:, :] data):
        cdef Py_ssize_t i, row, col
        cdef float64_t[:] values = self.values
        for col in range(self.first_col, self.last_col + 1):
            for i, row in enumerate(range(self.first_row, self.last_row + 1)):
                data[row, col] = values[i]


cdef class Datetime641DArrayOverwrite(ArrayAdjustment):
    """
    An adjustment that overwrites subarrays with a value for each subarray.

    Example
    -------

    >>> import numpy as np; import pandas as pd
    >>> dts = pd.date_range('2014', freq='D', periods=9, tz='UTC')
    >>> arr = dts.values.reshape(3, 3)
    >>> arr == np.datetime64(0, 'ns')
    array([[False, False, False],
       [False, False, False],
       [False, False, False]], dtype=bool)
    >>> adj = Datetime641DArrayOverwrite(
    ...           first_row=1,
    ...           last_row=2,
    ...           first_col=1,
    ...           last_col=2,
    ...           values=np.array([
    ...               np.datetime64(0, 'ns'),
    ...               np.datetime64(1, 'ns')
    ...           ])
    ...       )
    >>> adj.mutate(arr.view(np.int64))
    >>> arr == np.datetime64(0, 'ns')
    array([[False, False, False],
       [False,  True,  True],
       [False, False, False]], dtype=bool)
    >>> arr == np.datetime64(1, 'ns')
    array([[False, False, False],
       [False, False, False],
       [False,  True,  True]], dtype=bool)
    """
    def __init__(self,
                 int64_t first_row,
                 int64_t last_row,
                 int64_t first_col,
                 int64_t last_col,
                 object values):
        super(Datetime641DArrayOverwrite, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        if last_row + 1 - first_row != len(values):
            raise ValueError("Mismatch: got %d values for rows starting at"
            " index %d and ending at index %d." % (
                len(values), first_row, last_row)
            )
        self.values = asarray([datetime_to_int(value) for value in values])

    cpdef mutate(self, int64_t[:, :] data):
        cdef Py_ssize_t i, row, col
        cdef int64_t[:] values = self.values
        for col in range(self.first_col, self.last_col + 1):
            for i, row in enumerate(range(self.first_row, self.last_row + 1)):
                data[row, col] = values[i]

    def __reduce__(self):
        return type(self), (
            self.first_row,
            self.last_row,
            self.first_col,
            self.last_col,
            self.values.view('datetime64[ns]'),
        )


cdef class Object1DArrayOverwrite(ArrayAdjustment):
    def __init__(self,
                 int64_t first_row,
                 int64_t last_row,
                 int64_t first_col,
                 int64_t last_col,
                 object values):
        super(Object1DArrayOverwrite, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        if last_row + 1 - first_row != len(values):
            raise ValueError(
                "Mismatch: got %d values for rows starting at index %d and "
                "ending at index %d." % (len(values), first_row, last_row)
            )
        self.values = values[:, None]

    cpdef mutate(self, object data):
        # data is an object here because this is intended to be used with a
        # `zipline.lib.LabelArray`.

        data[
            self.first_row:self.last_row + 1,
            self.first_col:self.last_col + 1,
        ] = self.values



cdef class Boolean1DArrayOverwrite(ArrayAdjustment):
    def __init__(self,
                 int64_t first_row,
                 int64_t last_row,
                 int64_t first_col,
                 int64_t last_col,
                 object values):
        if values.dtype.kind != 'b':
            raise TypeError('dtype is not bool, got: %r' % values.dtype)

        super(Boolean1DArrayOverwrite, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )

        if last_row + 1 - first_row != len(values):
            raise ValueError(
                "Mismatch: got %d values for rows starting at index %d and "
                "ending at index %d." % (len(values), first_row, last_row)
            )
        self.values = values.view('uint8')

    cpdef mutate(self, np.uint8_t[:, :] data):
        cdef Py_ssize_t i, row, col
        cdef np.uint8_t[:] values = self.values
        for col in range(self.first_col, self.last_col + 1):
            for i, row in enumerate(range(self.first_row, self.last_row + 1)):
                data[row, col] = values[i]

    def __repr__(self):
        return (
            "%s(first_row=%d, last_row=%d,"
            " first_col=%d, last_col=%d, value=%r)" % (
                type(self).__name__,
                self.first_row,
                self.last_row,
                self.first_col,
                self.last_col,
                self.value.view('?'),
            )
        )

    def __reduce__(self):
        return type(self), (
            self.first_row,
            self.last_row,
            self.first_col,
            self.last_col,
            self.value.view('?'),
        )


cdef class Float64Add(Float64Adjustment):
    """
    An adjustment that adds a float.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(9, dtype=float).reshape(3, 3)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])

    >>> adj = Float64Add(
    ...     first_row=1,
    ...     last_row=2,
    ...     first_col=1,
    ...     last_col=2,
    ...     value=1.0,
    ... )
    >>> adj.mutate(arr)
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  5.,  6.],
           [ 6.,  8.,  9.]])
    """

    cpdef mutate(self, float64_t[:, :] data):
        cdef Py_ssize_t row, col
        cdef float64_t value = self.value

        # last_col + 1 because last_col should also be affected.
        for col in range(self.first_col, self.last_col + 1):
            # last_row + 1 because last_row should also be affected.
            for row in range(self.first_row, self.last_row + 1):
                data[row, col] += value


cdef class _Int64Adjustment(Adjustment):
    """
    Base class for adjustments that operate on integral data.

    This is private because we never actually operate on integers as data, but
    we use integer arrays to represent datetime and timedelta data.
    """
    def __init__(self,
                 Py_ssize_t first_row,
                 Py_ssize_t last_row,
                 Py_ssize_t first_col,
                 Py_ssize_t last_col,
                 int64_t value):
        super(_Int64Adjustment, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        self.value = value

    def __repr__(self):
        return (
            "%s(first_row=%d, last_row=%d,"
            " first_col=%d, last_col=%d, value=%d)" % (
                type(self).__name__,
                self.first_row,
                self.last_row,
                self.first_col,
                self.last_col,
                self.value,
            )
        )


cdef class Int64Overwrite(_Int64Adjustment):
    """
    An adjustment that overwrites with an int.

    Example
    -------

    >>> import numpy as np
    >>> arr = np.arange(9, dtype=int).reshape(3, 3)
    >>> arr
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8]])

    >>> adj = Int64Overwrite(
    ...     first_row=1,
    ...     last_row=2,
    ...     first_col=1,
    ...     last_col=2,
    ...     value=0,
    ... )
    >>> adj.mutate(arr)
    >>> arr
    array([[ 0,  1,  2],
           [ 3,  0,  0],
           [ 6,  0,  0]])
    """

    cpdef mutate(self, int64_t[:, :] data):
        cdef Py_ssize_t row, col
        cdef int64_t value = self.value

        # last_col + 1 because last_col should also be affected.
        for col in range(self.first_col, self.last_col + 1):
            # last_row + 1 because last_row should also be affected.
            for row in range(self.first_row, self.last_row + 1):
                data[row, col] = value


cdef datetime_to_int(object datetimelike):
    """
    Coerce a datetime-like object to the int format used by AdjustedArrays of
    Datetime64 type.
    """
    if isinstance(datetimelike, Timestamp):
        return datetimelike.value

    if not isinstance(datetimelike, datetime64):
        raise TypeError("Expected datetime64, got %s" % type(datetimelike))

    elif datetimelike.dtype.name != 'datetime64[ns]':
        raise TypeError(
            "Expected datetime64[ns], got %s",
            datetimelike.dtype.name,
        )

    return datetimelike.astype(int64)


cdef class Datetime64Adjustment(_Int64Adjustment):
    """
    Base class for adjustments that operate on Datetime64 data.

    Notes
    -----
    Numpy stores datetime64 values in arrays of type int64.  There's no
    straightforward way to work with statically-typed datetime64 data, so
    instead we work with int64 values everywhere, and we do validation/coercion
    at API boundaries.
    """
    def __init__(self,
                 Py_ssize_t first_row,
                 Py_ssize_t last_row,
                 Py_ssize_t first_col,
                 Py_ssize_t last_col,
                 object value):

        super(Datetime64Adjustment, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
            value=datetime_to_int(value),
        )

    def __repr__(self):
        return (
            "%s(first_row=%d, last_row=%d,"
            " first_col=%d, last_col=%d, value=%r)" % (
                type(self).__name__,
                self.first_row,
                self.last_row,
                self.first_col,
                self.last_col,
                datetime64(self.value, 'ns'),
            )
        )

    def __reduce__(self):
        return type(self), (
            self.first_row,
            self.last_row,
            self.first_col,
            self.last_col,
            datetime64(self.value, 'ns'),
        )


cdef class Datetime64Overwrite(Datetime64Adjustment):
    """
    An adjustment that overwrites with a datetime.

    This operates on int64 data which should be interpreted as nanoseconds
    since the epoch.

    Example
    -------

    >>> import numpy as np; import pandas as pd
    >>> dts = pd.date_range('2014', freq='D', periods=9, tz='UTC')
    >>> arr = dts.values.reshape(3, 3)
    >>> arr == np.datetime64(0, 'ns')
    array([[False, False, False],
           [False, False, False],
           [False, False, False]], dtype=bool)
    >>> adj = Datetime64Overwrite(
    ...     first_row=1,
    ...     last_row=2,
    ...     first_col=1,
    ...     last_col=2,
    ...     value=np.datetime64(0, 'ns'),
    ... )
    >>> adj.mutate(arr.view(np.int64))
    >>> arr == np.datetime64(0, 'ns')
    array([[False, False, False],
           [False,  True,  True],
           [False,  True,  True]], dtype=bool)
    """
    cpdef mutate(self, int64_t[:, :] data):
        cdef Py_ssize_t row, col
        cdef int64_t value = self.value

        # last_col + 1 because last_col should also be affected.
        for col in range(self.first_col, self.last_col + 1):
            # last_row + 1 because last_row should also be affected.
            for row in range(self.first_row, self.last_row + 1):
                data[row, col] = value


cdef class _ObjectAdjustment(Adjustment):
    """
    Base class for adjustments that operate on arbitrary objects.

    We use only this for categorical data, where our data buffer is an array of
    indices into an array of unique Python string objects.
    """
    def __init__(self,
                 Py_ssize_t first_row,
                 Py_ssize_t last_row,
                 Py_ssize_t first_col,
                 Py_ssize_t last_col,
                 object value):
        super(_ObjectAdjustment, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        self.value = value

    def __repr__(self):
        return (
            "%s(first_row=%d, last_row=%d,"
            " first_col=%d, last_col=%d, value=%r)" % (
                type(self).__name__,
                self.first_row,
                self.last_row,
                self.first_col,
                self.last_col,
                self.value,
            )
        )


cdef class ObjectOverwrite(_ObjectAdjustment):

    cpdef mutate(self, object data):
        # data is an object here because this is intended to be used with a
        # `zipline.lib.LabelArray`.

        # We don't do this in a loop because we only want to look up the label
        # code in the array's categories once.
        data.set_scalar(
            (
                slice(self.first_row, self.last_row + 1),
                slice(self.first_col, self.last_col + 1),
            ),
            self.value,
        )


cdef class BooleanAdjustment(Adjustment):
    """
    Base class for adjustments that operate on boolean data.

    Notes
    -----
    Numpy stores boolean values in arrays of type uint8.  There's no
    straightforward way to work with statically-typed boolean data, so
    instead we work with uint8 values everywhere, and we do validation/coercion
    at API boundaries.
    """
    def __init__(self,
                 Py_ssize_t first_row,
                 Py_ssize_t last_row,
                 Py_ssize_t first_col,
                 Py_ssize_t last_col,
                 bint value):

        super(BooleanAdjustment, self).__init__(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        self.value = value

    def __repr__(self):
        return (
            "%s(first_row=%d, last_row=%d,"
            " first_col=%d, last_col=%d, value=%r)" % (
                type(self).__name__,
                self.first_row,
                self.last_row,
                self.first_col,
                self.last_col,
                bool(self.value),
            )
        )

    def __reduce__(self):
        return type(self), (
            self.first_row,
            self.last_row,
            self.first_col,
            self.last_col,
            bool(self.value),
        )


cdef class BooleanOverwrite(BooleanAdjustment):
    """
    An adjustment that overwrites with a boolean.

    This operates on uint8 data.
    """
    cpdef mutate(self, uint8_t[:, :] data):
        cdef Py_ssize_t row, col
        cdef uint8_t value = self.value

        # last_col + 1 because last_col should also be affected.
        for col in range(self.first_col, self.last_col + 1):
            # last_row + 1 because last_row should also be affected.
            for row in range(self.first_row, self.last_row + 1):
                data[row, col] = value
