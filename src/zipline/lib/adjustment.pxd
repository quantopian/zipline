cimport numpy as np

# Purely for readability. There aren't C-level declarations for these types.
ctypedef object Int64Index_t
ctypedef object DatetimeIndex_t
ctypedef object Timestamp_t


ctypedef fused column_type:
    np.int64_t
    np.float64_t
    np.uint8_t
    object


cpdef enum AdjustmentKind:
    MULTIPLY = 0
    ADD = 1
    OVERWRITE = 2


cdef Adjustment make_adjustment_from_indices_fused(Py_ssize_t first_row,
                                                   Py_ssize_t last_row,
                                                   Py_ssize_t first_column,
                                                   Py_ssize_t last_column,
                                                   AdjustmentKind adjustment_kind,
                                                   column_type value)


cdef class Adjustment:
    """
    Base class for Adjustments.

    Subclasses should inherit and provide a `value` attribute and a `mutate`
    method.
    """
    cdef readonly Py_ssize_t first_col, last_col, first_row, last_row
    cpdef tuple _key(self)


cdef class Float64Adjustment(Adjustment):
    """
    Base class for adjustments that operate on Float64 data.
    """
    cdef public np.float64_t value


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

    cpdef mutate(self, np.float64_t[:, :] data)


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

    cpdef mutate(self, np.float64_t[:, :] data)


cdef class ArrayAdjustment(Adjustment):
    """
    Base class for ArrayAdjustments.

    Subclasses should inherit and provide a `values` attribute and a `mutate`
    method.
    """


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
    cdef public np.float64_t[:] values
    cpdef mutate(self, np.float64_t[:, :] data)


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
    cdef public np.int64_t[:] values
    cpdef mutate(self, np.int64_t[:, :] data)


cdef class Object1DArrayOverwrite(ArrayAdjustment):
    """
    An adjustment that overwrites subarrays with a value for each subarray.
    """
    cdef public object values
    cpdef mutate(self, object data)


cdef class Boolean1DArrayOverwrite(ArrayAdjustment):
    """
    An adjustment that overwrites subarrays with a value for each subarray.
    """
    cdef public np.uint8_t[:] values
    cpdef mutate(self, np.uint8_t[:, :] data)


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

    cpdef mutate(self, np.float64_t[:, :] data)


cdef class _Int64Adjustment(Adjustment):
    """
    Base class for adjustments that operate on integral data.

    This is private because we never actually operate on integers as data, but
    we use integer arrays to represent datetime and timedelta data.
    """
    cdef public np.int64_t value


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
    cpdef mutate(self, np.int64_t[:, :] data)


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
    cpdef mutate(self, np.int64_t[:, :] data)


cdef class _ObjectAdjustment(Adjustment):
    """
    Base class for adjustments that operate on arbitrary objects.

    We use only this for categorical data, where our data buffer is an array of
    indices into an array of unique Python string objects.
    """
    cdef public object value


cdef class ObjectOverwrite(_ObjectAdjustment):
    cpdef mutate(self, object data)


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
    cdef public np.uint8_t value


cdef class BooleanOverwrite(BooleanAdjustment):
    """
    An adjustment that overwrites with a boolean.

    This operates on uint8 data.
    """
    cpdef mutate(self, np.uint8_t[:, :] data)
