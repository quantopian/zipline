cimport cython
cimport numpy as np
from libc.string cimport memcpy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_i8(const np.uint8_t[::1] column_mask,
                         const np.int64_t[:, :] data,
                         np.int64_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    cdef Py_ssize_t cp_ix
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            for cp_ix in range(cp_size):
                out[cp_ix, out_ix] = data[cp_ix, in_ix]
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_i8_fcontig(const np.uint8_t[::1] column_mask,
                                 const np.int64_t[:, :] data,
                                 np.int64_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            memcpy(
                cython.address(out[0, out_ix]),
                cython.address(data[0, in_ix]),
                cp_size,
            )
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')



@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u1(const np.uint8_t[::1] column_mask,
                         const np.uint8_t[:, :] data,
                         np.uint8_t[:, :] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    cdef Py_ssize_t cp_ix
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            for cp_ix in range(cp_size):
                out[cp_ix, out_ix] = data[cp_ix, in_ix]
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u1_fcontig(const np.uint8_t[::1] column_mask,
                                 const np.uint8_t[:, :] data,
                                 np.uint8_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            memcpy(
                cython.address(out[0, out_ix]),
                cython.address(data[0, in_ix]),
                cp_size,
            )
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')



@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u2(const np.uint8_t[::1] column_mask,
                         const np.uint16_t[:, :] data,
                         np.uint16_t[:, :] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0]
    cdef Py_ssize_t cp_ix
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            for cp_ix in range(cp_size):
                out[cp_ix, out_ix] = data[cp_ix, in_ix]
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u2_fcontig(const np.uint8_t[::1] column_mask,
                                 const np.uint16_t[:, :] data,
                                 np.uint16_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            memcpy(
                cython.address(out[0, out_ix]),
                cython.address(data[0, in_ix]),
                cp_size,
            )
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')



@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u4(const np.uint8_t[::1] column_mask,
                         const np.uint32_t[:, :] data,
                         np.uint32_t[:, :] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0]
    cdef Py_ssize_t cp_ix
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            for cp_ix in range(cp_size):
                out[cp_ix, out_ix] = data[cp_ix, in_ix]
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')

@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u4_fcontig(const np.uint8_t[::1] column_mask,
                           const np.uint32_t[:, :] data,
                           np.uint32_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            memcpy(
                cython.address(out[0, out_ix]),
                cython.address(data[0, in_ix]),
                cp_size,
            )
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')



@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u8(const np.uint8_t[::1] column_mask,
                         const np.uint64_t[:, :] data,
                         np.uint64_t[:, :] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0]
    cdef Py_ssize_t cp_ix
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            for cp_ix in range(cp_size):
                out[cp_ix, out_ix] = data[cp_ix, in_ix]
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_u8_fcontig(const np.uint8_t[::1] column_mask,
                           const np.uint64_t[:, :] data,
                           np.uint64_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            memcpy(
                cython.address(out[0, out_ix]),
                cython.address(data[0, in_ix]),
                cp_size,
            )
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_f8(const np.uint8_t[::1] column_mask,
                         const np.float64_t[:, :] data,
                         np.float64_t[:, :] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0]
    cdef Py_ssize_t cp_ix
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            for cp_ix in range(cp_size):
                out[cp_ix, out_ix] = data[cp_ix, in_ix]
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compress_columns_f8_fcontig(const np.uint8_t[::1] column_mask,
                           const np.float64_t[:, :] data,
                           np.float64_t[:, ::] out):
    cdef Py_ssize_t out_size = out.shape[1]
    cdef Py_ssize_t out_ix = 0
    cdef Py_ssize_t in_ix
    cdef Py_ssize_t cp_size = out.shape[0] * data.itemsize
    for in_ix in range(len(column_mask)):
        if column_mask[in_ix]:
            if out_ix >= out_size:
                raise ValueError('out is not large enough to hold output')
            memcpy(
                cython.address(out[0, out_ix]),
                cython.address(data[0, in_ix]),
                cp_size,
            )
            out_ix += 1

    if out_ix != out_size:
        raise ValueError('out was too large for column mask')


def is_f_contig(data):
    return data.strides[0] == data.itemsize


def compress_columns(column_mask, data, out):
    """Specialization for ``np.compress`` which only works for 2d arrays with
    dtypes that may appear in Pipeline terms. The filtering always happens
    along axis 1 (columns).

    Parameters
    ----------
    column_mask : np.ndarray[bool]
        The mask to select columns. This array must be contiguous.
    data : np.ndarray[any, ndim=2]
        The input array to filter down. For optimal performance, this array
        should be either F-contiguous or at least contiguous along the columns,
        like a slice of rows out of an F-contiguous array.
    out : np.ndarray[any, ndim=2]
        The array to write the filtered results into. The shape of this array
        must be ``(data.shape[0], column_mask.sum())``.

    Returns
    -------
    out : np.ndarray[any, ndim=2]
        The ``out`` parameter.

    Raises
    ------
    ValueError
        Raised when ``out`` is not the correct shape to hold the result of the
        masking.
    """
    cdef bint f_contig = is_f_contig(data)
    mask = column_mask.view('u1')
    if data.dtype.kind == 'f':
        if f_contig:
            compress_columns_f8_fcontig(mask, data, out)
        else:
            compress_columns_f8(mask, data, out)
        return out
    if data.dtype.kind in 'Mi':
        if f_contig:
            compress_columns_i8_fcontig(mask, data.view('i8'), out.view('i8'))
        else:
            compress_columns_i8(mask, data.view('i8'), out.view('i8'))
        return out
    if data.dtype.kind == 'b':
        if f_contig:
            compress_columns_u1_fcontig(mask, data.view('u1'), out.view('u1'))
        else:
            compress_columns_u1(mask, data.view('u1'), out.view('u1'))
        return out
    if data.dtype.kind == 'u':
        if data.itemsize == 1:
            if f_contig:
                compress_columns_u1_fcontig(mask, data, out)
            else:
                compress_columns_u1(mask, data, out)
            return out
        if data.itemsize == 2:
            if f_contig:
                compress_columns_u2_fcontig(mask, data, out)
            else:
                compress_columns_u2(mask, data, out)
            return out
        if data.itemsize == 4:
            if f_contig:
                compress_columns_u4_fcontig(mask, data, out)
            else:
                compress_columns_u4(mask, data, out)
            return out
        if data.itemsize == 8:
            if f_contig:
                compress_columns_u8_fcontig(mask, data, out)
            else:
                compress_columns_u8(mask, data, out)
            return out

    raise TypeError(
        'cannot call compress_columns with array of dtype: %s' % data.dtype,
    )
