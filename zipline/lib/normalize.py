import numpy as np


def naive_grouped_rowwise_apply(data,
                                group_labels,
                                func,
                                func_args=(),
                                out=None):
    """
    Simple implementation of grouped row-wise function application.

    Parameters
    ----------
    data : ndarray[ndim=2]
        Input array over which to apply a grouped function.
    group_labels : ndarray[ndim=2, dtype=int64]
        Labels to use to bucket inputs from array.
        Should be the same shape as array.
    func : function[ndarray[ndim=1]] -> function[ndarray[ndim=1]]
        Function to apply to pieces of each row in array.
    func_args : tuple
        Additional positional arguments to provide to each row in array.
    out : ndarray, optional
        Array into which to write output.  If not supplied, a new array of the
        same shape as ``data`` is allocated and returned.

    Examples
    --------
    >>> data = np.array([[1., 2., 3.],
    ...                  [2., 3., 4.],
    ...                  [5., 6., 7.]])
    >>> labels = np.array([[0, 0, 1],
    ...                    [0, 1, 0],
    ...                    [1, 0, 2]])
    >>> naive_grouped_rowwise_apply(data, labels, lambda row: row - row.min())
    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  2.],
           [ 0.,  0.,  0.]])
    >>> naive_grouped_rowwise_apply(data, labels, lambda row: row / row.sum())
    array([[ 0.33333333,  0.66666667,  1.        ],
           [ 0.33333333,  1.        ,  0.66666667],
           [ 1.        ,  1.        ,  1.        ]])
    """
    if out is None:
        out = np.empty_like(data)

    for (row, label_row, out_row) in zip(data, group_labels, out):
        for label in np.unique(label_row):
            locs = (label_row == label)
            out_row[locs] = func(row[locs], *func_args)
    return out
