"""
Algorithms for computing quantiles on numpy arrays.
"""
from numpy.lib import apply_along_axis
from pandas import qcut


def quantiles(data, nbins_or_partition_bounds):
    """
    Compute rowwise array quantiles on an input.
    """
    return apply_along_axis(
        qcut,
        1,
        data,
        q=nbins_or_partition_bounds,
        labels=False,
    )
