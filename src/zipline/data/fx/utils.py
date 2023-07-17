import numpy as np


def check_dts(requested_dts):
    """Validate that ``requested_dts`` are valid for querying from an FX reader."""
    if not is_sorted_ascending(requested_dts):
        raise ValueError("Requested fx rates with non-ascending dts.")


def is_sorted_ascending(array):
    return (np.maximum.accumulate(array) <= array).all()
