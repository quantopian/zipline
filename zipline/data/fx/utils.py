import numpy as np


def check_dts(stored_dts, requested_dts):
    """
    Validate that ``requested_dts`` are valid for querying from an FX reader
    that has data for ``stored_dts``.
    """
    request_end = requested_dts[-1]
    data_end = stored_dts[-1]

    if not is_sorted_ascending(requested_dts):
        raise ValueError("Requested fx rates with non-ascending dts.")

    if request_end > data_end:
        raise ValueError(
            "Requested fx rates ending at {}, but data ends at {}"
            .format(request_end, data_end)
        )


def is_sorted_ascending(array):
    return (np.maximum.accumulate(array) <= array).all()
