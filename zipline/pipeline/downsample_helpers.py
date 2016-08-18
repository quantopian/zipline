"""
Helpers for downsampling code.
"""
from operator import attrgetter

from zipline.utils.input_validation import expect_element
from zipline.utils.numpy_utils import changed_locations
from zipline.utils.sharedoc import (
    templated_docstring,
    PIPELINE_DOWNSAMPLING_FREQUENCY_DOC,
)

_dt_to_period = {
    'year_start': attrgetter('year'),
    'quarter_start': attrgetter('quarter'),
    'month_start': attrgetter('month'),
    'week_start': attrgetter('week'),
}

SUPPORTED_DOWNSAMPLE_FREQUENCIES = frozenset(_dt_to_period)


expect_downsample_frequency = expect_element(
    frequency=SUPPORTED_DOWNSAMPLE_FREQUENCIES,
)


@expect_downsample_frequency
@templated_docstring(frequency=PIPELINE_DOWNSAMPLING_FREQUENCY_DOC)
def select_sampling_indices(dates, frequency):
    """
    Choose entries from ``dates`` to use for downsampling at ``frequency``.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates from which to select sample choices.
    {frequency}

    Returns
    -------
    indices : np.array[int64]
        An array condtaining indices of dates on which samples should be taken.

        The resulting index will always include 0 as a sample index, and it
        will include the first date of each subsequent year/quarter/month/week,
        as determined by ``frequency``.

    Notes
    -----
    This function assumes that ``dates`` does not have large gaps.

    In particular, it assumes that the maximum distance between any two entries
    in ``dates`` is never greater than a year, which we rely on because we use
    ``np.diff(dates.<frequency>)`` to find dates where the sampling
    period has changed.
    """
    return changed_locations(
        _dt_to_period[frequency](dates),
        include_first=True
    )
