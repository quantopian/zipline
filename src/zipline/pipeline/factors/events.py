"""
Factors describing information about event data (e.g. earnings
announcements, acquisitions, dividends, etc.).
"""
from numpy import newaxis
from zipline.utils.numpy_utils import (
    NaTD,
    busday_count_mask_NaT,
    datetime64D_dtype,
    float64_dtype,
)

from .factor import Factor


class BusinessDaysSincePreviousEvent(Factor):
    """
    Abstract class for business days since a previous event.
    Returns the number of **business days** (not trading days!) since
    the most recent event date for each asset.

    This doesn't use trading days for symmetry with
    BusinessDaysUntilNextEarnings.

    Assets which announced or will announce the event today will produce a
    value of 0.0. Assets that announced the event on the previous business
    day will produce a value of 1.0.

    Assets for which the event date is `NaT` will produce a value of `NaN`.


    Example
    -------
    ``BusinessDaysSincePreviousEvent`` can be used to create an event-driven
    factor. For instance, you may want to only trade assets that have
    a data point with an asof_date in the last 5 business days. To do this,
    you can create a ``BusinessDaysSincePreviousEvent`` factor, supplying
    the relevant asof_date column from your dataset as input, like this::

        # Factor computing number of days since most recent asof_date
        # per asset.
        days_since_event = BusinessDaysSincePreviousEvent(
            inputs=[MyDataset.asof_date]
        )

        # Filter returning True for each asset whose most recent asof_date
        # was in the last 5 business days.
        recency_filter = (days_since_event <= 5)

    """

    window_length = 0
    dtype = float64_dtype

    def _compute(self, arrays, dates, assets, mask):

        # Coerce from [ns] to [D] for numpy busday_count.
        announce_dates = arrays[0].astype(datetime64D_dtype)

        # Set masked values to NaT.
        announce_dates[~mask] = NaTD

        # Convert row labels into a column vector for broadcasted comparison.
        reference_dates = dates.values.astype(datetime64D_dtype)[:, newaxis]
        return busday_count_mask_NaT(announce_dates, reference_dates)


class BusinessDaysUntilNextEvent(Factor):
    """
    Abstract class for business days since a next event.
    Returns the number of **business days** (not trading days!) until
    the next known event date for each asset.

    This doesn't use trading days because the trading calendar includes
    information that may not have been available to the algorithm at the time
    when `compute` is called.

    For example, the NYSE closings September 11th 2001, would not have been
    known to the algorithm on September 10th.

    Assets that announced or will announce the event today will produce a value
    of 0.0.  Assets that will announce the event on the next upcoming business
    day will produce a value of 1.0.

    Assets for which the event date is `NaT` will produce a value of `NaN`.
    """

    window_length = 0
    dtype = float64_dtype

    def _compute(self, arrays, dates, assets, mask):

        # Coerce from [ns] to [D] for numpy busday_count.
        announce_dates = arrays[0].astype(datetime64D_dtype)

        # Set masked values to NaT.
        announce_dates[~mask] = NaTD

        # Convert row labels into a column vector for broadcasted comparison.
        reference_dates = dates.values.astype(datetime64D_dtype)[:, newaxis]
        return busday_count_mask_NaT(reference_dates, announce_dates)
