"""
Factors describing information about event data (e.g. earnings
announcements, acquisitions, dividends, etc.).
"""
from numpy import newaxis
from ..data import (
    _13DFilings,
    BuybackAuthorizations,
    DividendsByAnnouncementDate,
    DividendsByExDate,
    EarningsCalendar
)
from zipline.utils.numpy_utils import (
    NaTD,
    busday_count_mask_NaT,
    datetime64D_dtype,
    float64_dtype,
)

from .factor import Factor


class BusinessDaysSincePreviousEvents(Factor):
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


class BusinessDaysUntilNextEvents(Factor):
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


class BusinessDaysUntilNextEarnings(BusinessDaysUntilNextEvents):
    """
    Factor returning the number of **business days** (not trading days!) until
    the next known earnings date for each asset.

    Assets that announced or will announce earnings on the day of ``compute``
    will produce a value of 0.0.  Assets that will announce the event on the
    next upcoming business day will produce a value of 1.0.

    Assets for which the date of the next earnings announcement is ``NaT`` will
    produce a value of ``NaN``.  This most commonly occurs because many
    companies do not publish the exact date of their upcoming earnings
    announcements until a few weeks before the announcement.

    See Also
    --------
    BusinessDaysSincePreviousEarnings
    """
    inputs = [EarningsCalendar.next_announcement]


class BusinessDaysSincePreviousEarnings(BusinessDaysSincePreviousEvents):
    """
    Factor returning the number of **business days** (not trading days!) since
    the most recent earnings date for each asset.

    Assets that announced or will announce earnings on the day of ``compute``
    will produce a value of 0.0.  Assets that will announce the event on the
    next upcoming business day will produce a value of 1.0.

    Assets which announced or will announce the earnings today will produce a
    value of 0.0. Assets that announced the on the previous business day will
    produce a value of 1.0.

    Assets for which the previous earnings date is `NaT` will produce a value
    of `NaN`.  This will happen in the interval between IPO and first earnings
    for most companies.

    See Also
    --------
    BusinessDaysUntilNextEarnings
    """
    inputs = [EarningsCalendar.previous_announcement]


class BusinessDaysSinceBuybackAuth(
    BusinessDaysSincePreviousEvents
):
    """
    Factor returning the number of **business days** (not trading days!) since
    the most recent buyback authorization for each asset.

    See Also
    --------
    zipline.pipeline.factors.BusinessDaysSinceBuybackAuth
    """
    inputs = [BuybackAuthorizations.previous_date]


class BusinessDaysSinceDividendAnnouncement(
    BusinessDaysSincePreviousEvents
):
    """
    Factor returning the number of **business days** (not trading days!) since
    the most recent dividend announcement for each asset.


    See Also
    --------
    zipline.pipeline.factors.BusinessDaysSinceDividendAnnouncement
    """
    inputs = [DividendsByAnnouncementDate.previous_announcement_date]


class BusinessDaysUntilNextExDate(
    BusinessDaysUntilNextEvents
):
    """
    Factor returning the number of **business days** (not trading days!) until
    the next ex date for each asset.


    See Also
    --------
    zipline.pipeline.factors.BusinessDaysSinceDividendAnnouncement
    """
    inputs = [DividendsByExDate.next_date]


class BusinessDaysSincePreviousExDate(
    BusinessDaysSincePreviousEvents
):
    """
    Factor returning the number of **business days** (not trading days!) since
    the most recent ex date for each asset.


    See Also
    --------
    zipline.pipeline.factors.BusinessDaysSinceDividendAnnouncement
    """
    inputs = [DividendsByExDate.previous_date]


class BusinessDaysSince13DFilingsDate(
    BusinessDaysSincePreviousEvents
):
    """
    Factor returning the number of **business days** (not trading days!) since
    the most recent 13d filings for each asset.

    """
    inputs = [_13DFilings.disclosure_date]
