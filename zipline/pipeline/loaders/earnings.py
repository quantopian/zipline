"""
Reference implementation for EarningsCalendar loaders.
"""
from numpy import full_like
import pandas as pd
from six import iteritems

from zipline.utils.memoize import lazyval

from .base import PipelineLoader
from .frame import DataFrameLoader
from ..data.earnings import EarningsCalendar


class EarningsCalendarLoader(PipelineLoader):
    """
    Reference loader for `zipline.pipeline.data.earnings.EarningsCalendar`.

    Does not currently support adjustments to the dates of known earnings.

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Index of dates for which we can serve queries.
    announcement_dates : dict[int -> DatetimeIndex]
        Dict mapping column labels to an index of dates on which earnings were
        announced.
    """
    def __init__(self, all_dates, announcement_dates):
        self._all_dates = all_dates
        self._announcment_dates = announcement_dates

    def get_loader(self, column):
        """
        Dispatch to the loader for `column`.
        """
        if column is EarningsCalendar.next_announcement:
            return self.next_announcement_loader
        elif column is EarningsCalendar.previous_announcement:
            return self.previous_annoucement_loader
        else:
            raise ValueError("Don't know how to load column %s." % column)

    @lazyval
    def next_announcement_loader(self):
        return DataFrameLoader(
            EarningsCalendar.next_announcement,
            next_earnings_date_frame(
                self._all_dates,
                self._announcement_dates,
            ),
            adjustments=None,
        )

    @lazyval
    def previous_announcement_loader(self):
        return DataFrameLoader(
            EarningsCalendar.previous_announcement,
            previous_earnings_date_frame(
                self._all_dates,
                self._announcement_dates,
            ),
            adjustments=None,
        )

    def load_adjusted_array(self, columns, dates, assets, mask):
        return {
            column: self.get_loader(column).load_adjusted_array(
                [column], dates, assets, mask
            )
            for column in columns
        }


def next_earnings_date_frame(dates, announcement_dates):
    """
    Make a DataFrame representing simulated next earnings dates.

    Parameters
    ----------
    dates : DatetimeIndex.
        The index of the returned DataFrame.
    announcement_dates : dict[int -> DatetimeIndex]
        Dict mapping sids to an index of dates on which earnings were announced
        for that sid.

    Returns
    -------
    next_earnings: pd.DataFrame
        A DataFrame representing, for each (label, date) pair, the first entry
        in `earnings_calendars[label]` on or after `date`.  Entries falling
        after the last date in a calendar will have `NaT` as the result in the
        output.

    See Also
    --------
    next_earnings_date_frame
    """
    cols = {equity: full_like(dates, "NaT") for equity in announcement_dates}
    for equity, earnings_dates in iteritems(announcement_dates):
        next_dt_indices = earnings_dates.searchsorted(dates)
        mask = next_dt_indices < len(earnings_dates)
        cols[equity][mask] = earnings_dates[next_dt_indices[mask]]

    return pd.DataFrame(index=dates, data=cols)


def previous_earnings_date_frame(dates, announcement_dates):
    """
    Make a DataFrame representing simulated next earnings dates.

    Parameters
    ----------
    dates : DatetimeIndex.
        The index of the returned DataFrame.
    announcement_dates : dict[int -> DatetimeIndex]
        Dict mapping sids to an index of dates on which earnings were announced
        for that sid.

    Returns
    -------
    prev_earnings: pd.DataFrame
        A DataFrame representing, for (label, date) pair, the first entry in
        `announcement_dates[label]` strictly before `date`.  Entries falling
        before the first date in a calendar will have `NaT` as the result in
        the output.

    See Also
    --------
    next_earnings_date_frame
    """
    cols = {equity: full_like(dates, "NaT") for equity in announcement_dates}
    for equity, earnings_dates in iteritems(announcement_dates):
        # Subtract one to roll back to the index of the previous date.
        prev_dt_indices = earnings_dates.searchsorted(dates) - 1
        mask = prev_dt_indices > 0
        cols[equity][mask] = earnings_dates[prev_dt_indices[mask]]

    return pd.DataFrame(index=dates, data=cols)
