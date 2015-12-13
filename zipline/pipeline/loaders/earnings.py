"""
Reference implementation for EarningsCalendar loaders.
"""
from itertools import repeat

from numpy import full_like, full
import pandas as pd
from six import iteritems
from six.moves import zip
from toolz import merge

from .base import PipelineLoader
from .frame import DataFrameLoader
from ..data.earnings import EarningsCalendar
from zipline.utils.numpy_utils import np_NaT
from zipline.utils.memoize import lazyval


class EarningsCalendarLoader(PipelineLoader):
    """
    Reference loader for
    :class:`zipline.pipeline.data.earnings.EarningsCalendar`.

    Does not currently support adjustments to the dates of known earnings.

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Index of dates for which we can serve queries.
    announcement_dates : dict[int -> pd.Series or pd.DatetimeIndex]
        Dict mapping sids to objects representing dates on which earnings
        occurred.

        If a dict value is a Series, it's interpreted as a mapping from the
        date on which we learned an announcement was coming to the date on
        which the announcement was made.

        If a dict value is a DatetimeIndex, it's interpreted as just containing
        the dates that announcements were made, and we assume we knew about the
        announcement on all prior dates.  This mode is only supported if
        ``infer_timestamp`` is explicitly passed as a truthy value.

    infer_timestamps : bool, optional
        Whether to allow passing ``DatetimeIndex`` values in
        ``announcement_dates``.
    """
    def __init__(self, all_dates, announcement_dates, infer_timestamps=False):
        self.all_dates = all_dates

        self.announcement_dates = announcement_dates = (
            announcement_dates.copy()
        )
        dates = self.all_dates.values
        for k, v in iteritems(announcement_dates):
            if isinstance(v, pd.DatetimeIndex):
                if not infer_timestamps:
                    raise ValueError(
                        "Got DatetimeIndex of announcement dates for sid %d.\n"
                        "Pass `infer_timestamps=True` to use the first date in"
                        " `all_dates` as implicit timestamp."
                    )
                # If we are passed a DatetimeIndex, we always have
                # knowledge of the announcements.
                announcement_dates[k] = pd.Series(
                    v, index=repeat(dates[0], len(v)),
                )

    def get_loader(self, column):
        """Dispatch to the loader for ``column``.
        """
        if column is EarningsCalendar.next_announcement:
            return self.next_announcement_loader
        elif column is EarningsCalendar.previous_announcement:
            return self.previous_announcement_loader
        else:
            raise ValueError("Don't know how to load column '%s'." % column)

    @lazyval
    def next_announcement_loader(self):
        return DataFrameLoader(
            EarningsCalendar.next_announcement,
            next_earnings_date_frame(
                self.all_dates,
                self.announcement_dates,
            ),
            adjustments=None,
        )

    @lazyval
    def previous_announcement_loader(self):
        return DataFrameLoader(
            EarningsCalendar.previous_announcement,
            previous_earnings_date_frame(
                self.all_dates,
                self.announcement_dates,
            ),
            adjustments=None,
        )

    def load_columns(self, columns, dates, sids, mask):
        return merge(
            self.get_loader(column).load_columns(
                [column], dates, sids, mask
            )
            for column in columns
        )


def next_earnings_date_frame(dates, announcement_dates):
    """
    Make a DataFrame representing simulated next earnings dates.

    Parameters
    ----------
    dates : pd.DatetimeIndex.
        The index of the returned DataFrame.
    announcement_dates : dict[int -> pd.Series]
        Dict mapping sids to an index of dates on which earnings were announced
        for that sid.

    Returns
    -------
    next_earnings: pd.DataFrame
        A DataFrame representing, for each (label, date) pair, the first entry
        in `earnings_calendars[label]` on or after `date`.  Entries falling
        after the last date in a calendar will have `np_NaT` as the result in
        the output.

    See Also
    --------
    previous_earnings_date_frame
    """
    cols = {equity: full_like(dates, np_NaT) for equity in announcement_dates}
    raw_dates = dates.values
    for equity, earnings_dates in iteritems(announcement_dates):
        data = cols[equity]
        if not earnings_dates.index.is_monotonic_increasing:
            earnings_dates = earnings_dates.sort_index()

        # Iterate over the raw Series values, since we're comparing against
        # numpy arrays anyway.
        iterkv = zip(earnings_dates.index.values, earnings_dates.values)
        for timestamp, announce_date in iterkv:
            date_mask = (timestamp <= raw_dates) & (raw_dates <= announce_date)
            value_mask = (announce_date <= data) | (data == np_NaT)
            data[date_mask & value_mask] = announce_date

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
    sids = list(announcement_dates)
    out = full((len(dates), len(sids)), np_NaT, dtype='datetime64[ns]')
    dn = dates[-1].asm8
    for col_idx, sid in enumerate(sids):
        # announcement_dates[sid] is Series mapping knowledge_date to actual
        # announcement date.  We don't care about the knowledge date for
        # computing previous earnings.
        values = announcement_dates[sid].values
        values = values[values <= dn]
        out[dates.searchsorted(values), col_idx] = values

    frame = pd.DataFrame(out, index=dates, columns=sids)
    frame.ffill(inplace=True)
    return frame
