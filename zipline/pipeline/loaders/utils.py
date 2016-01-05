import numpy as np
import pandas as pd
from six import iteritems
from six.moves import zip

from zipline.utils.numpy_utils import np_NaT


def next_date_frame(dates, announcement_dates):
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
    cols = {
        equity: np.full_like(dates, np_NaT) for equity in announcement_dates
    }
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


def previous_date_frame(dates, announcement_dates):
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
    out = np.full((len(dates), len(sids)), np_NaT, dtype='datetime64[ns]')
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
