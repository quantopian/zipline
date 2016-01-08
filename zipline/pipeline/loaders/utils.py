import numpy as np
import pandas as pd
from six import iteritems
from six.moves import zip

from zipline.utils.numpy_utils import np_NaT


def next_date_frame(dates, events_by_sid):
    """
    Make a DataFrame representing the simulated next known date for an event.

    Parameters
    ----------
    dates : pd.DatetimeIndex.
        The index of the returned DataFrame.
    events_by_sid : dict[int -> pd.Series]
        Dict mapping sids to a series of dates. Each k:v pair of the series
        represents the date we learned of the event mapping to the date the
        event will occur.
    Returns
    -------
    next_events: pd.DataFrame
        A DataFrame where each column is a security from `events_by_sid` where
        the values are the dates of the next known event with the knowledge we
        had on the date of the index. Entries falling after the last date will
        have `NaT` as the result in the output.


    See Also
    --------
    previous_date_frame
    """
    cols = {
        equity: np.full_like(dates, np_NaT) for equity in events_by_sid
    }
    raw_dates = dates.values
    for equity, event_dates in iteritems(events_by_sid):
        data = cols[equity]
        if not event_dates.index.is_monotonic_increasing:
            event_dates = event_dates.sort_index()

        # Iterate over the raw Series values, since we're comparing against
        # numpy arrays anyway.
        iterkv = zip(event_dates.index.values, event_dates.values)
        for knowledge_date, event_date in iterkv:
            date_mask = (
                (knowledge_date <= raw_dates) &
                (raw_dates <= event_date)
            )
            value_mask = (event_date <= data) | (data == np_NaT)
            data[date_mask & value_mask] = event_date

    return pd.DataFrame(index=dates, data=cols)


def previous_date_frame(date_index, events_by_sid):
    """
    Make a DataFrame representing simulated next earnings date_index.

    Parameters
    ----------
    date_index : DatetimeIndex.
        The index of the returned DataFrame.
    events_by_sid : dict[int -> DatetimeIndex]
        Dict mapping sids to a series of dates. Each k:v pair of the series
        represents the date we learned of the event mapping to the date the
        event will occur.

    Returns
    -------
    previous_events: pd.DataFrame
        A DataFrame where each column is a security from `events_by_sid` where
        the values are the dates of the previous event that occured on the date
        of the index. Entries falling before the first date will have `NaT` as
        the result in the output.

    See Also
    --------
    next_date_frame
    """
    sids = list(events_by_sid)
    out = np.full((len(date_index), len(sids)), np_NaT, dtype='datetime64[ns]')
    dn = date_index[-1].asm8
    for col_idx, sid in enumerate(sids):
        # events_by_sid[sid] is Series mapping knowledge_date to actual
        # event_date.  We don't care about the knowledge date for
        # computing previous earnings.
        values = events_by_sid[sid].values
        values = values[values <= dn]
        out[date_index.searchsorted(values), col_idx] = values

    frame = pd.DataFrame(out, index=date_index, columns=sids)
    frame.ffill(inplace=True)
    return frame
