from toolz import partition_all


def roll_dates_to_previous_session(sessions, *dates):
    """
    Roll `dates` to the last session of `calendar`. Return input date if it
    is a valid session.

    Parameters
    ----------
    sessions : pandas.tseries.index.DatetimeIndex
        The list of valid session dates.
    *dates : pd.Timestamp
        The dates for which the last trading date is needed.

    Returns
    -------
    rolled_dates: pandas.tseries.index.DatetimeIndex
        The last trading date of the input dates, inclusive.

    """
    # Find the previous index value if there is no exact match.
    locs = [sessions.get_loc(dt, method='ffill') for dt in dates]
    return sessions[locs].tolist()


def compute_date_range_chunks(sessions, start_date, end_date, chunksize):
    """Compute the start and end dates to run a pipeline for.

    Parameters
    ----------
    sessions : DatetimeIndex
        The available dates.
    start_date : pd.Timestamp
        The first date in the pipeline.
    end_date : pd.Timestamp
        The last date in the pipeline.
    chunksize : int or None
        The size of the chunks to run. Setting this to None returns one chunk.

    Returns
    -------
    ranges : iterable[(np.datetime64, np.datetime64)]
        A sequence of start and end dates to run the pipeline for.
    """
    if start_date not in sessions:
        raise KeyError("Start date %s is not found in calendar." %
                       (start_date.strftime("%Y-%m-%d"),))
    if end_date not in sessions:
        raise KeyError("End date %s is not found in calendar." %
                       (end_date.strftime("%Y-%m-%d"),))
    if end_date < start_date:
        raise ValueError("End date %s cannot precede start date %s." %
                         (end_date.strftime("%Y-%m-%d"),
                          start_date.strftime("%Y-%m-%d")))

    if chunksize is None:
        return [(start_date, end_date)]

    start_ix, end_ix = sessions.slice_locs(start_date, end_date)
    return (
        (r[0], r[-1]) for r in partition_all(
            chunksize, sessions[start_ix:end_ix]
        )
    )
