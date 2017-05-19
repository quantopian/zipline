def roll_dates_to_previous_session(calendar, *dates):
    """
    Roll ``dates`` to the next session of ``calendar``.

    Parameters
    ----------
    calendar : zipline.utils.calendars.trading_calendar.TradingCalendar
        The calendar to use as a reference.
    *dates : pd.Timestamp
        The dates for which the last trading date is needed.

    Returns
    -------
    rolled_dates: pandas.tseries.index.DatetimeIndex
        The last trading date of the input dates, inclusive.

    """
    all_sessions = calendar.all_sessions

    locs = [all_sessions.get_loc(dt, method='ffill') for dt in dates]
    return all_sessions[locs]
