#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import bisect

from zipline.errors import NoFurtherDataError


def normalize_date(date):
    date = pd.Timestamp(date, tz='UTC')
    return pd.tseries.tools.normalize_date(date)


def delta_from_time(t):
    """
    Convert a datetime.time into a timedelta.
    """
    return pd.Timedelta(
        hours=t.hour,
        minutes=t.minute,
        seconds=t.second,
    )


def _get_index(dt, all_trading_days):
    """
    Return the index of the given @dt, or the index of the preceding
    trading day if the given dt is not in the trading calendar.
    """
    ndt = normalize_date(dt)
    if ndt in all_trading_days:
        return all_trading_days.searchsorted(ndt)
    else:
        return all_trading_days.searchsorted(ndt) - 1

# The following methods are intended to be inserted in both the
# ExchangeCalendar and TradingSchedule classes with partial hooks to those
# class' methods. These methods live in the helpers module to avoid code
# duplication.


def next_scheduled_day(date, last_trading_day, is_scheduled_day_hook):
    dt = normalize_date(date)
    delta = pd.Timedelta(days=1)

    while dt <= last_trading_day:
        dt += delta
        if is_scheduled_day_hook(dt):
            return dt
    return None


def previous_scheduled_day(date, first_trading_day, is_scheduled_day_hook):
    dt = normalize_date(date)
    delta = pd.Timedelta(days=-1)

    while first_trading_day < dt:
        dt += delta
        if is_scheduled_day_hook(dt):
            return dt
    return None


def next_open_and_close(date, open_and_close_hook,
                        next_scheduled_day_hook):
    return open_and_close_hook(next_scheduled_day_hook(date))


def previous_open_and_close(date, open_and_close_hook,
                            previous_scheduled_day_hook):
    return open_and_close_hook(previous_scheduled_day_hook(date))


def scheduled_day_distance(first_date, second_date, all_days):
    first_date = normalize_date(first_date)
    second_date = normalize_date(second_date)

    i = bisect.bisect_left(all_days, first_date)
    if i == len(all_days):  # nothing found
        return None
    j = bisect.bisect_left(all_days, second_date)
    if j == len(all_days):
        return None
    distance = j - 1
    assert distance >= 0
    return distance


def minutes_for_day(day, open_and_close_hook):
    start, end = open_and_close_hook(day)
    return pd.date_range(start, end, freq='T')


def days_in_range(start, end, all_days):
    """
    Get all execution days between start and end,
    inclusive.
    """
    start_date = normalize_date(start)
    end_date = normalize_date(end)

    mask = ((all_days >= start_date) & (all_days <= end_date))
    return all_days[mask]


def minutes_for_days_in_range(start, end, days_in_range_hook,
                              minutes_for_day_hook):
    """
    Get all execution minutes for the days between start and end,
    inclusive.
    """
    start_date = normalize_date(start)
    end_date = normalize_date(end)

    all_minutes = []
    for day in days_in_range_hook(start_date, end_date):
        day_minutes = minutes_for_day_hook(day)
        all_minutes.append(day_minutes)

    # Concatenate all minutes and truncate minutes before start/after end.
    return pd.DatetimeIndex(np.concatenate(all_minutes), copy=False, tz='UTC')


def add_scheduled_days(n, date, next_scheduled_day_hook,
                       previous_scheduled_day_hook, all_trading_days):
    """
    Adds n trading days to date. If this would fall outside of the
    trading calendar, a NoFurtherDataError is raised.

    :Arguments:
        n : int
            The number of days to add to date, this can be positive or
            negative.
        date : datetime
            The date to add to.

    :Returns:
        new_date : datetime
            n trading days added to date.
    """
    if n == 1:
        return next_scheduled_day_hook(date)
    if n == -1:
        return previous_scheduled_day_hook(date)

    idx = _get_index(date, all_trading_days) + n
    if idx < 0 or idx >= len(all_trading_days):
        raise NoFurtherDataError(
            msg='Cannot add %d days to %s' % (n, date)
        )

    return all_trading_days[idx]


def all_scheduled_minutes(all_days, minutes_for_days_in_range_hook):
    first_day = all_days[0]
    last_day = all_days[-1]
    return minutes_for_days_in_range_hook(first_day, last_day)


def next_scheduled_minute(start, is_scheduled_day_hook, open_and_close_hook,
                          next_open_and_close_hook):
    """
    Get the next market minute after @start. This is either the immediate
    next minute, the open of the same day if @start is before the market
    open on a trading day, or the open of the next market day after @start.
    """
    if is_scheduled_day_hook(start):
        market_open, market_close = open_and_close_hook(start)
        # If start before market open on a trading day, return market open.
        if start < market_open:
            return market_open
        # If start is during trading hours, then get the next minute.
        elif start < market_close:
            return start + pd.Timedelta(minutes=1)
    # If start is not in a trading day, or is after the market close
    # then return the open of the *next* trading day.
    return next_open_and_close_hook(start)[0]


def previous_scheduled_minute(start, is_scheduled_day_hook,
                              open_and_close_hook,
                              previous_open_and_close_hook):
    """
    Get the next market minute before @start. This is either the immediate
    previous minute, the close of the same day if @start is after the close
    on a trading day, or the close of the market day before @start.
    """
    if is_scheduled_day_hook(start):
        market_open, market_close = open_and_close_hook(start)
        # If start after the market close, return market close.
        if start > market_close:
            return market_close
        # If start is during trading hours, then get previous minute.
        if start > market_open:
            return start - pd.Timedelta(minutes=1)
    # If start is not a trading day, or is before the market open
    # then return the close of the *previous* trading day.
    return previous_open_and_close_hook(start)[1]
