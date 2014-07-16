#
# Copyright 2014 Quantopian, Inc.
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

from __future__ import division

import numpy as np
import pandas as pd
import re

from zipline.finance import trading


def parse_freq_str(freq_str):
    # TODO: Wish we were more aligned with pandas here.
    num_str, unit_str = re.match('([0-9]+)([A-Za-z]+)', freq_str).groups()
    return int(num_str), unit_str


class Frequency(object):
    """
    Represents how the data is sampled, as specified by the algoscript
    via units like "1d", "1m", etc.

    Currently only two frequencies are supported, "1d" and "1m"

    - "1d" provides data at daily frequency, with the latest bar aggregating
    the elapsed minutes of the (incomplete) current day
    - "1m" provides data at minute frequency
    """
    SUPPORTED_FREQUENCIES = frozenset({'1d', '1m'})
    MAX_MINUTES = {'m': 1, 'd': 390}

    def __init__(self, freq_str, daily_at_midnight=False):

        if freq_str not in self.SUPPORTED_FREQUENCIES:
            raise ValueError(
                "history frequency must be in {supported}".format(
                    supported=self.SUPPORTED_FREQUENCIES,
                ))
        # The string the at the algoscript specifies.
        # Hold onto to use a key for caching.
        self.freq_str = freq_str

        # num - The number of units of the frequency.
        # unit_str - The unit type, e.g. 'd'
        self.num, self.unit_str = parse_freq_str(freq_str)

        self.daily_at_midnight = daily_at_midnight

    def next_window_start(self, previous_window_close):
        """
        Get the first minute of the window starting after a window that
        finished on @previous_window_close.
        """
        if self.unit_str == 'd':
            return self.next_day_window_start(previous_window_close,
                                              self.daily_at_midnight)
        elif self.unit_str == 'm':
            return self.next_minute_window_start(previous_window_close)

    @staticmethod
    def next_day_window_start(previous_window_close, daily_at_midnight=False):
        """
        Get the next day window start after @previous_window_close.  This is
        defined as the first market open strictly greater than
        @previous_window_close.
        """
        env = trading.environment
        if daily_at_midnight:
            next_open = env.next_trading_day(previous_window_close)
        else:
            next_open, _ = env.next_open_and_close(previous_window_close)
        return next_open

    @staticmethod
    def next_minute_window_start(previous_window_close):
        """
        Get the next minute window start after @previous_window_close.  This is
        defined as the first market minute strictly greater than
        @previous_window_close.
        """
        env = trading.environment
        return env.next_market_minute(previous_window_close)

    def window_open(self, window_close):
        """
        For a period ending on `window_end`, calculate the date of the first
        minute bar that should be used to roll a digest for this frequency.
        """
        if self.unit_str == 'd':
            return self.day_window_open(window_close, self.num)
        elif self.unit_str == 'm':
            return self.minute_window_open(window_close, self.num)

    def window_close(self, window_start):
        """
        For a period starting on `window_start`, calculate the date of the last
        minute bar that should be used to roll a digest for this frequency.
        """
        if self.unit_str == 'd':
            return self.day_window_close(window_start, self.num)
        elif self.unit_str == 'm':
            return self.minute_window_close(window_start, self.num)

    def day_window_open(self, window_close, num_days):
        """
        Get the first minute for a daily window of length @num_days with last
        minute @window_close.  This is calculated by searching backward until
        @num_days market_closes are encountered.
        """
        env = trading.environment
        open_ = env.open_close_window(
            window_close,
            1,
            offset=-(num_days - 1)
        ).market_open.iloc[0]

        if self.daily_at_midnight:
            open_ = pd.tslib.normalize_date(open_)

        return open_

    def minute_window_open(self, window_close, num_minutes):
        """
        Get the first minute for a minutely window of length @num_minutes with
        last minute @window_close.

        This is defined as window_close if num_minutes == 1, and otherwise as
        the N-1st market minute after @window_start.
        """
        if num_minutes == 1:
            # Short circuit this case.
            return window_close

        env = trading.environment
        return env.market_minute_window(window_close, count=-num_minutes)[-1]

    def day_window_close(self, window_start, num_days):
        """
        Get the last minute for a daily window of length @num_days with first
        minute @window_start.  This is calculated by searching forward until
        @num_days market closes are encountered.

        Examples:

        window_start = Thursday March 2nd, 2006, 9:31 AM EST
        num_days = 1
          --> window_close = Thursday March 2nd, 2006, 4:00 PM EST

        window_start = Thursday March 2nd, 2006, 3:59 AM EST
        num_days = 1
          --> window_close = Thursday March 2nd, 2006, 4:00 PM EST

        window_start = Thursday March 2nd, 2006, 9:31 AM EST
        num_days = 2
          --> window_close = Friday March 2nd, 2006, 4:00 PM EST

        window_start = Thursday March 2nd, 2006, 9:31 AM EST
        num_days = 3
          --> window_close = Monday March 6th, 2006, 4:00 PM EST

        # Day before July 4th is an early close
        window_start = Wednesday July 3rd, 2013, 9:31 AM EST
        num_days = 1
          --> window_close = Wednesday July 3rd, 2013, 1:00 PM EST
        """
        env = trading.environment
        close = env.open_close_window(
            window_start,
            1,
            offset=num_days - 1
        ).market_close.iloc[0]

        if self.daily_at_midnight:
            close = pd.tslib.normalize_date(close)

        return close

    def minute_window_close(self, window_start, num_minutes):
        """
        Get the last minute for a minutely window of length @num_minutes with
        first minute @window_start.

        This is defined as window_start if num_minutes == 1, and otherwise as
        the N-1st market minute after @window_start.
        """
        if num_minutes == 1:
            # Short circuit this case.
            return window_start

        env = trading.environment
        return env.market_minute_window(window_start, count=num_minutes)[-1]

    @property
    def max_minutes(self):
        """
        The maximum number of minutes required to roll a bar at this frequency.
        """
        return self.MAX_MINUTES[self.unit_str] * self.num

    def __eq__(self, other):
        return self.freq_str == other.freq_str

    def __hash__(self):
        return hash(self.freq_str)

    def __repr__(self):
        return ''.join([str(self.__class__.__name__),
                        "('", self.freq_str, "')"])


class HistorySpec(object):
    """
    Maps to the parameters of the history() call made by the algoscript

    An object is used here so that get_history calls are not constantly
    parsing the parameters and provides values for caching and indexing into
    result frames.
    """

    FORWARD_FILLABLE = frozenset({'price'})

    @classmethod
    def spec_key(cls, bar_count, freq_str, field, ffill):
        """
        Used as a hash/key value for the HistorySpec.
        """
        return "{0}:{1}:{2}:{3}".format(
            bar_count, freq_str, field, ffill)

    def __init__(self, bar_count, frequency, field, ffill,
                 daily_at_midnight=False):
        # Number of bars to look back.
        self.bar_count = bar_count
        if isinstance(frequency, str):
            frequency = Frequency(frequency, daily_at_midnight)
        # The frequency at which the data is sampled.
        self.frequency = frequency
        # The field, e.g. 'price', 'volume', etc.
        self.field = field
        # Whether or not to forward fill the nan data.
        self._ffill = ffill

        # Calculate the cache key string once.
        self.key_str = self.spec_key(
            bar_count, frequency.freq_str, field, ffill)

    @property
    def ffill(self):
        """
        Wrapper around self._ffill that returns False for fields which are not
        forward-fillable.
        """
        return self._ffill and self.field in self.FORWARD_FILLABLE

    def __repr__(self):
        return ''.join([self.__class__.__name__, "('", self.key_str, "')"])


def days_index_at_dt(history_spec, algo_dt):
    """
    Get the index of a frame to be used for a get_history call with daily
    frequency.
    """
    env = trading.environment
    # Get the previous (bar_count - 1) days' worth of market closes.
    day_delta = (history_spec.bar_count - 1) * history_spec.frequency.num
    market_closes = env.open_close_window(
        algo_dt,
        day_delta,
        offset=(-day_delta),
        step=history_spec.frequency.num,
    ).market_close

    if history_spec.frequency.daily_at_midnight:
        market_closes = market_closes.apply(pd.tslib.normalize_date)

    # Append the current algo_dt as the last index value.
    # Using the 'rawer' numpy array values here because of a bottleneck
    # that appeared when using DatetimeIndex
    return np.append(market_closes.values, algo_dt)


def minutes_index_at_dt(history_spec, algo_dt):
    """
    Get the index of a frame to be used for a get_history_call with minutely
    frequency.
    """
    # TODO: This is almost certainly going to be too slow for production.
    env = trading.environment
    return env.market_minute_window(
        algo_dt,
        history_spec.bar_count,
        step=-1,
    )[::-1]


def index_at_dt(history_spec, algo_dt):
    """
    Returns index of a frame returned by get_history() with the given
    history_spec and algo_dt.

    The resulting index `@history_spec.bar_count` bars, increasing in units of
    `@history_spec.frequency`, terminating at the given @algo_dt.

    Note: The last bar of the returned frame represents an as-of-yet incomplete
    time window, so the delta between the last and second-to-last bars is
    usually always less than `@history_spec.frequency` for frequencies greater
    than 1m.
    """
    frequency = history_spec.frequency
    if frequency.unit_str == 'd':
        return days_index_at_dt(history_spec, algo_dt)
    elif frequency.unit_str == 'm':
        return minutes_index_at_dt(history_spec, algo_dt)
