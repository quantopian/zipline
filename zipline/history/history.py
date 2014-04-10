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

    Currently only one frequency is supported, "1d"
    "1d" provides data keyed by closing, and the last minute of the current
    day.
    """

    def __init__(self, freq_str):
        # The string the at the algoscript specifies.
        # Hold onto to use a key for caching.
        self.freq_str = freq_str
        # num - The number of units of the frequency.
        # unit_str - The unit type, e.g. 'd'
        self.num, self.unit_str = parse_freq_str(freq_str)


class HistorySpec(object):
    """
    Maps to the parameters of the history() call made by the algoscript

    An object is used here so that get_history calls are not constantly
    parsing the parameters and provides values for caching and indexing into
    result frames.
    """

    @classmethod
    def spec_key(cls, bar_count, freq_str, field, ffill):
        """
        Used as a hash/key value for the HistorySpec.
        """
        return "{0}:{1}:{2}:{3}".format(
            bar_count, freq_str, field, ffill)

    def __init__(self, bar_count, frequency, field, ffill):
        # Number of bars to look back.
        self.bar_count = bar_count
        if isinstance(frequency, str):
            frequency = Frequency(frequency)
        # The frequency at which the data is sampled.
        self.frequency = frequency
        # The field, e.g. 'price', 'volume', etc.
        self.field = field
        # Whether or not to forward fill the nan data.
        self.ffill = ffill

        # How many trading days the spec needs to look back.
        # Used by index creation to see how large of an overarching window
        # is needed.
        self.days_needed = calculate_days_needed(
            self.bar_count, self.frequency)

        # Calculate the cache key string once.
        self.key_str = self.spec_key(
            bar_count, frequency.freq_str, field, ffill)


def calculate_days_needed(bar_count, freq):
    """ Returns number trading days needed.
    Overshoots so that we more than enough to sample from the current
    frequency slot plus previous ones.
    """
    if freq.unit_str == 'd':
        return bar_count * freq.num


def days_index_at_dt(days_needed, algo_dt):
    """
    The timestamps of previous days closes with the size of @days_needed
    at @algo_dt.
    """
    env = trading.environment

    latest_algo_dt = algo_dt

    current_index = env.open_and_closes.index.searchsorted(algo_dt.date())

    previous_days_num = days_needed - 1

    previous_days = env.open_and_closes['market_close'][
        current_index - previous_days_num:current_index]

    # Using the 'rawer' numpy array values here because of a bottleneck
    # that appeared when using DatetimeIndex
    return np.append(previous_days.values, latest_algo_dt)


def index_at_dt(history_spec, algo_dt):
    """
    The index, including @algo_dt at the given @algo_dt for the count
    and frequency of the @history_spec.
    """
    days_index = days_index_at_dt(history_spec.days_needed, algo_dt)

    frequency = history_spec.frequency

    if frequency.unit_str == 'd':

        index_of_algo_dt = days_index.searchsorted(algo_dt)

        start_index = index_of_algo_dt + 1 - history_spec.bar_count
        end_index = index_of_algo_dt + 1

        return days_index[start_index:end_index]
