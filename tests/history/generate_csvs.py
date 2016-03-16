#
# Copyright 2015 Quantopian, Inc.
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

import random
import numpy as np
import pandas as pd

from zipline.finance.trading import TradingEnvironment
from zipline.data.us_equity_minutes import BcolzMinuteBarWriter


def generate_daily_test_data(first_day,
                             last_day,
                             starting_open,
                             starting_volume,
                             multipliers_list,
                             path):

    days = TradingEnvironment.instance().days_in_range(first_day, last_day)

    days_count = len(days)
    o = np.zeros(days_count, dtype=np.uint32)
    h = np.zeros(days_count, dtype=np.uint32)
    l = np.zeros(days_count, dtype=np.uint32)
    c = np.zeros(days_count, dtype=np.uint32)
    v = np.zeros(days_count, dtype=np.uint32)

    last_open = starting_open * 1000
    last_volume = starting_volume

    for idx in range(days_count):
        new_open = last_open + round((random.random() * 5), 2)

        o[idx] = new_open
        h[idx] = new_open + round((random.random() * 10000), 2)
        l[idx] = new_open - round((random.random() * 10000),  2)
        c[idx] = (h[idx] + l[idx]) / 2
        v[idx] = int(last_volume + (random.randrange(-10, 10) * 1e4))

        last_open = o[idx]
        last_volume = v[idx]

    # now deal with multipliers
    if len(multipliers_list) > 0:
        range_start = 0

        for multiplier_info in multipliers_list:
            range_end = days.searchsorted(multiplier_info[0])

            # dividing by the multiplier because we're going backwards
            # and generating the original data that will then be adjusted.
            o[range_start:range_end] /= multiplier_info[1]
            h[range_start:range_end] /= multiplier_info[1]
            l[range_start:range_end] /= multiplier_info[1]
            c[range_start:range_end] /= multiplier_info[1]
            v[range_start:range_end] *= multiplier_info[1]

            range_start = range_end

    df = pd.DataFrame({
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v
    }, columns=[
        "open",
        "high",
        "low",
        "close",
        "volume"
    ], index=days)

    df.to_csv(path, index_label="day")


def generate_minute_test_data(first_day,
                              last_day,
                              starting_open,
                              starting_volume,
                              multipliers_list,
                              path):
    """
    Utility method to generate fake minute-level CSV data.
    :param first_day: first trading day
    :param last_day: last trading day
    :param starting_open: first open value, raw value.
    :param starting_volume: first volume value, raw value.
    :param multipliers_list: ordered list of pd.Timestamp -> float, one per day
            in the range
    :param path: path to save the CSV
    :return: None
    """

    full_minutes = BcolzMinuteBarWriter.full_minutes_for_days(
        first_day, last_day)
    minutes_count = len(full_minutes)

    minutes = TradingEnvironment.instance().minutes_for_days_in_range(
        first_day, last_day)

    o = np.zeros(minutes_count, dtype=np.uint32)
    h = np.zeros(minutes_count, dtype=np.uint32)
    l = np.zeros(minutes_count, dtype=np.uint32)
    c = np.zeros(minutes_count, dtype=np.uint32)
    v = np.zeros(minutes_count, dtype=np.uint32)

    last_open = starting_open * 1000
    last_volume = starting_volume

    for minute in minutes:
        # ugly, but works
        idx = full_minutes.searchsorted(minute)

        new_open = last_open + round((random.random() * 5), 2)

        o[idx] = new_open
        h[idx] = new_open + round((random.random() * 10000), 2)
        l[idx] = new_open - round((random.random() * 10000),  2)
        c[idx] = (h[idx] + l[idx]) / 2
        v[idx] = int(last_volume + (random.randrange(-10, 10) * 1e4))

        last_open = o[idx]
        last_volume = v[idx]

    # now deal with multipliers
    if len(multipliers_list) > 0:
        for idx, multiplier_info in enumerate(multipliers_list):
            start_idx = idx * 390
            end_idx = start_idx + 390

            # dividing by the multipler because we're going backwards
            # and generating the original data that will then be adjusted.
            o[start_idx:end_idx] /= multiplier_info[1]
            h[start_idx:end_idx] /= multiplier_info[1]
            l[start_idx:end_idx] /= multiplier_info[1]
            c[start_idx:end_idx] /= multiplier_info[1]
            v[start_idx:end_idx] *= multiplier_info[1]

    df = pd.DataFrame({
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v
    }, columns=[
        "open",
        "high",
        "low",
        "close",
        "volume"
    ], index=minutes)

    df.to_csv(path, index_label="minute")
