#
# Copyright 2013 Quantopian, Inc.
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
import collections

from datetime import datetime

import csv

from functools import partial

import requests

from loader_utils import (
    date_conversion,
    source_to_records,
    Mapping
)
from zipline.protocol import DailyReturn


class BenchmarkDataNotFoundError(Exception):
    pass

_BENCHMARK_MAPPING = {
    # Need to add 'symbol'
    'volume': (int, 'Volume'),
    'open': (float, 'Open'),
    'close': (float, 'Close'),
    'high': (float, 'High'),
    'low': (float, 'Low'),
    'adj_close': (float, 'Adj Close'),
    'date': (partial(date_conversion, date_pattern='%Y-%m-%d'), 'Date')
}


def benchmark_mappings():
    return {key: Mapping(*value)
            for key, value
            in _BENCHMARK_MAPPING.iteritems()}


def get_raw_benchmark_data(start_date, end_date, symbol):

    # create benchmark files
    # ^GSPC 19500103
    params = collections.OrderedDict((
        ('s', symbol),
        # start_date month, zero indexed
        ('a', start_date.month - 1),
        # start_date day
        ('b', start_date.day),
        # start_date year
        ('c', start_date.year),
        # end_date month, zero indexed
        ('d', end_date.month - 1),
        # end_date day str(int(todate[6:8])) #day
        ('e', end_date.day),
        # end_date year str(int(todate[0:4]))
        ('f', end_date.year),
        # daily frequency
        ('g', 'd'),
    ))

    res = requests.get('http://ichart.yahoo.com/table.csv',
                       params=params, stream=True)

    if not res.ok:
        raise BenchmarkDataNotFoundError("""
No benchmark data found for date range.
start_date={start_date}, end_date={end_date}, url={url}""".strip().
                                         format(start_date=start_date,
                                                end_date=end_date,
                                                url=res.url))

    return csv.DictReader(res.iter_lines())


def get_benchmark_data(symbol, start_date=None, end_date=None):
    """
    Benchmarks from Yahoo.
    """
    if start_date is None:
        start_date = datetime(year=1950, month=1, day=3)
    if end_date is None:
        end_date = datetime.utcnow()

    raw_benchmark_data = get_raw_benchmark_data(start_date, end_date, symbol)

    mappings = benchmark_mappings()

    return source_to_records(mappings, raw_benchmark_data)


def get_benchmark_returns(symbol, start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime(year=1950, month=1, day=3)
    if end_date is None:
        end_date = datetime.utcnow()

    benchmark_returns = []

    for data_point in get_benchmark_data(symbol, start_date, end_date):
        returns = (data_point['close'] - data_point['open']) / \
            data_point['open']
        daily_return = DailyReturn(date=data_point['date'], returns=returns)
        benchmark_returns.append(daily_return)

    # Reverse data so we can load it in reverse chron order.
    benchmark_returns.reverse()
    return benchmark_returns
