#
# Copyright 2012 Quantopian, Inc.
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


import os
from os.path import expanduser

import msgpack

from treasuries import get_treasury_data
from benchmarks import get_benchmark_returns

# TODO: Make this path customizable.
DATA_PATH = os.path.join(
    expanduser("~"),
    '.zipline',
    'data'
)


def get_datafile(name, mode='r'):
    """
    Returns a handle to data file.

    Creates containing directory, if needed.
    """

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    return open(os.path.join(DATA_PATH, name), mode)


def dump_treasury_curves():
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    tr_data = []

    for curve in get_treasury_data():
        date_as_tuple = curve['date'].timetuple()[0:6] + \
            (curve['date'].microsecond,)
        # Not ideal but massaging data into expected format
        del curve['date']
        tr = (date_as_tuple, curve)
        tr_data.append(tr)

    with get_datafile('treasury_curves.msgpack', mode='wb') as tr_fp:
        tr_fp.write(msgpack.dumps(tr_data))


def dump_benchmarks():
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    benchmark_data = []
    for daily_return in get_benchmark_returns():
        date_as_tuple = daily_return.date.timetuple()[0:6] + \
            (daily_return.date.microsecond,)
        # Not ideal but massaging data into expected format
        benchmark = (date_as_tuple, daily_return.returns)
        benchmark_data.append(benchmark)

    with get_datafile('benchmark.msgpack', mode='wb') as bmark_fp:
        bmark_fp.write(msgpack.dumps(benchmark_data))
