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

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath('.'))
    print sys.path

import msgpack

from treasuries import get_treasury_data
from benchmarks import get_benchmark_returns


def dump_treasury_curves():
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    tr_data = []

    for curve in get_treasury_data():
        print curve
        date_as_tuple = curve['date'].timetuple()[0:6] + \
                        (curve['date'].microsecond,)
        # Not ideal but massaging data into expected format
        del curve['date']
        tr = (date_as_tuple, curve)
        tr_data.append(tr)

    tr_path = os.path.join(os.path.dirname(__file__),
                           "treasury_curves.msgpack")
    tr_fp = open(tr_path, "wb")
    tr_fp.write(msgpack.dumps(tr_data))


def dump_benchmarks():
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    benchmark_path = os.path.join(os.path.dirname(__file__),
                           "benchmark.msgpack")
    benchmark_fp = open(benchmark_path, "wb")
    benchmark_data = []
    for daily_return in get_benchmark_returns():
        print daily_return
        date_as_tuple = daily_return.date.timetuple()[0:6] + \
                        (daily_return.date.microsecond,)
        # Not ideal but massaging data into expected format
        benchmark = (date_as_tuple, daily_return.returns)
        benchmark_data.append(benchmark)

    benchmark_fp.write(msgpack.dumps(benchmark_data))
