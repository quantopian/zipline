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
from collections import OrderedDict


from treasuries import get_treasury_data
from benchmarks import get_benchmark_returns

from zipline.utils.date_utils import tuple_to_date
from operator import attrgetter


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


def dump_benchmarks(symbol):
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    benchmark_data = []
    for daily_return in get_benchmark_returns(symbol):
        date_as_tuple = daily_return.date.timetuple()[0:6] + \
            (daily_return.date.microsecond,)
        # Not ideal but massaging data into expected format
        benchmark = (date_as_tuple, daily_return.returns)
        benchmark_data.append(benchmark)

    with get_datafile(get_benchmark_filename(symbol), mode='wb') as bmark_fp:
        bmark_fp.write(msgpack.dumps(benchmark_data))


def get_benchmark_filename(symbol):
    return "%s_benchmark.msgpack" % symbol


def load_market_data(bm_symbol='^GSPC'):
    from zipline.finance.trading import DailyReturn

    try:
        fp_bm = get_datafile(get_benchmark_filename(bm_symbol), "rb")
    except IOError:
        print """
data msgpacks aren't distribute with source.
Fetching data from Yahoo Finance.
""".strip()
        dump_benchmarks(bm_symbol)
        fp_bm = get_datafile(get_benchmark_filename(bm_symbol), "rb")

    bm_list = msgpack.loads(fp_bm.read())
    bm_returns = []
    for packed_date, returns in bm_list:
        event_dt = tuple_to_date(packed_date)

        daily_return = DailyReturn(date=event_dt, returns=returns)
        bm_returns.append(daily_return)

    fp_bm.close()

    bm_returns = sorted(bm_returns, key=attrgetter('date'))

    try:
        fp_tr = get_datafile('treasury_curves.msgpack', "rb")
    except IOError:
        print """
data msgpacks aren't distribute with source.
Fetching data from data.treasury.gov
""".strip()
        dump_treasury_curves()
        fp_tr = get_datafile('treasury_curves.msgpack', "rb")

    tr_list = msgpack.loads(fp_tr.read())
    tr_curves = {}
    for packed_date, curve in tr_list:
        tr_dt = tuple_to_date(packed_date)
        #tr_dt = tr_dt.replace(hour=0, minute=0, second=0, tzinfo=pytz.utc)
        tr_curves[tr_dt] = curve

    fp_tr.close()

    tr_curves = OrderedDict(sorted(
                            ((dt, c) for dt, c in tr_curves.iteritems()),
                            key=lambda t: t[0]))

    return bm_returns, tr_curves
