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


"""
Factory functions to prepare useful data for tests.
"""
import pytz
import msgpack
import random
from os.path import join, abspath, dirname
from operator import attrgetter

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import zipline.finance.risk as risk
from zipline.utils.date_utils import tuple_to_date
from zipline.utils.protocol_utils import ndict
from zipline.gens.tradegens import SpecificEquityTrades, DataFrameSource
from zipline.gens.utils import create_trade
from zipline.finance.trading import TradingEnvironment

from zipline import data


# TODO
def data_path():
    data_path = dirname(abspath(data.__file__))
    return data_path


def load_market_data():
    benchmark_data_path = join(data_path(), "benchmark.msgpack")
    try:
        fp_bm = open(benchmark_data_path, "rb")
    except IOError:
        print """
data msgpacks aren't distribute with source.
Fetching data from Yahoo Finance.
""".strip()
        data.loader.dump_benchmarks()
        fp_bm = open(benchmark_data_path, "rb")

    bm_list = msgpack.loads(fp_bm.read())
    bm_returns = []
    for packed_date, returns in bm_list:
        event_dt = tuple_to_date(packed_date)
        #event_dt = event_dt.replace(
        #    hour=0,
        #    minute=0,
        #    second=0,
        #    tzinfo=pytz.utc
        #)

        daily_return = risk.DailyReturn(date=event_dt, returns=returns)
        bm_returns.append(daily_return)

    bm_returns = sorted(bm_returns, key=attrgetter('date'))

    treasury_data_path = join(data_path(), "treasury_curves.msgpack")
    try:
        fp_bm = open(treasury_data_path, "rb")
    except IOError:
        print """
data msgpacks aren't distribute with source.
Fetching data from data.treasury.gov
""".strip()
        data.loader.dump_treasury_curves()
        fp_bm = open(treasury_data_path, "rb")

    fp_tr = open(join(data_path(), "treasury_curves.msgpack"), "rb")
    tr_list = msgpack.loads(fp_tr.read())
    tr_curves = {}
    for packed_date, curve in tr_list:
        tr_dt = tuple_to_date(packed_date)
        #tr_dt = tr_dt.replace(hour=0, minute=0, second=0, tzinfo=pytz.utc)
        tr_curves[tr_dt] = curve

    return bm_returns, tr_curves


def create_trading_environment(year=2006, start=None, end=None):
    """Construct a complete environment with reasonable defaults"""
    benchmark_returns, treasury_curves = load_market_data()

    if start is None:
        start = datetime(year, 1, 1, tzinfo=pytz.utc)
    if end is None:
        end = datetime(year, 12, 31, tzinfo=pytz.utc)

    trading_environment = TradingEnvironment(
        benchmark_returns,
        treasury_curves,
        period_start=start,
        period_end=end,
        capital_base=100000.0
    )

    return trading_environment


def get_next_trading_dt(current, interval, trading_calendar):
    next = current
    while True:
        next = next + interval
        if trading_calendar.is_market_hours(next):
            break

    return next


def create_trade_history(sid, prices, amounts, interval, trading_calendar):
    trades = []
    current = trading_calendar.first_open

    for price, amount in zip(prices, amounts):
        trade = create_trade(sid, price, amount, current)
        trades.append(trade)
        current = get_next_trading_dt(current, interval, trading_calendar)

    assert len(trades) == len(prices)
    return trades


def create_txn(sid, price, amount, datetime):
    txn = ndict({
        'sid': sid,
        'amount': amount,
        'dt': datetime,
        'price': price,
    })
    return txn


def create_txn_history(sid, priceList, amtList, interval, trading_calendar):
    txns = []
    current = trading_calendar.first_open

    for price, amount in zip(priceList, amtList):
        current = get_next_trading_dt(current, interval, trading_calendar)

        txns.append(create_txn(sid, price, amount, current))
        current = current + interval
    return txns


def create_returns(daycount, trading_calendar):
    """
    For the given number of calendar (not trading) days return all the trading
    days between start and start + daycount.
    """
    test_range = []
    current = trading_calendar.first_open
    one_day = timedelta(days=1)

    for day in range(daycount):
        current = current + one_day
        if trading_calendar.is_trading_day(current):
            r = risk.DailyReturn(current, random.random())
            test_range.append(r)

    return test_range


def create_returns_from_range(trading_calendar):
    current = trading_calendar.first_open
    end = trading_calendar.last_close
    one_day = timedelta(days=1)
    test_range = []
    while current <= end:
        r = risk.DailyReturn(current, random.random())
        test_range.append(r)
        current = get_next_trading_dt(current, one_day, trading_calendar)

    return test_range


def create_returns_from_list(returns, trading_calendar):
    current = trading_calendar.first_open
    one_day = timedelta(days=1)
    test_range = []

    #sometimes the range starts with a non-trading day.
    if not trading_calendar.is_trading_day(current):
        current = get_next_trading_dt(current, one_day, trading_calendar)

    for return_val in returns:
        r = risk.DailyReturn(current, return_val)
        test_range.append(r)
        current = get_next_trading_dt(current, one_day, trading_calendar)

    return test_range


def create_daily_trade_source(sids, trade_count, trading_environment,
                              concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on trading_environment.period_start, and daily
    thereafter for each sid. Thus, two sids should result in two trades per
    day.

    Important side-effect: trading_environment.period_end will be modified
    to match the day of the final trade.
    """
    return create_trade_source(
        sids,
        trade_count,
        timedelta(days=1),
        trading_environment,
        concurrent=concurrent
    )


def create_minutely_trade_source(sids, trade_count, trading_environment,
                                 concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on trading_environment.period_start, and every minute
    thereafter for each sid. Thus, two sids should result in two trades per
    minute.

    Important side-effect: trading_environment.period_end will be modified
    to match the day of the final trade.
    """
    return create_trade_source(
        sids,
        trade_count,
        timedelta(minutes=1),
        trading_environment,
        concurrent=concurrent
    )


def create_trade_source(sids, trade_count,
                        trade_time_increment, trading_environment,
                        concurrent=False):

    args = tuple()
    kwargs = {
        'count': trade_count,
        'sids': sids,
        'start': trading_environment.first_open,
        'delta': trade_time_increment,
        'filter': sids,
        'concurrent': concurrent
    }
    source = SpecificEquityTrades(*args, **kwargs)

    # TODO: do we need to set the trading environment's end to same dt as
    # the last trade in the history?
    #trading_environment.period_end = trade_history[-1].dt

    return source


def create_test_df_source():
    start = pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)
    end = pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)
    index = pd.DatetimeIndex(start=start, end=end, freq=pd.datetools.day)
    x = np.arange(2., 14.).reshape((6, 2))
    df = pd.DataFrame(x, index=index, columns=[0, 1])

    return DataFrameSource(df), df
