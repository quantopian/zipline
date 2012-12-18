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
from operator import attrgetter
from collections import OrderedDict

import pandas as pd
from pandas.io.data import DataReader
import numpy as np
from datetime import datetime, timedelta

import zipline.finance.risk as risk
from zipline.utils.date_utils import tuple_to_date
from zipline.utils.protocol_utils import ndict
from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource)
from zipline.gens.utils import create_trade
from zipline.finance.trading import TradingEnvironment
from zipline.data.loader import (
    get_datafile,
    dump_benchmarks,
    dump_treasury_curves
)


def load_market_data():
    try:
        fp_bm = get_datafile('benchmark.msgpack', "rb")
    except IOError:
        print """
data msgpacks aren't distribute with source.
Fetching data from Yahoo Finance.
""".strip()
        dump_benchmarks()
        fp_bm = get_datafile('benchmark.msgpack', "rb")

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


def create_trading_environment(year=2006, start=None, end=None,
                               capital_base=float("1.0e5")):
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
        capital_base=capital_base
    )

    return trading_environment


def get_next_trading_dt(current, interval, trading_calendar):
    next = current
    while True:
        next = next + interval
        if trading_calendar.is_market_hours(next):
            break

    return next


def create_trade_history(sid, prices, amounts, interval, trading_calendar,
                         source_id="test_factory"):
    trades = []
    current = trading_calendar.first_open

    for price, amount in zip(prices, amounts):
        trade = create_trade(sid, price, amount, current, source_id)
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


def create_test_df_source(trading_calendar=None):
    start = trading_calendar.first_open \
        if trading_calendar else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = trading_calendar.last_close \
        if trading_calendar else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

    index = pd.DatetimeIndex(start=start, end=end, freq=pd.datetools.day)
    x = np.arange(0, len(index))

    df = pd.DataFrame(x, index=index, columns=[0])

    return DataFrameSource(df), df


def create_test_panel_source(trading_calendar=None):
    start = trading_calendar.first_open \
        if trading_calendar else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = trading_calendar.last_close \
        if trading_calendar else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

    index = pd.DatetimeIndex(start=start, end=end, freq=pd.datetools.day)
    price = np.arange(0, len(index))
    volume = np.ones(len(index)) * 1000
    arbitrary = np.ones(len(index))

    df = pd.DataFrame({'price': price,
                       'volume': volume,
                       'arbitrary': arbitrary},
                      index=index)
    panel = pd.Panel.from_dict({0: df})

    return DataPanelSource(panel), panel


def load_from_yahoo(indexes=None, stocks=None, start=None, end=None):
    """Load closing prices from yahoo finance.

    :Optional:
        indexes : dict (Default: {'SPX': '^GSPC'})
            Financial indexes to load.
        stocks : list (Default: ['AAPL', 'GE', 'IBM', 'MSFT',
                                 'XOM', 'AA', 'JNJ', 'PEP', 'KO'])
            Stock closing prices to load.
        start : datetime (Default: datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc))
            Retrieve prices from start date on.
        end : datetime (Default: datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc))
            Retrieve prices until end date.

    :Note:
        This is based on code presented in a talk by Wes McKinney:
        http://wesmckinney.com/files/20111017/notebook_output.pdf
    """

    if indexes is None:
        indexes = {'SPX': '^GSPC'}
    if stocks is None:
        stocks = ['AAPL', 'GE', 'IBM', 'MSFT', 'XOM', 'AA', 'JNJ', 'PEP', 'KO']
    if start is None:
        start = pd.datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc)
    if end is None:
        end = pd.datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)

    assert start < end, "start date is later than end date."

    data = OrderedDict()

    for stock in stocks:
        print stock
        stkd = DataReader(stock, 'yahoo', start, end).sort_index()
        data[stock] = stkd

    for name, ticker in indexes.iteritems():
        print name
        stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
        data[name] = stkd

    df = pd.DataFrame({key: d['Close'] for key, d in data.iteritems()})
    df.index = df.index.tz_localize(pytz.utc)

    return df
