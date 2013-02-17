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
import random
from collections import OrderedDict

import pandas as pd
from pandas.io.data import DataReader
import numpy as np
from datetime import datetime, timedelta

import zipline.finance.risk as risk
from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource)
from zipline.gens.utils import create_trade
from zipline.finance.trading import SimulationParameters, TradingEnvironment
import zipline.finance.trading as trading


def create_simulation_parameters(year=2006, start=None, end=None,
                                 capital_base=float("1.0e5")
                                 ):
    """Construct a complete environment with reasonable defaults"""
    trading.environment = TradingEnvironment()
    if start is None:
        start = datetime(year, 1, 1, tzinfo=pytz.utc)
    if end is None:
        end = datetime(year, 12, 31, tzinfo=pytz.utc)

    sim_params = SimulationParameters(
        period_start=start,
        period_end=end,
        capital_base=capital_base,
    )

    return sim_params


def create_random_simulation_parameters():
        trading.environment = TradingEnvironment()
        treasury_curves = trading.environment.treasury_curves

        for n in range(100):

            random_index = random.randint(
                0,
                len(treasury_curves)
            )

            start_dt = treasury_curves.keys()[random_index]
            end_dt = start_dt + timedelta(days=365)

            now = datetime.utcnow().replace(tzinfo=pytz.utc)

            if end_dt <= now:
                break

        assert end_dt <= now, """
failed to find a suitable daterange after 100 attempts. please double
check treasury and benchmark data in findb, and re-run the test."""

        sim_params = SimulationParameters(
            period_start=start_dt,
            period_end=end_dt
        )

        return sim_params, start_dt, end_dt


def get_next_trading_dt(current, interval):
    next = current
    while True:
        next = next + interval
        if trading.environment.is_market_hours(next):
            break

    return next


def create_trade_history(sid, prices, amounts, interval, sim_params,
                         source_id="test_factory"):
    trades = []
    current = sim_params.first_open

    for price, amount in zip(prices, amounts):
        trade = create_trade(sid, price, amount, current, source_id)
        trades.append(trade)
        current = get_next_trading_dt(current, interval)

    assert len(trades) == len(prices)
    return trades


def create_dividend(sid, payment, declared_date, ex_date, pay_date):
    div = Event({
        'sid': sid,
        'gross_amount': payment,
        'net_amount': payment,
        'dt': declared_date.replace(hour=0, minute=0, second=0),
        'ex_date': ex_date.replace(hour=0, minute=0, second=0),
        'pay_date': pay_date.replace(hour=0, minute=0, second=0),
        'type': DATASOURCE_TYPE.DIVIDEND
    })

    return div


def create_txn(sid, price, amount, datetime):
    txn = Event({
        'sid': sid,
        'amount': amount,
        'dt': datetime,
        'price': price,
    })
    return txn


def create_txn_history(sid, priceList, amtList, interval, sim_params):
    txns = []
    current = sim_params.first_open

    for price, amount in zip(priceList, amtList):
        current = get_next_trading_dt(current, interval)

        txns.append(create_txn(sid, price, amount, current))
        current = current + interval
    return txns


def create_returns(daycount, sim_params):
    """
    For the given number of calendar (not trading) days return all the trading
    days between start and start + daycount.
    """
    test_range = []
    current = sim_params.first_open
    one_day = timedelta(days=1)

    for day in range(daycount):
        current = current + one_day
        if trading.environment.is_trading_day(current):
            r = risk.DailyReturn(current, random.random())
            test_range.append(r)

    return test_range


def create_returns_from_range(sim_params):
    current = sim_params.first_open
    end = sim_params.last_close
    one_day = timedelta(days=1)
    test_range = []
    while current <= end:
        r = risk.DailyReturn(current, random.random())
        test_range.append(r)
        current = get_next_trading_dt(current, one_day)

    return test_range


def create_returns_from_list(returns, sim_params):
    current = sim_params.first_open
    one_day = timedelta(days=1)
    test_range = []

    #sometimes the range starts with a non-trading day.
    if not trading.environment.is_trading_day(current):
        current = get_next_trading_dt(current, one_day)

    for return_val in returns:
        r = risk.DailyReturn(current, return_val)
        test_range.append(r)
        current = get_next_trading_dt(current, one_day)

    return test_range


def create_daily_trade_source(sids, trade_count, sim_params,
                              concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on sim_params.period_start, and daily
    thereafter for each sid. Thus, two sids should result in two trades per
    day.

    Important side-effect: sim_params.period_end will be modified
    to match the day of the final trade.
    """
    return create_trade_source(
        sids,
        trade_count,
        timedelta(days=1),
        sim_params,
        concurrent=concurrent
    )


def create_minutely_trade_source(sids, trade_count, sim_params,
                                 concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on sim_params.period_start, and every minute
    thereafter for each sid. Thus, two sids should result in two trades per
    minute.

    Important side-effect: sim_params.period_end will be modified
    to match the day of the final trade.
    """
    return create_trade_source(
        sids,
        trade_count,
        timedelta(minutes=1),
        sim_params,
        concurrent=concurrent
    )


def create_trade_source(sids, trade_count,
                        trade_time_increment, sim_params,
                        concurrent=False):

    args = tuple()
    kwargs = {
        'count': trade_count,
        'sids': sids,
        'start': sim_params.first_open,
        'delta': trade_time_increment,
        'filter': sids,
        'concurrent': concurrent
    }
    source = SpecificEquityTrades(*args, **kwargs)

    # TODO: do we need to set the trading environment's end to same dt as
    # the last trade in the history?
    #sim_params.period_end = trade_history[-1].dt

    return source


def create_test_df_source(sim_params=None):

    if sim_params:
        index = sim_params.trading_days
    else:
        start = pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)
        index = pd.DatetimeIndex(
            start=start,
            end=end,
            freq=pd.datetools.BDay()
        )

    x = np.arange(0, len(index))

    df = pd.DataFrame(x, index=index, columns=[0])

    return DataFrameSource(df), df


def create_test_panel_source(sim_params=None):
    start = sim_params.first_open \
        if sim_params else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = sim_params.last_close \
        if sim_params else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

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


def _load_raw_yahoo_data(indexes=None, stocks=None, start=None, end=None):
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
    return data


def load_from_yahoo(indexes=None, stocks=None, start=None, end=None):
    data = _load_raw_yahoo_data(indexes, stocks, start, end)
    df = pd.DataFrame({key: d['Adj Close'] for key, d in data.iteritems()})
    df.index = df.index.tz_localize(pytz.utc)
    return df


def load_bars_from_yahoo(indexes=None, stocks=None, start=None, end=None):
    data = _load_raw_yahoo_data(indexes, stocks, start, end)
    panel = pd.Panel(data)
    # Rename columns
    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume', 'price']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)

    return panel
