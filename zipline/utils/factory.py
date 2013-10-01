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


"""
Factory functions to prepare useful data.
"""
import pytz
import random
from collections import OrderedDict
from delorean import Delorean

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from zipline.protocol import DailyReturn, Event, DATASOURCE_TYPE
from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource)
from zipline.finance.trading import SimulationParameters
import zipline.finance.trading as trading
from zipline.sources.test_source import (
    date_gen,
    create_trade
)

# For backwards compatibility
from zipline.data.loader import (load_from_yahoo,
                                 load_bars_from_yahoo)

__all__ = ['load_from_yahoo', 'load_bars_from_yahoo']


def create_simulation_parameters(year=2006, start=None, end=None,
                                 capital_base=float("1.0e5"),
                                 num_days=None
                                 ):
    """Construct a complete environment with reasonable defaults"""
    if start is None:
        start = datetime(year, 1, 1, tzinfo=pytz.utc)
    if end is None:
        if num_days:
            trading.environment = trading.TradingEnvironment()
            start_index = trading.environment.trading_days.searchsorted(
                start)
            end = trading.environment.trading_days[start_index + num_days - 1]
        else:
            end = datetime(year, 12, 31, tzinfo=pytz.utc)
    sim_params = SimulationParameters(
        period_start=start,
        period_end=end,
        capital_base=capital_base,
    )

    return sim_params


def create_noop_environment():
    oneday = timedelta(days=1)
    start = datetime(2006, 1, 1, tzinfo=pytz.utc)

    bm_returns = []
    tr_curves = OrderedDict()
    for day in date_gen(start=start, delta=oneday, count=252):
        dr = DailyReturn(day, 0.01)
        bm_returns.append(dr)
        curve = {
            '10year': 0.0799,
            '1month': 0.0799,
            '1year': 0.0785,
            '20year': 0.0765,
            '2year': 0.0794,
            '30year': 0.0804,
            '3month': 0.0789,
            '3year': 0.0796,
            '5year': 0.0792,
            '6month': 0.0794,
            '7year': 0.0804,
            'tid': 1752
        }
        tr_curves[day] = curve

    load_nodata = lambda x: (bm_returns, tr_curves)

    return trading.TradingEnvironment(load=load_nodata)


def create_random_simulation_parameters():
    trading.environment = trading.TradingEnvironment()
    treasury_curves = trading.environment.treasury_curves

    for n in range(100):

        random_index = random.randint(
            0,
            len(treasury_curves) - 1
        )

        start_dt = treasury_curves.index[random_index]
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
    naive = current.replace(tzinfo=None)
    delo = Delorean(naive, pytz.utc.zone)
    ex_tz = trading.environment.exchange_tz
    next_dt = delo.shift(ex_tz).datetime

    while True:
        next_dt = next_dt + interval
        next_delo = Delorean(next_dt.replace(tzinfo=None), ex_tz)
        next_utc = next_delo.shift(pytz.utc.zone).datetime
        if trading.environment.is_market_hours(next_utc):
            break

    return next_utc


def create_trade_history(sid, prices, amounts, interval, sim_params,
                         source_id="test_factory"):
    trades = []
    current = sim_params.first_open

    oneday = timedelta(days=1)
    use_midnight = interval >= oneday
    for price, amount in zip(prices, amounts):
        if use_midnight:
            trade_dt = current.replace(hour=0, minute=0)
        else:
            trade_dt = current
        trade = create_trade(sid, price, amount, trade_dt, source_id)
        trades.append(trade)
        current = get_next_trading_dt(current, interval)

    assert len(trades) == len(prices)
    return trades


def create_dividend(sid, payment, declared_date, ex_date, pay_date):
    div = Event({
        'sid': sid,
        'gross_amount': payment,
        'net_amount': payment,
        'dt': declared_date.replace(hour=0, minute=0, second=0, microsecond=0),
        'ex_date': ex_date.replace(hour=0, minute=0, second=0, microsecond=0),
        'pay_date': pay_date.replace(hour=0, minute=0, second=0,
                                     microsecond=0),
        'type': DATASOURCE_TYPE.DIVIDEND
    })

    return div


def create_split(sid, ratio, date):
    return Event({
        'sid': sid,
        'ratio': ratio,
        'dt': date.replace(hour=0, minute=0, second=0, microsecond=0),
        'type': DATASOURCE_TYPE.SPLIT
    })


def create_txn(sid, price, amount, datetime):
    txn = Event({
        'sid': sid,
        'amount': amount,
        'dt': datetime,
        'price': price,
        'type': DATASOURCE_TYPE.TRANSACTION
    })
    return txn


def create_commission(sid, value, datetime):
    txn = Event({
        'dt': datetime,
        'type': DATASOURCE_TYPE.COMMISSION,
        'cost': value,
        'sid': sid
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


def create_returns_from_range(sim_params):
    current = sim_params.first_open
    end = sim_params.last_close
    test_range = []
    while current <= end:
        r = DailyReturn(current, random.random())
        test_range.append(r)
        current = trading.environment.next_trading_day(current)

    return test_range


def create_returns_from_list(returns, sim_params):
    current = sim_params.first_open
    test_range = []

    # sometimes the range starts with a non-trading day.
    if not trading.environment.is_trading_day(current):
        current = trading.environment.next_trading_day(current)

    for return_val in returns:
        r = DailyReturn(current, return_val)
        test_range.append(r)
        current = trading.environment.next_trading_day(current)

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
    # sim_params.period_end = trade_history[-1].dt

    return source


def create_test_df_source(sim_params=None, bars='daily'):
    if bars == 'daily':
        freq = pd.datetools.BDay()
    elif bars == 'minute':
        freq = pd.datetools.Minute()
    else:
        raise ValueError('%s bars not understood.' % freq)

    if sim_params:
        index = sim_params.trading_days
    else:
        start = pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)
        index = pd.DatetimeIndex(
            start=start,
            end=end,
            freq=freq
        )
        if bars == 'minute':
            new_index = []
            for i in index:
                market_open = i.replace(hour=14,
                                        minute=31)
                market_close = i.replace(hour=21,
                                         minute=0)

                if i >= market_open and i <= market_close:
                    new_index.append(i)
            index = new_index
    x = np.arange(1, len(index) + 1)

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


def create_test_panel_ohlc_source(sim_params=None):
    start = sim_params.first_open \
        if sim_params else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = sim_params.last_close \
        if sim_params else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

    index = pd.DatetimeIndex(start=start, end=end, freq=pd.datetools.day)
    price = np.arange(0, len(index)) + 100
    high = price * 1.05
    low = price * 0.95
    open_ = price + .1 * (price % 2 - .5)
    volume = np.ones(len(index)) * 1000
    arbitrary = np.ones(len(index))

    df = pd.DataFrame({'price': price,
                       'high': high,
                       'low': low,
                       'open': open_,
                       'volume': volume,
                       'arbitrary': arbitrary},
                      index=index)
    panel = pd.Panel.from_dict({0: df})

    return DataPanelSource(panel), panel
