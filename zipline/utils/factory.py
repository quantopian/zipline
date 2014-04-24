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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource)
from zipline.finance.trading import SimulationParameters
from zipline.finance import trading
from zipline.sources.test_source import create_trade


# For backwards compatibility
from zipline.data.loader import (load_from_yahoo,
                                 load_bars_from_yahoo)

__all__ = ['load_from_yahoo', 'load_bars_from_yahoo']


def create_simulation_parameters(year=2006, start=None, end=None,
                                 capital_base=float("1.0e5"),
                                 num_days=None, load=None,
                                 sids=None):
    """Construct a complete environment with reasonable defaults"""
    if start is None:
        start = datetime(year, 1, 1, tzinfo=pytz.utc)
    if end is None:
        if num_days:
            trading.environment = trading.TradingEnvironment(load=load)
            start_index = trading.environment.trading_days.searchsorted(
                start)
            end = trading.environment.trading_days[start_index + num_days - 1]
        else:
            end = datetime(year, 12, 31, tzinfo=pytz.utc)
    sim_params = SimulationParameters(
        period_start=start,
        period_end=end,
        capital_base=capital_base,
        sids=sids,
    )

    return sim_params


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
    next_dt = pd.Timestamp(current).tz_convert(trading.environment.exchange_tz)

    while True:
        # Convert timestamp to naive before adding day, otherwise the when
        # stepping over EDT an hour is added.
        next_dt = pd.Timestamp(next_dt.replace(tzinfo=None))
        next_dt = next_dt + interval
        next_dt = pd.Timestamp(next_dt, tz=trading.environment.exchange_tz)
        next_dt_utc = next_dt.tz_convert('UTC')
        if trading.environment.is_market_hours(next_dt_utc):
            break
        next_dt = next_dt_utc.tz_convert(trading.environment.exchange_tz)

    return next_dt_utc


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
        'payment_sid': None,
        'ratio': None,
        'dt': pd.tslib.normalize_date(declared_date),
        'ex_date': pd.tslib.normalize_date(ex_date),
        'pay_date': pd.tslib.normalize_date(pay_date),
        'type': DATASOURCE_TYPE.DIVIDEND,
        'source_id': 'MockDividendSource'
    })

    return div


def create_stock_dividend(sid, payment_sid, ratio, declared_date,
                          ex_date, pay_date):
    return Event({
        'sid': sid,
        'payment_sid': payment_sid,
        'ratio': ratio,
        'net_amount': None,
        'gross_amount': None,
        'dt': pd.tslib.normalize_date(declared_date),
        'ex_date': pd.tslib.normalize_date(ex_date),
        'pay_date': pd.tslib.normalize_date(pay_date),
        'type': DATASOURCE_TYPE.DIVIDEND,
        'source_id': 'MockDividendSource'
    })


def create_split(sid, ratio, date):
    return Event({
        'sid': sid,
        'ratio': ratio,
        'dt': date.replace(hour=0, minute=0, second=0, microsecond=0),
        'type': DATASOURCE_TYPE.SPLIT,
        'source_id': 'MockSplitSource'
    })


def create_txn(sid, price, amount, datetime):
    txn = Event({
        'sid': sid,
        'amount': amount,
        'dt': datetime,
        'price': price,
        'type': DATASOURCE_TYPE.TRANSACTION,
        'source_id': 'MockTransactionSource'
    })
    return txn


def create_commission(sid, value, datetime):
    txn = Event({
        'dt': datetime,
        'type': DATASOURCE_TYPE.COMMISSION,
        'cost': value,
        'sid': sid,
        'source_id': 'MockCommissionSource'
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
    return pd.Series(index=sim_params.trading_days,
                     data=np.random.rand(len(sim_params.trading_days)))


def create_returns_from_list(returns, sim_params):
    return pd.Series(index=sim_params.trading_days[:len(returns)],
                     data=returns)


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
        raise ValueError('%s bars not understood.' % bars)

    if sim_params:
        index = sim_params.trading_days
    else:
        if trading.environment is None:
            trading.environment = trading.TradingEnvironment()

        start = pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

        days = trading.environment.days_in_range(start, end)

        if bars == 'daily':
            index = days
        if bars == 'minute':
            index = pd.DatetimeIndex([], freq=freq)

            for day in days:
                day_index = trading.environment.market_minutes_for_day(day)
                index = index.append(day_index)

    x = np.arange(1, len(index) + 1)

    df = pd.DataFrame(x, index=index, columns=[0])

    return DataFrameSource(df), df


def create_test_panel_source(sim_params=None):
    start = sim_params.first_open \
        if sim_params else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = sim_params.last_close \
        if sim_params else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

    if trading.environment is None:
        trading.environment = trading.TradingEnvironment()

    index = trading.environment.days_in_range(start, end)

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

    if trading.environment is None:
        trading.environment = trading.TradingEnvironment()

    index = trading.environment.days_in_range(start, end)
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
