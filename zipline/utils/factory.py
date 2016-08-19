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

import pandas as pd
import numpy as np
from datetime import timedelta

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource)
from zipline.finance.trading import (
    SimulationParameters, TradingEnvironment, noop_load
)
from zipline.sources.test_source import create_trade
from zipline.data.loader import (  # For backwards compatibility
    load_from_yahoo,
    load_bars_from_yahoo,
)


__all__ = ['load_from_yahoo', 'load_bars_from_yahoo']


def create_simulation_parameters(year=2006, start=None, end=None,
                                 capital_base=float("1.0e5"),
                                 num_days=None,
                                 data_frequency='daily',
                                 emission_rate='daily',
                                 env=None):
    if env is None:
        # Construct a complete environment with reasonable defaults
        env = TradingEnvironment(load=noop_load)
    if start is None:
        start = pd.Timestamp("{0}-01-01".format(year), tz='UTC')
    if end is None:
        if num_days:
            start_index = env.trading_days.searchsorted(start)
            end = env.trading_days[start_index + num_days - 1]
        else:
            end = pd.Timestamp("{0}-12-31".format(year), tz='UTC')
    sim_params = SimulationParameters(
        period_start=start,
        period_end=end,
        capital_base=capital_base,
        data_frequency=data_frequency,
        emission_rate=emission_rate,
        env=env,
    )

    return sim_params


def get_next_trading_dt(current, interval, env):
    next_dt = pd.Timestamp(current).tz_convert(env.exchange_tz)

    while True:
        # Convert timestamp to naive before adding day, otherwise the when
        # stepping over EDT an hour is added.
        next_dt = pd.Timestamp(next_dt.replace(tzinfo=None))
        next_dt = next_dt + interval
        next_dt = pd.Timestamp(next_dt, tz=env.exchange_tz)
        next_dt_utc = next_dt.tz_convert('UTC')
        if env.is_market_hours(next_dt_utc):
            break
        next_dt = next_dt_utc.tz_convert(env.exchange_tz)

    return next_dt_utc


def create_trade_history(sid, prices, amounts, interval, sim_params, env,
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
        current = get_next_trading_dt(current, interval, env)

    assert len(trades) == len(prices)
    return trades


def create_dividend(sid, payment, declared_date, ex_date, pay_date):
    div = Event({
        'sid': sid,
        'gross_amount': payment,
        'net_amount': payment,
        'payment_sid': None,
        'ratio': None,
        'declared_date': pd.tslib.normalize_date(declared_date),
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


def create_txn_history(sid, priceList, amtList, interval, sim_params, env):
    txns = []
    current = sim_params.first_open

    for price, amount in zip(priceList, amtList):
        current = get_next_trading_dt(current, interval, env)

        txns.append(create_txn(sid, price, amount, current))
        current = current + interval
    return txns


def create_returns_from_range(sim_params):
    return pd.Series(index=sim_params.trading_days,
                     data=np.random.rand(len(sim_params.trading_days)))


def create_returns_from_list(returns, sim_params):
    return pd.Series(index=sim_params.trading_days[:len(returns)],
                     data=returns)


def create_daily_trade_source(sids, sim_params, env, concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on sim_params.period_start, and daily
    thereafter for each sid. Thus, two sids should result in two trades per
    day.
    """
    return create_trade_source(
        sids,
        timedelta(days=1),
        sim_params,
        env=env,
        concurrent=concurrent,
    )


def create_minutely_trade_source(sids, sim_params, env, concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on sim_params.period_start, and every minute
    thereafter for each sid. Thus, two sids should result in two trades per
    minute.
    """
    return create_trade_source(
        sids,
        timedelta(minutes=1),
        sim_params,
        env=env,
        concurrent=concurrent,
    )


def create_trade_source(sids, trade_time_increment, sim_params, env,
                        concurrent=False):

    # If the sim_params define an end that is during market hours, that will be
    # used as the end of the data source
    if env.is_market_hours(sim_params.period_end):
        end = sim_params.period_end
    # Otherwise, the last_close after the period_end is used as the end of the
    # data source
    else:
        end = sim_params.last_close

    args = tuple()
    kwargs = {
        'sids': sids,
        'start': sim_params.first_open,
        'end': end,
        'delta': trade_time_increment,
        'filter': sids,
        'concurrent': concurrent,
        'env': env,
    }
    source = SpecificEquityTrades(*args, **kwargs)

    return source


def create_test_df_source(sim_params=None, env=None, bars='daily'):
    if bars == 'daily':
        freq = pd.datetools.BDay()
    elif bars == 'minute':
        freq = pd.datetools.Minute()
    else:
        raise ValueError('%s bars not understood.' % bars)

    if sim_params and bars == 'daily':
        index = sim_params.trading_days
    else:
        if env is None:
            env = TradingEnvironment(load=noop_load)

        start = pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

        days = env.days_in_range(start, end)

        if bars == 'daily':
            index = days
        if bars == 'minute':
            index = pd.DatetimeIndex([], freq=freq)

            for day in days:
                day_index = env.market_minutes_for_day(day)
                index = index.append(day_index)

    x = np.arange(1, len(index) + 1)

    df = pd.DataFrame(x, index=index, columns=[0])

    return DataFrameSource(df), df


def create_test_panel_source(sim_params=None, env=None, source_type=None):
    start = sim_params.first_open \
        if sim_params else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = sim_params.last_close \
        if sim_params else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

    if env is None:
        env = TradingEnvironment(load=noop_load)

    index = env.days_in_range(start, end)

    price = np.arange(0, len(index))
    volume = np.ones(len(index)) * 1000

    arbitrary = np.ones(len(index))

    df = pd.DataFrame({'price': price,
                       'volume': volume,
                       'arbitrary': arbitrary},
                      index=index)
    if source_type:
        df['type'] = source_type

    panel = pd.Panel.from_dict({0: df})

    return DataPanelSource(panel), panel


def create_test_panel_ohlc_source(sim_params, env):
    start = sim_params.first_open \
        if sim_params else pd.datetime(1990, 1, 3, 0, 0, 0, 0, pytz.utc)

    end = sim_params.last_close \
        if sim_params else pd.datetime(1990, 1, 8, 0, 0, 0, 0, pytz.utc)

    index = env.days_in_range(start, end)
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
