#
# Copyright 2016 Quantopian, Inc.
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
import pandas as pd
import numpy as np
from datetime import timedelta, datetime

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.sources import SpecificEquityTrades
from zipline.finance.trading import SimulationParameters
from zipline.sources.test_source import create_trade
from zipline.data.loader import (  # For backwards compatibility
    load_from_yahoo,
    load_bars_from_yahoo,
)
from zipline.utils.calendars import get_calendar


__all__ = ['load_from_yahoo', 'load_bars_from_yahoo']


def create_simulation_parameters(year=2006, start=None, end=None,
                                 capital_base=float("1.0e5"),
                                 num_days=None,
                                 data_frequency='daily',
                                 emission_rate='daily',
                                 trading_calendar=None):

    if not trading_calendar:
        trading_calendar = get_calendar("NYSE")

    if start is None:
        start = pd.Timestamp("{0}-01-01".format(year), tz='UTC')
    elif type(start) == datetime:
        start = pd.Timestamp(start)

    if end is None:
        if num_days:
            start_index = trading_calendar.all_sessions.searchsorted(start)
            end = trading_calendar.all_sessions[start_index + num_days - 1]
        else:
            end = pd.Timestamp("{0}-12-31".format(year), tz='UTC')
    elif type(end) == datetime:
        end = pd.Timestamp(end)

    sim_params = SimulationParameters(
        start_session=start,
        end_session=end,
        capital_base=capital_base,
        data_frequency=data_frequency,
        emission_rate=emission_rate,
        trading_calendar=trading_calendar,
    )

    return sim_params


def get_next_trading_dt(current, interval, trading_calendar):
    next_dt = pd.Timestamp(current).tz_convert(trading_calendar.tz)

    while True:
        # Convert timestamp to naive before adding day, otherwise the when
        # stepping over EDT an hour is added.
        next_dt = pd.Timestamp(next_dt.replace(tzinfo=None))
        next_dt = next_dt + interval
        next_dt = pd.Timestamp(next_dt, tz=trading_calendar.tz)
        next_dt_utc = next_dt.tz_convert('UTC')
        if trading_calendar.is_open_on_minute(next_dt_utc):
            break
        next_dt = next_dt_utc.tz_convert(trading_calendar.tz)

    return next_dt_utc


def create_trade_history(sid, prices, amounts, interval, sim_params,
                         trading_calendar, source_id="test_factory"):
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
        current = get_next_trading_dt(current, interval, trading_calendar)

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


def create_txn_history(sid, priceList, amtList, interval, sim_params,
                       trading_calendar):
    txns = []
    current = sim_params.first_open

    for price, amount in zip(priceList, amtList):
        current = get_next_trading_dt(current, interval, trading_calendar)

        txns.append(create_txn(sid, price, amount, current))
        current = current + interval
    return txns


def create_returns_from_range(sim_params):
    return pd.Series(index=sim_params.sessions,
                     data=np.random.rand(len(sim_params.sessions)))


def create_returns_from_list(returns, sim_params):
    return pd.Series(index=sim_params.sessions[:len(returns)],
                     data=returns)


def create_daily_trade_source(sids, sim_params, env, trading_calendar,
                              concurrent=False):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on sim_params.start_session, and daily
    thereafter for each sid. Thus, two sids should result in two trades per
    day.
    """
    return create_trade_source(
        sids,
        timedelta(days=1),
        sim_params,
        env=env,
        trading_calendar=trading_calendar,
        concurrent=concurrent,
    )


def create_trade_source(sids, trade_time_increment, sim_params, env,
                        trading_calendar, concurrent=False):

    # If the sim_params define an end that is during market hours, that will be
    # used as the end of the data source
    if trading_calendar.is_open_on_minute(sim_params.end_session):
        end = sim_params.end_session
    # Otherwise, the last_close after the end_session is used as the end of the
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
        'trading_calendar': trading_calendar,
    }
    source = SpecificEquityTrades(*args, **kwargs)

    return source
