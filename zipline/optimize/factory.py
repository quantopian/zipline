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
Factory functions to prepare useful data for optimize tests.

Author: Thomas V. Wiecki (thomas.wiecki@gmail.com), 2012
"""
from datetime import timedelta

from zipline.utils.protocol_utils import ndict
import zipline.protocol as zp

from zipline.utils.factory import (
    get_next_trading_dt,
    create_trading_environment
)
from zipline.gens.tradegens import SpecificEquityTrades
from zipline.optimize.algorithms import BuySellAlgorithm
from zipline.finance.slippage import FixedSlippage

from copy import copy
from itertools import cycle


def create_updown_trade_source(sid, trade_count, trading_environment,
                               base_price, amplitude):
    """Create the updown trade source. This source emits events with
    the price going up and down by the same amount in each
    iteration. The trade source is thus perfectly predictable. This is
    used for a test case for the optimization code.

    :Arguments:
        sid : int
            SID of stock to create.
        trade_count : int
            How many trade events to create (will also influence order count)
        trading_environment : TradeEnvironment object
            The trading environment to use
            (see zipline.factory.create_trading_environment)
        base_price : int
            The average price that each iteration will hover around.
        amplitude : int
            How much the price will go up and down each iteration.

    :Returns:
        source : SpecificEquityTrades
            The trade source emitting up down events.
    """
    volume = 1000
    events = []
    price = base_price - amplitude / 2.

    cur = trading_environment.first_open
    one_day = timedelta(minutes=1)

    #create iterator to cycle through up and down phases
    change = cycle([1, -1])

    for i in xrange(trade_count + 2):
        cur = get_next_trading_dt(cur, one_day, trading_environment)

        event = ndict({
            "type": zp.DATASOURCE_TYPE.TRADE,
            "sid": sid,
            "price": price,
            "volume": volume,
            "dt": cur,
        })

        events.append(event)

        price += change.next() * amplitude

    trading_environment.period_end = cur

    source = SpecificEquityTrades(events)

    return source


def create_predictable_zipline(config, offset=0, simulate=True):
    """Create a test zipline object as specified by config. The
    zipline will use the UpDown tradesource which is perfectly
    predictable.

    Trade source parameters can be specified inside the config object.

    :Trade source arguments:
        config['sid'] : int
            SID of stock to create.
        config['amplitude'] : int (default 10)
            How much the price will go up and down each iteration.
        config['base_price'] : int (default 50)
            The average price that each iteration will hover around.
        config['trade_count'] : int (default 3)
            How many trade events to create (will also influence order count)

    If not specified, the BuySellAlgorithm is used by default. This
    can be changed by setting config['algorithm'].

    :Arguments:
        offset : int (default 0)
            The offset parameter specifies how much the BuySellAlgorithm will
            order each iteration and is a negative quadratic centered around
            0. Thus, any deviations from 0 will lead to less buy and sell
            orders each iteration and ultimately to less compound returns.
        simulate : bool (default True)
            Whether to call .simulate(blocking=True) on the created zipline
            argument.

    :Returns:
        zipline : class zipline
            created zipline object
        config : dict
            the config dict used to create the zipline
    """
    config = copy(config)
    sid = config['sid']
    # remove
    amplitude = config.pop('amplitude', 10)
    base_price = config.pop('base_price', 50)
    trade_count = config.pop('trade_count', 3)

    trading_environment = create_trading_environment()
    source = create_updown_trade_source(sid,
                                        trade_count,
                                        trading_environment,
                                        base_price,
                                        amplitude)

    if 'algorithm' not in config:
        algorithm = BuySellAlgorithm(sids=[sid], amount=100, offset=offset)

    config['order_count'] = trade_count - 1
    config['trade_count'] = trade_count
    config['trade_source'] = source
    config['environment'] = trading_environment
    config['slippage'] = FixedSlippage()
    config['devel'] = True

    return algorithm, config
