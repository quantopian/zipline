"""
Factory functions to prepare useful data for optimize tests.

Author: Thomas V. Wiecki (thomas.wiecki@gmail.com), 2012
"""
from datetime import datetime, timedelta

import zipline.protocol as zp

from zipline.test.factory import get_next_trading_dt
from zipline.finance.sources import SpecificEquityTrades
from zipline.optimize.algorithms import BuySellAlgorithm
from zipline.lines import SimulatedTrading

def create_updown_trade_source(sid, trade_count, trading_environment, start_price, amplitude):
    from itertools import cycle
    volume = 1000
    events = []
    price = start_price-amplitude/2.

    cur = trading_environment.first_open
    one_day = timedelta(days = 1)

    #create iterator to cycle through up and down phases
    change = cycle([1,-1])

    for i in xrange(trade_count + 2):
        cur = get_next_trading_dt(cur, one_day, trading_environment)

        event = zp.ndict({
            "type"      : zp.DATASOURCE_TYPE.TRADE,
            "sid"       : sid,
            "price"     : price,
            "volume"    : volume,
            "dt"        : cur,
        })

        events.append(event)

        price += change.next()*amplitude

    trading_environment.period_end = cur

    source = SpecificEquityTrades(sid, events)

    return source


def create_predictable_zipline(config, sid=133, amplitude=10, base_price=50, offset=0):
    config = deepcopy(config)
    trading_environment = create_trading_environment()
    source = create_updown_trade_source(sid,
                                        config['trade_count'],
                                        trading_environment,
                                        base_price,
                                        amplitude)

    algo = RegularIntervalBuySellAlgorithm(sid, 100, offset)
    config['algorithm'] = algo
    config['trade_source'] = source
    config['environment'] = trading_environment
    zipline = SimulatedTrading.create_test_zipline(**config)
    zipline.simulate(blocking=True)

    return zipline
