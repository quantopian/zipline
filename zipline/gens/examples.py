import pytz

from time import sleep
from pprint import pprint as pp
from datetime import datetime, timedelta

from zipline.utils.factory import create_trading_environment
from zipline.test_algorithms import TestAlgorithm

from zipline.gens.composites import SourceBundle, TransformBundle, \
    date_sorted_sources, merged_transforms
from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import MovingAverage, Passthrough, StatefulTransform
from zipline.gens.tradesimulation import TradeSimulationClient as tsc

import zipline.protocol as zp

if __name__ == "__main__":
    
    filter = [2,3]
    #Set up source a. One minute between events.
    args_a = tuple()
    kwargs_a = {
        'count'  : 325,
        'sids'   : [1,2,3],
        'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
        'delta'  : timedelta(hours = 6),
        'filter' : filter
    }
    source_a = SpecificEquityTrades(*args_a, **kwargs_a)

    #Set up source b. Two minutes between events.
    args_b = tuple()
    kwargs_b = {
        'count'  : 7500,
        'sids'   : [2,3,4],
        'start'  : datetime(2012,1,3,14, tzinfo = pytz.utc),
        'delta'  : timedelta(minutes = 5),
        'filter' : filter
    }
    source_b = SpecificEquityTrades(*args_b, **kwargs_b)

    #Set up source c. Three minutes between events.

    sorted = date_sorted_sources(source_a, source_b)     
    
    passthrough = StatefulTransform(Passthrough)
    mavg_price = StatefulTransform(MovingAverage, timedelta(minutes = 20), ['price'])
    
    merged = merged_transforms(sorted, passthrough, mavg_price)
    
    algo = TestAlgorithm(2, 10, 100, sid_filter = [2,3])
    environment = create_trading_environment(year = 2012)
    style = zp.SIMULATION_STYLE.FIXED_SLIPPAGE
    
    trading_client = tsc(algo, environment, style)
    
    for message in trading_client.simulate(merged):
       pp(message)
    
