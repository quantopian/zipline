import pytz
import time

from time import sleep
from pprint import pprint as pp
from datetime import datetime, timedelta
from itertools import izip

from zipline.utils.factory import create_trading_environment
from zipline.test_algorithms import TestAlgorithm

from zipline.gens.composites import SourceBundle, TransformBundle, \
    date_sorted_sources, merged_transforms, sequential_transforms
from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import MovingAverage, Passthrough, StatefulTransform
from zipline.gens.tradesimulation import TradeSimulationClient as tsc

import zipline.protocol as zp

if __name__ == "__main__":
    
    filter = [2,3]
    #Set up source a. Six minutes between events.
    args_a = tuple()
    kwargs_a = {
        'count'  : 1000,
        'sids'   : [1,2,3],
        'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
        'delta'  : timedelta(minutes = 6),
        'filter' : filter
    }
    source_a = SpecificEquityTrades(*args_a, **kwargs_a)
    source_a_prime = SpecificEquityTrades(*args_a, **kwargs_a)

    #Set up source b. Five minutes between events.
    args_b = tuple()
    kwargs_b = {
        'count'  : 1000,
        'sids'   : [2,3,4],
        'start'  : datetime(2012,1,3,14, tzinfo = pytz.utc),
        'delta'  : timedelta(minutes = 5),
        'filter' : filter
    }
    source_b = SpecificEquityTrades(*args_b, **kwargs_b)
    source_b_prime = SpecificEquityTrades(*args_b, **kwargs_b)

    sorted = date_sorted_sources(source_a, source_b)     
    sorted_prime = date_sorted_sources(
        source_a_prime, 
        source_b_prime
    )     
    
    passthrough = StatefulTransform(Passthrough)
    mavg_price = StatefulTransform(
        MovingAverage, 
        timedelta(minutes = 20), 
        ['price']
    )
    
    passthrough_prime = StatefulTransform(Passthrough)
    mavg_price_prime = StatefulTransform(
        MovingAverage, 
        timedelta(minutes = 20), 
        ['price']
    )
    
    merged = merged_transforms(sorted, passthrough, mavg_price)
    start = time.time()
    for message in merged:
        assert 1 + 1 == 2
    stop = time.time()
    merge_time = stop - start
    print "Merge time: %s" % str(merge_time)

    sequential = sequential_transforms(
        sorted_prime, 
        passthrough_prime, 
        mavg_price_prime
    )
    
    start = time.time()
    for message in sequential:
        assert 1 + 1 == 2
    stop = time.time()
    seq_time = stop - start
    print "Sequential time: %s" % str(seq_time)
    print "Merge/Seq: %s" % (str(merge_time/seq_time))
    
   
#    merged = merged_transforms(sorted, passthrough, mavg_price)
    
    # algo = TestAlgorithm(2, 10, 100, sid_filter = [2,3])
#     environment = create_trading_environment(year = 2012)
#     style = zp.SIMULATION_STYLE.FIXED_SLIPPAGE
    
#     trading_client = tsc(algo, environment, style)
    
#     for message in trading_client.simulate(merged):
#        pp(message)
    
