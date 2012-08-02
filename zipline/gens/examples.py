import pytz
from time import sleep

from pprint import pprint as pp
from datetime import datetime, timedelta

from zipline.utils.factory import create_trading_environment
from zipline.test_algorithms import TestAlgorithm

from zipline.gens.composites import SourceBundle, TransformBundle, \
    date_sorted_sources, merged_transforms
from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import MovingAverage, Passthrough
from zipline.gens.tradesimulation import trade_simulation_client as tsc

import zipline.protocol as zp

if __name__ == "__main__":
    
    filter = [2,3]
    #Set up source a. One minute between events.
    args_a = tuple()
    kwargs_a = {
        'sids'   : [1,2,3],
        'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
        'delta'  : timedelta(minutes = 1),
        'filter' : filter
    }
    source_a = SpecificEquityTrades(*args_a, **kwargs_a)

    #Set up source b. Two minutes between events.
    args_b = tuple()
    kwargs_b = {
        'sids'   : [2,3,4],
        'start'  : datetime(2012,1,3,14, tzinfo = pytz.utc),
        'delta'  : timedelta(minutes = 1),
        'filter' : filter
    }
    source_b = SpecificEquityTrades(*args_a, **kwargs_a)
    
    #Set up source c. Three minutes between events.
    sort_out = date_sorted_sources(source_a, source_b)     

#     passthrough = TransformBundle(Passthrough, (), {})
#     mavg_price = TransformBundle(MovingAverage, (timedelta(minutes = 20), ['price']), {})
#     tnfm_bundles = (passthrough, mavg_price)

#     merge_out = merged_transforms(sort_out, tnfm_bundles)

# #   for message in merge_out:
# #       print message
    
#     algo = TestAlgorithm(2, 100, 100)
#     environment = create_trading_environment(year = 2012)
#     style = zp.SIMULATION_STYLE.FIXED_SLIPPAGE
    
#     client_out = tsc(merge_out, algo, environment, style)
#     for message in client_out:
    #    pp(message)
        
        


