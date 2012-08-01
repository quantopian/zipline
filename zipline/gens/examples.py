from datetime import datetime, timedelta

from zipline.utils.factory import create_trading_environment
from zipline.test_algorithms import TestAlgorithm

from zipline.gens.composites import SourceBundle, TransformBundle, date_sorted_sources, merged_transforms
from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import MovingAverage, Passthrough

if __name__ == "__main__":
    
    filter = [1,2,3,4]
    #Set up source a. One minute between events.
    args_a = tuple()
    kwargs_a = {
        'sids'   : [1,2],
        'start'  : datetime(2012,6,6,0),
        'delta'  : timedelta(minutes = 1),
        'filter' : filter
    }
    bundle_a = SourceBundle(SpecificEquityTrades, args_a, kwargs_a)

    #Set up source b. Two minutes between events.
    args_b = tuple()
    kwargs_b = {
        'sids'   : [2,3],
        'start'  : datetime(2012,6,6,0),
        'delta'  : timedelta(minutes = 2),
        'filter' : filter
    }
    bundle_b = SourceBundle(SpecificEquityTrades, args_b, kwargs_b)

    #Set up source c. Three minutes between events.
    args_c = tuple()
    kwargs_c = {
        'sids'   : [3,4],
        'start'  : datetime(2012,6,6,0),
        'delta'  : timedelta(minutes = 3),
        'filter' : filter
    }
    bundle_c = SourceBundle(SpecificEquityTrades, args_c, kwargs_c)
        
    source_bundles = (bundle_a, bundle_b, bundle_c)
    # Pipe our sources into sort.
    sort_out = date_sorted_sources(source_bundles)     

    passthrough = TransformBundle(Passthrough, (), {})
    mavg_price = TransformBundle(MovingAverage, (timedelta(minutes = 20), ['price', 'volume']), {})
    tnfm_bundles = (passthrough, mavg_price)

    merge_out = merged_transforms(sort_out, tnfm_bundles)

    for message in merge_out:
        print "Event: \n", message.event
        print "Transforms: \n", message.tnfms
        
    
