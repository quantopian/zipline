import random
from itertools import chain, repeat, cycle, ifilter, izip
from datetime import datetime, timedelta

from zipline.utils.factory import create_trade, create_trade
from zipline.gens.utils import date_gen

def mock_prices(n, rand = False):
    """Utility to generate a set of prices. By default
    cycles through values from 0.0 to 10.0 n times.  Optional
    flag to give random values between 0.0 and 10.0"""

    if rand:
        return (random.uniform(0.0, 10.0) for i in xrange(n))
    else:
        return (float(i % 11) for i in xrange(1,n+1))

def mock_volumes(n, rand = False):
    """Does the same as mock_prices.  Different function name
    for readability."""
    return mock_prices(n, rand)

def SpecificEquityTrades(count = 500, sids = [1, 2], event_list = None, filter = None):
    """Returns the first n events of event_list if specified. 
    Otherwise generates a sensible stream of events."""
    
    if event_list:
        unfiltered = (event for event in event_list)
    
    else:
        dates = date_gen(n = count)
        prices = mock_prices(count)
        volumes = mock_volumes(count)
        sids = cycle(iter(sids))
        
        arg_gen = izip(sids, prices, volumes, dates)
        
        unfiltered = (create_trade(*args) for args in arg_gen)
    if filter:
        filtered = ifilter(lambda event: event.sid in filter)
    else:
        filtered = unfiltered

    return filtered

if __name__ == "__main__":
    
    import nose.tools; nose.tools.set_trace()
    trades = SpecificEquityTrades()
