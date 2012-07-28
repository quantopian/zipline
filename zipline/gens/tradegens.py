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
    """Utility to generate a set of volumes. By default cycles
    through values from 100 to 1000, incrementing by 50.  Optional
    flag to give random values between 100 and 1000. """

    if rand:
        return (random.randrange(100, 1000) for i in xrange(n))
    else:
        return ((i * 50)%900 + 100 for i in xrange(n))
    
def fuzzy_dates(count = 500):
    """Add +-10 seconds to each event from a date_gen.  Note that
    this still guarantees sorting, since the default is minute separation
    of events."""
    for date in date_gen(n = count):
        yield date + timedelta(seconds = random.randint(-10, 10)) 

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
        filtered = ifilter(lambda event: event.sid in filter, unfiltered)
    else:
        filtered = unfiltered

    return filtered

def RandomEquityTrades(count = 500, sids = [1,2], filter = None):
    dates = fuzzy_dates(500)
    prices = mock_prices(500, rand = True)
    volumes = mock_volumes(500, rand = True)
    sids = cycle(iter(sids))

    arg_gen = izip(sids, prices, volumes, dates)
    
    unfiltered = (create_trade(*args) for args in arg_gen)
    
    if filter:
        filtered = ifilter(lambda event: event.sid in filter, unfiltered)
    else:
        filtered = unfiltered
    return filtered

if __name__ == "__main__":
    rand = RandomEquityTrades()
    pass
#     x = mock_volumes(500)
#     import nose.tools; nose.tools.set_trace()
#     trades = SpecificEquityTrades(filter = [1])
