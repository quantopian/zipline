import random
from itertools import chain, repeat, cycle, ifilter, izip
from datetime import datetime, timedelta

from zipline.utils.factory import create_trade
from zipline.gens.utils import hash_args, mock_done

def date_gen(start = datetime(2012, 6, 6, 0),
             delta = timedelta(minutes = 1), 
             count = 100):
    """
    Utility to generate a stream of dates.
    """
    return (start + (i * delta) for i in xrange(count))

def mock_prices(count, rand = False):
    """
    Utility to generate a stream of mock prices. By default
    cycles through values from 0.0 to 10.0, n times.  Optional
    flag to give random values between 0.0 and 10.0
    """

    if rand:
        return (random.uniform(0.0, 10.0) for i in xrange(count))
    else:
        return (float(i % 11) for i in xrange(1,count+1))

def mock_volumes(count, rand = False):
    """
    Utility to generate a set of volumes. By default cycles
    through values from 100 to 1000, incrementing by 50.  Optional
    flag to give random values between 100 and 1000. 
    """
    if rand:
        return (random.randrange(100, 1000) for i in xrange(count))
    else:
        return ((i * 50)%900 + 100 for i in xrange(count))
    
def fuzzy_dates(count = 500):
    """
    Add +-10 seconds to each event from a date_gen.  Note that this
    still guarantees sorting, since the default on date_gen is minute
    separation of events.
    """
    for date in date_gen(count = count):
        yield date + timedelta(seconds = random.randint(-10, 10)) 

def SpecificEquityTrades(*args, **config):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.
    """
    # We shouldn't get any positional arguments.
    assert args == ()

    # Unpack config dictionary with default values.
    count = config.get('count', 500)
    sids = config.get('sids', [1, 2])
    start = config.get('start', datetime(2012, 6, 6, 0))
    delta = config.get('delta', timedelta(minutes = 1))

    # Default to None for event_list and filter.
    event_list = config.get('event_list') 
    filter = config.get('filter')

    arg_string = hash_args(*args, **config)
    namestring = "SpecificEquityTrades" + arg_string
    # If we have an event_list, ignore the other arguments and use the list.
    # TODO: still append our namestring?
    if event_list:
        unfiltered = (event for event in event_list)

    # Set up iterators for each expected field.
    else:
        dates = date_gen(count = count, start = start, delta = delta)
        prices = mock_prices(count)
        volumes = mock_volumes(count)
        sids = cycle(sids)

        # Combine the iterators into a single iterator of arguments
        arg_gen = izip(sids, prices, volumes, dates)

        # Convert argument packages into events.
        unfiltered = (create_trade(*args, source_id = namestring)
                      for args in arg_gen)

    # If we specified a sid filter, filter out elements that don't match the filter.
    if filter:
        filtered = ifilter(lambda event: event.sid in filter, unfiltered)

    # Otherwise just use all events.
    else:
        filtered = unfiltered

    # Add a done message to the end of the stream. For a live
    # datasource this would be handled by the containing Component.
    out = chain(filtered, [mock_done(namestring)])
    return out

def RandomEquityTrades(*args, **config):
    # We shouldn't get any positional args.
    assert args == ()
    
    count = config.get('count', 500)
    sids = config.get('sids', [1,2])
    filter = config.get('filter')

    dates = fuzzy_dates(count)
    prices = mock_prices(count, rand = True)
    volumes = mock_volumes(count, rand = True)
    sids = cycle(sids)

    arg_gen = izip(sids, prices, volumes, dates)
    
    unfiltered = (create_trade(*args) for args in arg_gen)
    
    if filter:
        filtered = ifilter(lambda event: event.sid in filter, unfiltered)
    else:
        filtered = unfiltered
    return filtered

# if __name__ == "__main__":
#     import nose.tools; nose.tools.set_trace()
#     trades = SpecificEquityTrades(filter = [1])
