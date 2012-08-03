"""
Tools to generate trade events without a backing store. Useful for testing
and zipline development
"""
import random
from itertools import chain, cycle, ifilter, izip
from datetime import datetime, timedelta

from zipline.utils.factory import create_trade
from zipline.gens.utils import hash_args

def date_gen(start = datetime(2006, 6, 6, 12),
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

class SpecificEquityTrades(object):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    count  : integer representing number of trades
    sids   : list of values representing simulated internal sids
    start  : start date
    delta  : timedelta between internal events
    filter : filter to remove the sids
    """

    def __init__(self, *args, **kwargs):
        # We shouldn't get any positional arguments.
        assert len(args) == 0

        # Unpack config dictionary with default values.
        self.count = kwargs.get('count', 500)
        self.sids = kwargs.get('sids', [1, 2])
        self.start = kwargs.get('start', datetime(2012, 6, 6, 0))
        self.delta = kwargs.get('delta', timedelta(minutes = 1))

        # Default to None for event_list and filter.
        self.event_list = kwargs.get('event_list')
        self.filter = kwargs.get('filter')

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(*args, **kwargs)

        self.generator = self.create_fresh_generator()

    def __iter__(self):
        return self

    def next(self):
        return self.generator.next()

    def rewind(self):
        self.generator = self.create_fresh_generator()

    def get_hash(self):
        return self.__class__.__name__ + "-" + self.arg_string

    def create_fresh_generator(self):

        if self.event_list:
            unfiltered = (event for event in self.event_list)

        # Set up iterators for each expected field.
        else:
            dates = date_gen(count=self.count,
                             start=self.start,
                             delta=self.delta
            )
            prices = mock_prices(self.count)
            volumes = mock_volumes(self.count)
            sids = cycle(self.sids)

            # Combine the iterators into a single iterator of arguments
            arg_gen = izip(sids, prices, volumes, dates)

            # Convert argument packages into events.
            unfiltered = (create_trade(*args, source_id = self.get_hash())
                          for args in arg_gen)

        # If we specified a sid filter, filter out elements that don't
        # match the filter.
        if self.filter:
            filtered = ifilter(lambda event: event.sid in self.filter, unfiltered)

        # Otherwise just use all events.
        else:
            filtered = unfiltered

        # Return the filtered event stream.
        return filtered


# !!!!!!! Deprecated for now !!!!!!!!!

def RandomEquityTrades(object):

    def __init__(self):
        # We shouldn't get any positional args.
        assert args == ()

        self.count = config.get('count', 500)
        self.sids = config.get('sids', [1,2])
        self.filter = config.get('filter')

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
