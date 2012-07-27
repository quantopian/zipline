"""
Generator versions of transforms.
"""
import random
import pytz
import logbook
import pymongo
import types

from pymongo import ASCENDING
from datetime import datetime, timedelta
from collections import deque, defaultdict
from numbers import Number
from itertools import izip

from zipline import ndict
from zipline.gens.utils import hash_args, date_gen
from zipline.gens.utils import  assert_feed_unframe_protocol, assert_transform_protocol

import zipline.protocol as zp

def PassthroughTransformGen(stream_in):
    """Trivial transform for event forwarding."""

    # hash_args with no arguments is the same as:
    # hasher = hashlib.md5() 
    # hasher.update(":"); 
    # hashlib.md5.digest().

    namestring = "Passthrough" + hash_args()

    for message in stream_in:
        assert_feed_unframe_protocol(message)
        out_value = message
        assert_transform_protocol(out_value)
        yield (namestring, out_value)

def FunctionalTransformGen(stream_in, fun, *args, **kwargs):
    """
    Generic transform generator that takes each message from an in-stream
    and yields the output of a function on that message. Not sure how
    useful this will be in reality, but good for testing.
    """
    
    # TODO: Distinguish between functions and classes in hash_args.
    namestring = fun.__name__ + hash_args(*args, **kwargs)
    
    for message in stream_in:
        assert_feed_unframe_protocol(message)
        out_value = fun(message, *args, **kwargs)
        assert_transform_protocol(out_value)
        yield(namestring, out_value)
    
def StatefulTransformGen(stream_in, tnfm_class, *args, **kwargs):
    """
    Generic transform generator that takes each message from an in-stream
    and feeds it to a state class.  For each call to update, the state
    class must produce a message to be fed downstream.
    """    

    # Create an instance of our transform class.
    state = tnfm_class(*args, **kwargs)

    # Generate the string associated with this generator's output.
    namestring = tnfm_class.__name__ + hash_args(*args, **kwargs)

    for message in stream_in:
        assert_feed_unframe_protocol(message)
        out_value = state.update(message)
        assert_transform_protocol(out_value)
        yield (namestring, out_value)

def MovingAverageTransformGen(stream_in, days, fields):
    """
    Generator that uses the MovingAverage state class to calculate
    a moving average for all stocks over a specified number of days.
    """
    return StatefulTransformGen(stream_in, MovingAverage, timedelta(days=days), fields)

class MovingAverage(object):
    """
    Class that maintains a dictionary from sids to EventWindows
    Upon receipt of each message we update the
    corresponding window and return the calculated average.
    """
    
    def __init__(self, delta, fields):
        self.delta = delta
        self.fields = fields

        # No way to pass arguments to the defaultdict factory, so we
        # need to define a method to generate the correct EventWindows.
        self.sid_windows = defaultdict(self.create_window)

    def create_window(self):
        """Factory method for self.sid_windows."""
        return EventWindow(self.delta, self.fields)
    
    def update(self, event):
        """
        Update the event window for this event's sid.  Return an ndict from
        tracked fields to averages.
        """
        
        assert isinstance(event, ndict),"Bad event in MovingAverage: %s" % event
        assert event.has_key('sid'), "No sid in MovingAverage: %s" % event
        
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        
        return window.get_averages()
    
class EventWindow(object):
    """
    Maintains a list of events that are within a certain timedelta
    of the most recent tick.  The expected use of this class is to
    track events associated with a single sid. We provide simple
    functionality for averages, but anything more complicated 
    should be handled by a containing class.
    """

    def __init__(self, delta, fields):
        self.ticks  = deque()
        self.delta  = delta
        self.fields = fields
        self.totals = defaultdict(float)
        
    def __len__(self):
        return len(self.ticks)
        
    def update(self, event):
        self.assert_well_formed(event)
        # Add new event and increment totals.
        self.ticks.append(event)
        for field in self.fields:
            self.totals[field] += event[field]
        
        # We return a list of all out-of-range events we removed.
        out_of_range = []

        # Clear out expired events, decrementing totals.
        #           newest               oldest
        #             |                    |
        #             V                    V
        
        while (self.ticks[-1].dt - self.ticks[0].dt) >= self.delta:
            # popleft removes and returns ticks[0]
            popped = self.ticks.popleft()
            # Decrement totals
            for field in self.fields:
                self.totals[field] -= popped[field]
            # Add the popped element to the list of dropped events.
            out_of_range.append(popped)

        return out_of_range
            
    def average(self, field):
        assert field in self.fields
        if len(self.ticks) == 0:
            return 0.0
        else:
            return self.totals[field] / len(self.ticks)

    def get_averages(self):
        """
        Return an ndict of all our tracked averages.
        """
        out = ndict()
        for field in self.fields:
            out[field] = self.average(field)
        return out

    def assert_well_formed(self, event):
        assert isinstance(event, ndict), "Bad event in EventWindow:%s" % event
        assert event.has_key('dt'), "Missing dt in EventWindow:%s" % event
        assert isinstance(event.dt, datetime),"Bad dt in EventWindow:%s" % event
        if len(self.ticks) > 0:
            # Something is wrong if new event is older than previous.
            assert event.dt >= self.ticks[-1].dt, \
                "Events arrived out of order in EventWindow: %s -> %s" % (event, self.ticks[0])
        for field in self.fields:
            assert event.has_key(field), \
                "Event missing [%s] in EventWindow" % field 
            assert isinstance(event[field], Number), \
                "Got %s for %s in EventWindow" % (event[field], field)

if __name__ == "__main__":

    def make_event(**kwargs):
        e = ndict()
        for key, value in kwargs.iteritems():
            e[key] = value
        return e
    
    dates = date_gen(delta = timedelta(hours = 12))
    events = ( 
        make_event(
            sid = 'foo', price = random.random(), 
            dt = date, 
            type = zp.DATASOURCE_TYPE.TRADE, 
            source_id = 'ds',
            vol = i
        )
        for date, i in izip(dates, xrange(100))
    )

    gen = MovingAverageTransformGen(events, 1, ['price', 'vol'])
    

