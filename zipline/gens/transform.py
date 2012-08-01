"""
Generator versions of transforms.
"""
import types

from datetime import datetime
from collections import deque, defaultdict
from numbers import Number

from zipline import ndict
from zipline.gens.utils import assert_sort_unframe_protocol, \
    assert_transform_protocol, hash_args

class Passthrough(object):
    """
    Trivial class for forwarding events.
    """
    def __init__(self):
        pass

    def update(self, event):
        assert isinstance(event, ndict),"Bad event in Passthrough: %s" % event
        assert event.has_key('sid'), "No sid in Passthrough: %s" % event
        assert event.has_key('dt'), "No dt in Passthorughz: %s" % event
        return event

def functional_transform(stream_in, func, *args, **kwargs):
    """
    Generic transform generator that takes each message from an in-stream
    and yields the output of a function on that message. Not sure how
    useful this will be in reality, but good for testing.
    """
    assert isinstance(func, types.FunctionType), \
        "Functional"
    namestring = func.__name__ + hash_args(*args, **kwargs)

    for message in stream_in:
        assert_sort_unframe_protocol(message)
        out_value = func(message, *args, **kwargs)
        assert_transform_protocol(out_value)
        yield(namestring, out_value)

def stateful_transform(stream_in, tnfm_class, *args, **kwargs):
    """
    Generic transform generator that takes each message from an in-stream
    and passes it to a state class.  For each call to update, the state
    class must produce a message to be fed downstream.
    """
    
    assert isinstance(tnfm_class, (types.ObjectType, types.ClassType)), \
        "Stateful transform requires a class."
    assert tnfm_class.__dict__.has_key('update'), \
        "Stateful transform requires the class to have an update method"

    # Create an instance of our transform class.
    state = tnfm_class(*args, **kwargs)

    # Generate the string associated with this generator's output.
    namestring = tnfm_class.__name__ + hash_args(*args, **kwargs)

    for message in stream_in:
        assert_sort_unframe_protocol(message)
        out_value = state.update(message)
        assert_transform_protocol(out_value)
        yield (namestring, out_value)

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
        assert event.has_key('dt'), "No dt in MovingAverage: %s" % event

        output = ndict({'sid': event.sid, 'dt': event.dt})
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        averages =  window.get_averages()

        # Return the calculated averages along with
        output.merge(averages)
        return output

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
        # out.ticks = len(self.ticks)
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
