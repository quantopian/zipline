"""
Generator versions of transforms.
"""
import types

from copy import deepcopy
from datetime import datetime
from collections import deque, defaultdict
from numbers import Number
from abc import ABCMeta, abstractmethod

from zipline import ndict
from zipline.gens.utils import assert_sort_unframe_protocol, \
    assert_transform_protocol, hash_args

class Passthrough(object):
    FORWARDER = True
    """
    Trivial class for forwarding events.
    """
    def __init__(self):
        pass

    def update(self, event):
        pass

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

class StatefulTransform(object):
    """
    Generic transform generator that takes each message from an
    in-stream and passes it to a state object.  For each call to
    update, the state class must produce a message to be fed
    downstream. Any transform class with the FORWARDER class variable
    set to true will forward all fields in the original message.
    Otherwise only dt, tnfm_id, and tnfm_value are forwarded.
    """
    def __init__(self, tnfm_class, *args, **kwargs):
        assert isinstance(tnfm_class, (types.ObjectType, types.ClassType)), \
        "Stateful transform requires a class."
        assert tnfm_class.__dict__.has_key('update'), \
        "Stateful transform requires the class to have an update method"

        self.forward_all = tnfm_class.__dict__.get('FORWARDER', False)
        self.update_in_place = tnfm_class.__dict__.get('UPDATER', False)
        self.append_value = tnfm_class.__dict__.get('APPENDER', False)

        # You only one special behavior mode can be set.
        assert sum(map(int, [self.forward_all, 
                             self.update_in_place, 
                             self.append_value])) <= 1

        # Create an instance of our transform class.
        self.state = tnfm_class(*args, **kwargs)

        # Create the string associated with this generator's output.
        self.namestring = tnfm_class.__name__ + hash_args(*args, **kwargs)

    def get_hash(self):
        return self.namestring

    def transform(self, stream_in):
        return self._gen(stream_in)

    def _gen(self, stream_in):
        # IMPORTANT: Messages may contain pointers that are shared with
        # other streams, so we only manipulate copies.
        
        for message in stream_in:

            # allow upstream generators to yield None to avoid
            # blocking.
            if message == None:
                continue
            
            #TODO: refactor this to avoid unnecessary copying.

            assert_sort_unframe_protocol(message)
            message_copy = deepcopy(message)

            # Same shared pointer issue here as above.
            tnfm_value = self.state.update(deepcopy(message_copy))

            # FORWARDER flag means we want to keep all original
            # values, plus append tnfm_id and tnfm_value. Used for
            # preserving the original event fields when our output
            # will be fed into a merge.
            if self.forward_all:
                out_message = message_copy
                out_message.tnfm_id = self.namestring
                out_message.tnfm_value = tnfm_value
                yield out_message

            # UPDATER flag should be used for transforms that
            # side-effectfully modify the event they are passed.
            # Updated messages are passed along exactly as they are
            # returned to use by our state class. Useful for chaining
            # specific transforms that won't be fed to a merge.  (See
            # the implementation of TradeSimulationClient for example
            # usage of this flag with PerformanceTracker and
            # TransactionSimulator.
            elif self.update_in_place:
                yield tnfm_value
                
            # APPENDER flag should be used to add a single new
            # key-value pair to the event. The new key is this
            # transform's namestring, and it's value is the value
            # returned by state.update(event). This is almost
            # identical to the behavior of FORWARDER, except we
            # compress the two calculated values (tnfm_id, and
            # tnfm_value) into a single field.
            elif self.append_value:
                out_message = message_copy
                out_message[self.namestring] = tnfm_value
                yield out_message

            # If no flags are set, we create a new message containing
            # just the tnfm_id, the event's datetime, and the
            # calculated tnfm_value. This is the default behavior for
            # a transform being fed into a merge.
            else:
                out_message = ndict()
                out_message.tnfm_id = self.namestring
                out_message.tnfm_value = tnfm_value
                out_message.dt = message_copy.dt
                yield out_message

class MovingAverage(object):
    """
    Class that maintains a dictionary from sids to EventWindows
    Upon receipt of each message we update the
    corresponding window and return the calculated average.
    """
    FORWARDER = False

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

        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_averages()

class EventWindow:
    """
    Abstract base class for transform classes that calculate iterative
    metrics on events within a given timedelta.  Maintains a list of
    events that are within a certain timedelta of the most recent
    tick.  Calls self.handle_add(event) for each event added to the
    window.  Calls self.handle_remove(event) for each event removed
    from the window.  Subclass these methods along with init(*args,
    **kwargs) to calculate metrics over the window.  

    See zipline/gens/mavg.py and zipline/gens/vwap.py for example
    implementations of moving average and volume-weighted average
    price.
    """
    # Mark this as an abstract base class.
    __metaclass__ = ABCMeta

    def __init__(self, delta, *args, **kwargs):
        self.ticks  = deque()
        self.delta  = delta
        self.init(*args, **kwargs)
        
    @abstractmethod
    def init(self):
        raise NotImplementedError()

    @abstractmethod
    def handle_add(self, event):
        raise NotImplementedError()

    @abstractmethod
    def handle_remove(self, event):
        raise NotImplementedError()

    def __len__(self):
        return len(self.ticks)

    def update(self, event):
        self.assert_well_formed(event)
        # Add new event and increment totals.
        self.ticks.append(event)
        self.handle_add(event)

        # Clear out expired event.
        #
        #           newest               oldest
        #             |                    |
        #             V                    V
        while (self.ticks[-1].dt - self.ticks[0].dt) > self.delta:
            # popleft removes and returns the oldest tick in self.ticks
            popped = self.ticks.popleft()
            # Subclasses should override handle_remove to define
            # behavior for removing ticks.
            self.handle_remove(popped)

    def assert_well_formed(self, event):
        assert isinstance(event, ndict), "Bad event in EventWindow:%s" % event
        assert event.has_key('dt'), "Missing dt in EventWindow:%s" % event
        assert isinstance(event.dt, datetime),"Bad dt in EventWindow:%s" % event
        if len(self.ticks) > 0:
            # Something is wrong if new event is older than previous.
            assert event.dt >= self.ticks[-1].dt, \
                "Events arrived out of order in EventWindow: %s -> %s" % (event, self.ticks[0])
