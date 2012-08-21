"""
Generator versions of transforms.
"""
import types
import pytz
import logbook

from copy import deepcopy
from datetime import datetime, timedelta
from collections import deque, defaultdict
from numbers import Number
from abc import ABCMeta, abstractmethod

from zipline import ndict
from zipline.utils.tradingcalendar import trading_days_between
from zipline.gens.utils import assert_sort_unframe_protocol, \
    assert_transform_protocol, hash_args

log = logbook.Logger('Transform')

class Passthrough(object):
    FORWARDER = True
    """
    Trivial class for forwarding events.
    """
    def __init__(self):
        pass

    def update(self, event):
        pass

# Deprecated
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
        self._copying = False

        # Create the string associated with this generator's output.
        self.namestring = tnfm_class.__name__ + hash_args(*args, **kwargs)
        log.info('StatefulTransform [%s] initialized' % self.namestring)

    def get_hash(self):
        return self.namestring

    def set_copyting(self):
        self._copying = True

    def transform(self, stream_in):
        return self._gen(stream_in)

    def _gen(self, stream_in):
        # IMPORTANT: Messages may contain pointers that are shared with
        # other streams.  Transforms that modify their input
        # messages should only manipulate copies.
        log.info('Running StatefulTransform [%s]' % self.get_hash())
        for message in stream_in:

            # allow upstream generators to yield None to avoid
            # blocking.
            if message == None:
                continue

            assert_sort_unframe_protocol(message)
            
            # Copying flag is used by merged_transforms to ensure
            # isolation of messages.
            if self._copying:
                message = deepcopy(message)
                
            # Same shared pointer issue here as above.
            tnfm_value = self.state.update(message)

            # FORWARDER flag means we want to keep all original
            # values, plus append tnfm_id and tnfm_value. Used for
            # preserving the original event fields when our output
            # will be fed into a merge. Currently only Passthrough
            # uses this flag.
            if self.forward_all:
                out_message = message
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
            # tnfm_value) into a single field. This mode is used by
            # the sequential_transforms composite.
            elif self.append_value:
                out_message = message
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
                out_message.dt = message.dt
                yield out_message

        log.info('Finished StatefulTransform [%s]' % self.get_hash())
class EventWindow:
    """
    Abstract base class for transform classes that calculate iterative
    metrics on events within a given timedelta.  Maintains a list of
    events that are within a certain timedelta of the most recent
    tick.  Calls self.handle_add(event) for each event added to the
    window.  Calls self.handle_remove(event) for each event removed
    from the window.  Subclass these methods along with init(*args,
    **kwargs) to calculate metrics over the window.

    If the market_aware flag is True, the EventWindow drops old events
    based on the number of elapsed trading days between newest and oldest.
    Otherwise old events are dropped based on a raw timedelta.

    See zipline/gens/mavg.py and zipline/gens/vwap.py for example
    implementations of moving average and volume-weighted average
    price.
    """
    # Mark this as an abstract base class.
    __metaclass__ = ABCMeta

    def __init__(self, market_aware, days = None, delta = None):

        self.market_aware = market_aware
        self.days = days
        self.delta = delta

        self.ticks  = deque()

        # Market-aware mode only works with full-day windows.
        if self.market_aware:
            assert self.days and not self.delta,\
                "Market-aware mode only works with full-day windows."

        # Non-market-aware mode requires a timedelta.
        else:
            assert self.delta and not self.days, \
                "Non-market-aware mode requires a timedelta."

        # Set the behavior for dropping events from the back of the
        # event window.
        if self.market_aware:
            self.drop_condition = self.out_of_market_window
        else:
            self.drop_condition = self.out_of_delta

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

        # Subclasses should override handle_add to define behavior for
        # adding new ticks.
        self.handle_add(event)

        # Clear out any expired events. drop_condition changes depending
        # on whether or not we are running in market_aware mode.
        #
        #                              oldest               newest
        #                                |                    |
        #                                V                    V
        while self.drop_condition(self.ticks[0].dt, self.ticks[-1].dt):

            # popleft removes and returns the oldest tick in self.ticks
            popped = self.ticks.popleft()

            # Subclasses should override handle_remove to define
            # behavior for removing ticks.
            self.handle_remove(popped)

    def out_of_market_window(self, oldest, newest):
        return trading_days_between(oldest, newest) >= self.days

    def out_of_delta(self, oldest, newest):
        return (newest - oldest) >= self.delta

    # All event windows expect to receive events with datetime fields
    # that arrive in sorted order.
    def assert_well_formed(self, event):
        assert isinstance(event, ndict), "Bad event in EventWindow:%s" % event
        assert event.has_key('dt'), "Missing dt in EventWindow:%s" % event
        assert isinstance(event.dt, datetime),"Bad dt in EventWindow:%s" % event
        if len(self.ticks) > 0:
            # Something is wrong if new event is older than previous.
            assert event.dt >= self.ticks[-1].dt, \
                "Events arrived out of order in EventWindow: %s -> %s" % (event, self.ticks[0])
