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
from zipline.utils.tradingcalendar import non_trading_days
from zipline.gens.utils import assert_sort_unframe_protocol, \
    assert_transform_protocol, hash_args

log = logbook.Logger('Transform')

class Passthrough(object):
    PASSTHROUGH = True
    """
    Trivial class for forwarding events.
    """
    def __init__(self):
        pass

    def update(self, event):
        pass

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

        # Flag set inside the Passthrough transform class to signify special
        # behavior if we are being fed to merged_transforms.
        self.passthrough = tnfm_class.__dict__.get('PASSTHROUGH', False)
        
        # Flags specifying how to append the calculated value.
        # Merged is the default for ease of testing, but we use sequential
        # in production.
        self.sequential = False
        self.merged = True
        
        # Create an instance of our transform class.
        self.state = tnfm_class(*args, **kwargs)

        # Create the string associated with this generator's output.
        self.namestring = tnfm_class.__name__ + hash_args(*args, **kwargs)
        log.info('StatefulTransform [%s] initialized' % self.namestring)

    def get_hash(self):
        return self.namestring

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
            
            # This flag is set by by merged_transforms to ensure
            # isolation of messages.
            if self.merged:
                message = deepcopy(message)
            
            tnfm_value = self.state.update(message)

            # PASSTHROUGH flag means we want to keep all original
            # values, plus append tnfm_id and tnfm_value. Used for
            # preserving the original event fields when our output
            # will be fed into a merge. Currently only Passthrough
            # uses this flag.
            if self.passthrough and self.merged:
                out_message = message
                out_message.tnfm_id = self.namestring
                out_message.tnfm_value = tnfm_value
                yield out_message

            # If the merged flag is set, we create a new message
            # containing just the tnfm_id, the event's datetime, and
            # the calculated tnfm_value. This is the default behavior
            # for a non-passthrough transform being fed into a merge.
            elif self.merged:
                out_message = ndict()
                out_message.tnfm_id = self.namestring
                out_message.tnfm_value = tnfm_value
                out_message.dt = message.dt
                yield out_message
            
            # Sequential flag should be used to add a single new
            # key-value pair to the event. The new key is this
            # transform's namestring, and its value is the value
            # returned by state.update(event). This is almost
            # identical to the behavior of FORWARDER, except we
            # compress the two calculated values (tnfm_id, and
            # tnfm_value) into a single field. This mode is used by
            # the sequential_transforms composite and is the default
            # if no behavior is specified by the internal state class.
            elif self.sequential:
                out_message = message
                out_message[self.namestring] = tnfm_value
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
            assert self.days and self.delta == None,\
                "Market-aware mode only works with full-day windows."
            self.all_holidays = deque(non_trading_days)
            self.cur_holidays = deque()
            # Keeping a copy of days as a timedelta makes it easier
            # to track holidays.
            self.delta = timedelta(days=self.days)

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
        
        if self.market_aware:
            self.add_new_holidays(event.dt)

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
            
    def add_new_holidays(self, newest):
        # Add to our tracked window any untracked holidays that are
        # older than our newest event. (newest should always be
        # self.ticks[-1])
        while len(self.all_holidays) > 0 and self.all_holidays[0] <= newest:
            self.cur_holidays.append(self.all_holidays.popleft())

    def drop_old_holidays(self, oldest):
        # Drop from our tracked window any holidays that are older
        # than our oldest tracked event. (oldest should always
        # be self.ticks[0])
        while len(self.cur_holidays) > 0 and self.cur_holidays[0] < oldest:
            self.cur_holidays.popleft()

    def out_of_market_window(self, oldest, newest):
        self.drop_old_holidays(oldest)
        calendar_dates_between = (newest.date() - oldest.date()).days
        holidays_between = len(self.cur_holidays)
        trading_days_between = calendar_dates_between - holidays_between
        
        # "Put back" a day if oldest is earlier in its day than newest,
        # reflecting the fact that we haven't yet completed the last
        # day in the window.
        if oldest.time() > newest.time():
            trading_days_between -= 1
        return trading_days_between >= self.days

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
