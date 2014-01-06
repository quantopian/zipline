#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Generator versions of transforms.
"""
import logbook


from numbers import Integral

from datetime import datetime
from collections import deque
from abc import ABCMeta, abstractmethod

from six import with_metaclass

from zipline.protocol import DATASOURCE_TYPE
from zipline.gens.utils import assert_sort_unframe_protocol, hash_args
from zipline.finance import trading

log = logbook.Logger('Transform')


class UnsupportedEventWindowFlagValue(Exception):
    """
    Error state when an EventWindow option is attempted to be set
    to a value that is no longer supported by the library.

    This is to help enforce deprecation of the market_aware and delta flags,
    without completely removing it and breaking existing algorithms.
    """
    pass


class InvalidWindowLength(Exception):
    """
    Error raised when the window length is unusable.
    """
    pass


def check_window_length(window_length):
    """
    Ensure the window length provided to a transform is valid.
    """
    if window_length is None:
        raise InvalidWindowLength("window_length must be provided")
    if not isinstance(window_length, Integral):
        raise InvalidWindowLength(
            "window_length must be an integer-like number")
    if window_length == 0:
        raise InvalidWindowLength("window_length must be non-zero")
    if window_length < 0:
        raise InvalidWindowLength("window_length must be positive")


class TransformMeta(type):
    """
    Metaclass that automatically packages a class inside of
    StatefulTransform on initialization. Specifically, if Foo is a
    class with its __metaclass__ attribute set to TransformMeta, then
    calling Foo(*args, **kwargs) will return StatefulTransform(Foo,
    *args, **kwargs) instead of an instance of Foo. (Note that you can
    still recover an instance of a "raw" Foo by introspecting the
    resulting StatefulTransform's 'state' field.)
    """

    def __call__(cls, *args, **kwargs):
        return StatefulTransform(cls, *args, **kwargs)


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
        assert hasattr(tnfm_class, 'update'), \
            "Stateful transform requires the class to have an update method"

        # Create an instance of our transform class.
        if isinstance(tnfm_class, TransformMeta):
            # Classes derived TransformMeta have their __call__
            # attribute overridden.  Since this is what is usually
            # used to create an instance, we have to delegate the
            # responsibility of creating an instance to
            # TransformMeta's parent class, which is 'type'. This is
            # what is implicitly done behind the scenes by the python
            # interpreter for most classes anyway, but here we have to
            # be explicit because we've overridden the method that
            # usually resolves to our super call.
            self.state = super(TransformMeta, tnfm_class).__call__(
                *args, **kwargs)
        # Normal object instantiation.
        else:
            self.state = tnfm_class(*args, **kwargs)
        # save the window_length of the state for external access.
        self.window_length = self.state.window_length
        # Create the string associated with this generator's output.
        self.namestring = tnfm_class.__name__ + hash_args(*args, **kwargs)

    def get_hash(self):
        return self.namestring

    def transform(self, stream_in):
        return self._gen(stream_in)

    def _gen(self, stream_in):
        # IMPORTANT: Messages may contain pointers that are shared with
        # other streams.  Transforms that modify their input
        # messages should only manipulate copies.
        for message in stream_in:
            # we only handle TRADE events.
            if (hasattr(message, 'type')
                    and message.type not in (
                        DATASOURCE_TYPE.TRADE,
                        DATASOURCE_TYPE.CUSTOM)):
                yield message
                continue
            # allow upstream generators to yield None to avoid
            # blocking.
            if message is None:
                continue

            assert_sort_unframe_protocol(message)

            tnfm_value = self.state.update(message)

            out_message = message
            out_message[self.namestring] = tnfm_value
            yield out_message


class EventWindow(with_metaclass(ABCMeta)):
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

    See zipline/transforms/mavg.py and zipline/transforms/vwap.py for example
    implementations of moving average and volume-weighted average
    price.
    """
    # Mark this as an abstract base class.

    def __init__(self, market_aware=True, window_length=None, delta=None):

        check_window_length(window_length)
        self.window_length = window_length

        self.ticks = deque()

        # Only Market-aware mode is now supported.
        if not market_aware:
            raise UnsupportedEventWindowFlagValue(
                "Non-'market aware' mode is no longer supported."
            )
        if delta:
            raise UnsupportedEventWindowFlagValue(
                "delta values are no longer supported."
            )
        # Set the behavior for dropping events from the back of the
        # event window.
        self.drop_condition = self.out_of_market_window

    @abstractmethod
    def handle_add(self, event):
        raise NotImplementedError()

    @abstractmethod
    def handle_remove(self, event):
        raise NotImplementedError()

    def __len__(self):
        return len(self.ticks)

    def update(self, event):

        if (hasattr(event, 'type')
                and event.type not in (
                    DATASOURCE_TYPE.TRADE,
                    DATASOURCE_TYPE.CUSTOM)):
            return

        self.assert_well_formed(event)
        # Add new event and increment totals.
        self.ticks.append(event)

        # Subclasses should override handle_add to define behavior for
        # adding new ticks.
        self.handle_add(event)
        # Clear out any expired events.
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
        oldest_index = \
            trading.environment.trading_days.searchsorted(oldest)
        newest_index = \
            trading.environment.trading_days.searchsorted(newest)

        trading_days_between = newest_index - oldest_index

        # "Put back" a day if oldest is earlier in its day than newest,
        # reflecting the fact that we haven't yet completed the last
        # day in the window.
        if oldest.time() > newest.time():
            trading_days_between -= 1

        return trading_days_between >= self.window_length

    # All event windows expect to receive events with datetime fields
    # that arrive in sorted order.
    def assert_well_formed(self, event):
        assert isinstance(event.dt, datetime), \
            "Bad dt in EventWindow:%s" % event
        if len(self.ticks) > 0:
            # Something is wrong if new event is older than previous.
            assert event.dt >= self.ticks[-1].dt, \
                "Events arrived out of order in EventWindow: %s -> %s" % \
                (event, self.ticks[0])
