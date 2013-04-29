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
import functools
import types
import logbook

import numpy

from numbers import Integral

import pandas as pd

from zipline.utils.data import RollingPanel
from zipline.protocol import Event

from copy import deepcopy
from datetime import datetime
from collections import deque
from abc import ABCMeta, abstractmethod

from zipline.protocol import DATASOURCE_TYPE
from zipline.gens.utils import assert_sort_unframe_protocol, hash_args
import zipline.finance.trading as trading

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
        assert isinstance(tnfm_class, (types.ObjectType, types.ClassType)), \
            "Stateful transform requires a class."
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
        log.info('Running StatefulTransform [%s]' % self.get_hash())
        for message in stream_in:
            # we only handle TRADE events.
            if (hasattr(message, 'type')
                    and message.type not in (
                        DATASOURCE_TYPE.TRADE,
                        DATASOURCE_TYPE.CUSTOM)):
                yield message
            # allow upstream generators to yield None to avoid
            # blocking.
            if message is None:
                continue

            assert_sort_unframe_protocol(message)

            tnfm_value = self.state.update(message)

            out_message = message
            out_message[self.namestring] = tnfm_value
            yield out_message

        log.info('Finished StatefulTransform [%s]' % self.get_hash())


class EventWindow(object):
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
    __metaclass__ = ABCMeta

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
        self.ticks.append(deepcopy(event))

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


class BatchTransform(object):
    """Base class for batch transforms with a trailing window of
    variable length. As opposed to pure EventWindows that get a stream
    of events and are bound to a single SID, this class creates stream
    of pandas DataFrames with each colum representing a sid.

    There are two ways to create a new batch window:
    (i) Inherit from BatchTransform and overload get_value(data).
        E.g.:
        ```
        class MyBatchTransform(BatchTransform):
            def get_value(self, data):
               # compute difference between the means of sid 0 and sid 1
               return data[0].mean() - data[1].mean()
        ```

    (ii) Use the batch_transform decorator.
        E.g.:
        ```
        @batch_transform
        def my_batch_transform(data):
            return data[0].mean() - data[1].mean()

        ```

    In your algorithm you would then have to instantiate
    this in the initialize() method:
    ```
    self.my_batch_transform = MyBatchTransform()
    ```

    To then use it, inside of the algorithm handle_data(), call the
    handle_data() of the BatchTransform and pass it the current event:
    ```
    result = self.my_batch_transform(data)
    ```

    """

    def __init__(self,
                 func=None,
                 refresh_period=0,
                 window_length=None,
                 clean_nans=True,
                 sids=None,
                 fields=None,
                 compute_only_full=True):

        """Instantiate new batch_transform object.

        :Arguments:
            func : python function <optional>
                If supplied will be called after each refresh_period
                with the data panel and all args and kwargs supplied
                to the handle_data() call.
            refresh_period : int
                Interval to wait between advances in the window.
            window_length : int
                How many days the trailing window should have.
            clean_nans : bool <default=True>
                Whether to (forward) fill in nans.
            sids : list <optional>
                Which sids to include in the moving window.  If not
                supplied sids will be extracted from incoming
                events.
            fields : list <optional>
                Which fields to include in the moving window
                (e.g. 'price'). If not supplied, fields will be
                extracted from incoming events.
            compute_only_full : bool <default=True>
                Only call the user-defined function once the window is
                full. Returns None if window is not full yet.
        """
        if func is not None:
            self.compute_transform_value = func
        else:
            self.compute_transform_value = self.get_value

        self.clean_nans = clean_nans
        self.compute_only_full = compute_only_full

        # The following logic is to allow pre-specified sid filters
        # to operate on the data, but to also allow new symbols to
        # enter the batch transform's window IFF a sid filter is not
        # specified.
        if sids is not None:
            if isinstance(sids, (basestring, Integral)):
                self.static_sids = set([sids])
            else:
                self.static_sids = set(sids)
        else:
            self.static_sids = None

        self.initial_field_names = fields
        if isinstance(self.initial_field_names, basestring):
            self.initial_field_names = [self.initial_field_names]
        self.field_names = set()

        self.refresh_period = refresh_period

        check_window_length(window_length)
        self.window_length = window_length

        self.trading_days_total = 0
        self.window = None

        self.full = False
        # Set to -inf essentially to cause update on first attempt.
        self.last_dt = pd.Timestamp('1900-1-1', tz='UTC')

        self.updated = False
        self.cached = None
        self.last_args = None
        self.last_kwargs = None

        # Data panel that provides bar information to fill in the window,
        # when no bar ticks are available from the data source generator
        # Used in universes that 'rollover', e.g. one that has a different
        # set of stocks per quarter
        self.supplemental_data = None

        self.rolling_panel = None

    def handle_data(self, data, *args, **kwargs):
        """
        Point of entry. Process an event frame.
        """
        # extract dates
        dts = [event.datetime for event in data.itervalues()]
        # we have to provide the event with a dt. This is only for
        # checking if the event is outside the window or not so a
        # couple of seconds shouldn't matter. We don't add it to
        # the data parameter, because it would mix dt with the
        # sid keys.
        event = Event()
        event.dt = max(dts)
        event.data = {k: v.__dict__ for k, v in data.iteritems()
                      # Need to check if data has a 'length' to filter
                      # out sids without trade data available.
                      # TODO: expose more of 'no trade available'
                      # functionality to zipline
                      if len(v)}

        # only modify the trailing window if this is
        # a new event. This is intended to make handle_data
        # idempotent.
        if self.last_dt < event.dt:
            self.updated = True
            self._append_to_window(event)
        else:
            self.updated = False

        # return newly computed or cached value
        return self.get_transform_value(*args, **kwargs)

    def _append_to_window(self, event):
        self.field_names = self._get_field_names(event)

        if self.static_sids is None:
            sids = set(event.data.keys())
        else:
            sids = self.static_sids

        # Create rolling panel if not existant
        if self.rolling_panel is None:
            self.rolling_panel = RollingPanel(self.window_length,
                                              self.field_names, sids)

        # Store event in rolling frame
        self.rolling_panel.add_frame(event.dt,
                                     pd.DataFrame(event.data,
                                                  index=self.field_names,
                                                  columns=sids))

        # update trading day counters
        if self.last_dt.day != event.dt.day:
            self.last_dt = event.dt
            self.trading_days_total += 1

        if self.trading_days_total >= self.window_length:
            self.full = True

    def get_transform_value(self, *args, **kwargs):
        """Call user-defined batch-transform function passing all
        arguments.

        Note that this will only call the transform if the datapanel
        has actually been updated. Otherwise, the previously, cached
        value will be returned.
        """
        if self.compute_only_full and not self.full:
            return None

        #################################################
        # Determine whether we should call the transform
        # 0. Support historical/legacy usage of '0' signaling,
        #    'update on every bar'
        if self.refresh_period == 0:
            period_signals_update = True
        else:
        # 1. Is the refresh period over?
            period_signals_update = (
                self.trading_days_total % self.refresh_period == 0)
        # 2. Have the args or kwargs been changed since last time?
        args_updated = args != self.last_args or kwargs != self.last_kwargs
        recalculate_needed = args_updated or (period_signals_update and
                                              self.updated)

        if recalculate_needed:
            self.cached = self.compute_transform_value(
                self.get_data(),
                *args,
                **kwargs
            )

        self.last_args = args
        self.last_kwargs = kwargs
        return self.cached

    def get_data(self):
        """Create a pandas.Panel (i.e. 3d DataFrame) from the
        events in the current window.

        Returns:
        The resulting panel looks like this:
        index : field_name (e.g. price)
        major axis/rows : dt
        minor axis/colums : sid
        """
        data = self.rolling_panel.get_current()

        if self.supplemental_data:
            # item will be a date stamp
            for item in data.items:
                try:
                    data[item] = self.supplemental_data[item].combine_first(
                        data[item])
                except KeyError:
                    # Only filling in data available in supplemental data.
                    pass

        if self.clean_nans:
            # Fills in gaps of missing data during transform
            # of multiple stocks. E.g. we may be missing
            # minute data because of illiquidity of one stock
            data = data.fillna(method='ffill')

        # Hold on to a reference to the data,
        # so that it's easier to find the current data when stepping
        # through with a debugger
        self._curr_data = data

        return data

    def get_value(self, *args, **kwargs):
        raise NotImplementedError(
            "Either overwrite get_value or provide a func argument.")

    def __call__(self, f):
        self.compute_transform_value = f
        return self.handle_data

    def _extract_field_names(self, event):
        # extract field names from sids (price, volume etc), make sure
        # every sid has the same fields.
        sid_keys = []
        for sid in event.data.itervalues():
            keys = set([name for name, value in sid.items()
                        if isinstance(value,
                                      (int,
                                       float,
                                       numpy.integer,
                                       numpy.float,
                                       numpy.long))
                        ])
            sid_keys.append(keys)

        # with CUSTOM data events, there may be different fields
        # per sid. So the allowable keys are the union of all events.
        union = set.union(*sid_keys)
        unwanted_fields = set(['portfolio', 'sid', 'dt', 'type',
                               'datetime', 'source_id'])
        return union - unwanted_fields

    def _get_field_names(self, event):
        if self.initial_field_names is not None:
            return self.initial_field_names
        else:
            self.latest_names = self._extract_field_names(event)
            return set.union(self.field_names, self.latest_names)


def batch_transform(func):
    """Decorator function to use instead of inheriting from BatchTransform.
    For an example on how to use this, see the doc string of BatchTransform.
    """

    @functools.wraps(func)
    def create_window(*args, **kwargs):
        # passes the user defined function to BatchTransform which it
        # will call instead of self.get_value()
        return BatchTransform(*args, func=func, **kwargs)

    return create_window
