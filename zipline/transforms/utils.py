#
# Copyright 2012 Quantopian, Inc.
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
import types
import logbook

from copy import deepcopy
from datetime import datetime
from collections import deque
from abc import ABCMeta, abstractmethod

import pandas as pd

from zipline import ndict
from zipline.utils.tradingcalendar import non_trading_days
from zipline.gens.utils import assert_sort_unframe_protocol, hash_args

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


class TransformMeta(type):
    """
    Metaclass that automatically packages a class inside of
    StatefulTransform on initialization. Specifically, if Foo is a
    class with its __metaclass__ attribute set to TransformMeta, then
    calling Foo(*args, **kwargs) will return StatefulTransform(Foo,
    *args, **kwargs) instead of an instance of Foo. (Note that you can
    still recover an instance of a "raw" Foo by introspecting the
    resulting StatefulTransform's 'state' field.
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

        # Flag set inside the Passthrough transform class to signify special
        # behavior if we are being fed to merged_transforms.
        self.passthrough = hasattr(tnfm_class, 'PASSTHROUGH')

        # Flags specifying how to append the calculated value.
        # Merged is the default for ease of testing, but we use sequential
        # in production.
        self.sequential = False
        self.merged = True

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

            # allow upstream generators to yield None to avoid
            # blocking.
            if message is None:
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

    def __init__(self, market_aware=True, days=None, delta=None):

        self.market_aware = market_aware
        self.days = days
        self.delta = delta

        self.ticks = deque()

        # Market-aware mode only works with full-day windows.
        if self.market_aware:
            assert self.days and self.delta is None, \
                "Market-aware mode only works with full-day windows."
            self.all_holidays = deque(non_trading_days)
            self.cur_holidays = deque()

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
        self.ticks.append(deepcopy(event))

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
        assert 'dt' in event, "Missing dt in EventWindow:%s" % event
        assert isinstance(event.dt, datetime), \
            "Bad dt in EventWindow:%s" % event
        if len(self.ticks) > 0:
            # Something is wrong if new event is older than previous.
            assert event.dt >= self.ticks[-1].dt, \
                "Events arrived out of order in EventWindow: %s -> %s" % \
                (event, self.ticks[0])


class BatchTransform(EventWindow):
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

    In you algorithm you would then have to instantiate
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
                 refresh_period=None,
                 market_aware=True,
                 delta=None,
                 days=None):

        super(BatchTransform, self).__init__(market_aware,
                                             days=days, delta=delta)

        if func is not None:
            self.compute_transform_value = func
        else:
            self.compute_transform_value = self.get_value

        self.refresh_period = refresh_period
        self.days = days

        self.full = False
        self.last_refresh = None

        self.updated = False
        self.data = None

    def handle_data(self, data, *args, **kwargs):
        """
        New method to handle a data frame as sent to the algorithm's
        handle_data method.
        """
        # extract dates
        #dts = [data[sid].datetime for sid in self.sids]
        dts = [event.datetime for event in data.itervalues()]
        # we have to provide the event with a dt. This is only for
        # checking if the event is outside the window or not so a
        # couple of seconds shouldn't matter. We don't add it to
        # the data parameter, because it would mix dt with the
        # sid keys.
        event = ndict()
        event.dt = max(dts)
        event.data = data

        # append data frame to window. update() will call handle_add() and
        # handle_remove() appropriately
        self.update(event)

        # return newly computed or cached value
        return self.get_transform_value(*args, **kwargs)

    def handle_add(self, event):
        if not self.last_refresh:
            self.last_refresh = event.dt
            return

        age = event.dt - self.last_refresh
        if age.days >= self.refresh_period:
            # Create a pandas.Panel (i.e. 3d DataFrame) from the
            # events in the current window.
            #
            # The resulting panel looks like this:
            # index : field_name (e.g. price)
            # major axis/rows : dt
            # minor axis/colums : sid
            #
            # This Panel data structure ultimately gets passed to the
            # user-overloaded get_value() method.
            #
            # self.ticks contains ndicts with data, dt keys.
            # event parameter is an ndict with data, dt keys.
            fields = {}
            for field_name in ['price', 'volume']:
                sids = self.ticks[0].data.keys()
                # Skip non-existant fields
                if field_name not in self.ticks[0].data[sids[0]]:
                    continue

                values_per_sid = {}

                for sid in sids:
                    values_per_sid[sid] = pd.Series(
                        {tick.data[sid].dt: tick.data[sid][field_name]
                         for tick in self.ticks}
                    )

                # concatenate different sids into one df
                fields[field_name] = pd.DataFrame.from_dict(values_per_sid)

            self.data = pd.Panel.from_dict(fields, orient='items')

            self.updated = True
            self.last_refresh = event.dt
        else:
            self.updated = False

    def handle_remove(self, event):
        # since an event is expiring, we know the window is full
        self.full = True

    def get_value(self, *args, **kwargs):
        raise NotImplementedError(
            "Either overwrite get_value or provide a func argument.")

    def get_transform_value(self, *args, **kwargs):
        if self.data is None:
            return None

        if self.updated:
            self.cached = self.compute_transform_value(self.data,
                                                       *args, **kwargs)

        return self.cached

    def __call__(self, f):
        self.compute_transform_value = f
        return self.handle_data


def batch_transform(func):
    """Decorator function to use instead of inheriting from BatchTransform.
    For an example on how to use this, see the doc string of BatchTransform.
    """

    def create_window(*args, **kwargs):
        # passes the user defined function to BatchTransform which it
        # will call instead of self.get_value()
        return BatchTransform(*args, func=func, **kwargs)

    return create_window
