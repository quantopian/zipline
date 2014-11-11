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
import logbook

import numpy

from numbers import Integral

import pandas as pd

from six import (
    string_types,
    itervalues,
    iteritems
)

from zipline.utils.data import MutableIndexRollingPanel
from zipline.protocol import Event

from zipline.finance import trading

from . utils import check_window_length

log = logbook.Logger('BatchTransform')
func_map = {'open_price': 'first',
            'close_price': 'last',
            'low': 'min',
            'high': 'max',
            'volume': 'sum'
            }


def get_sample_func(item):
    if item in func_map:
        return func_map[item]
    else:
        return 'last'


def downsample_panel(minute_rp, daily_rp, mkt_close):
    """
    @minute_rp is a rolling panel, which should have minutely rows
    @daily_rp is a rolling panel, which should have daily rows
    @dt is the timestamp to use when adding a frame to daily_rp

    Using the history in minute_rp, a new daily bar is created by
    downsampling. The data from the daily bar is then added to the
    daily rolling panel using add_frame.
    """

    cur_panel = minute_rp.get_current()
    sids = minute_rp.minor_axis
    day_frame = pd.DataFrame(columns=sids, index=cur_panel.items)
    dt1 = trading.environment.normalize_date(mkt_close)
    dt2 = trading.environment.next_trading_day(mkt_close)
    by_close = functools.partial(get_date, mkt_close, dt1, dt2)
    for item in minute_rp.items:
        frame = cur_panel[item]
        func = get_sample_func(item)
        # group by trading day, using the market close of the current
        # day. If events occurred after the last close (yesterday) but
        # before today's close, group them into today.
        dframe = frame.groupby(lambda d: by_close(d)).agg(func)
        for stock in sids:
            day_frame[stock][item] = dframe[stock].ix[dt1]
    # store the frame at midnight instead of the close
    daily_rp.add_frame(dt1, day_frame)


def get_date(mkt_close, d1, d2, d):
    if d > mkt_close:
        return d2
    else:
        return d1


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
                 compute_only_full=True,
                 bars='daily',
                 downsample=False):

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
            downsample : bool <default=False>
                If true, downsample bars to daily bars. Otherwise, do nothing.
        """
        if func is not None:
            self.compute_transform_value = func
        else:
            self.compute_transform_value = self.get_value

        self.clean_nans = clean_nans
        self.compute_only_full = compute_only_full
        # no need to down sample if the bars are already daily
        self.downsample = downsample and (bars == 'minute')

        # How many bars are in a day
        self.bars = bars
        if self.bars == 'daily':
            self.bars_in_day = 1
        elif self.bars == 'minute':
            self.bars_in_day = int(6.5 * 60)
        else:
            raise ValueError('%s bars not understood.' % self.bars)

        # The following logic is to allow pre-specified sid filters
        # to operate on the data, but to also allow new symbols to
        # enter the batch transform's window IFF a sid filter is not
        # specified.
        if sids is not None:
            if isinstance(sids, (string_types, Integral)):
                self.static_sids = set([sids])
            else:
                self.static_sids = set(sids)
        else:
            self.static_sids = None

        self.initial_field_names = fields
        if isinstance(self.initial_field_names, string_types):
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
        self.daily_rolling_panel = None

    def handle_data(self, data, *args, **kwargs):
        """
        Point of entry. Process an event frame.
        """
        # extract dates
        dts = [event.dt for event in itervalues(data._data)]
        # we have to provide the event with a dt. This is only for
        # checking if the event is outside the window or not so a
        # couple of seconds shouldn't matter. We don't add it to
        # the data parameter, because it would mix dt with the
        # sid keys.
        event = Event()
        event.dt = max(dts)
        event.data = {k: v.__dict__ for k, v in iteritems(data._data)
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

    def _init_panels(self, sids):
        if self.downsample:
            self.rolling_panel = MutableIndexRollingPanel(
                self.bars_in_day,
                self.field_names,
                sids,
            )

            self.daily_rolling_panel = MutableIndexRollingPanel(
                self.window_length,
                self.field_names,
                sids,
            )
        else:
            self.rolling_panel = MutableIndexRollingPanel(
                self.window_length * self.bars_in_day,
                self.field_names,
                sids,
            )

    def _append_to_window(self, event):
        self.field_names = self._get_field_names(event)

        if self.static_sids is None:
            sids = set(event.data.keys())
        else:
            sids = self.static_sids

        # the panel sent to the transform code will have
        # columns masked with this set of sids. This is how
        # we guarantee that all (and only) the sids sent to the
        # algorithm's handle_data and passed to the batch
        # transform. See the get_data method to see it applied.
        # N.B. that the underlying panel grows monotonically
        # if the set of sids changes over time.
        self.latest_sids = sids
        # Create rolling panel if not existant
        if self.rolling_panel is None:
            self._init_panels(sids)

        # Store event in rolling frame
        self.rolling_panel.add_frame(event.dt,
                                     pd.DataFrame(event.data,
                                                  index=self.field_names,
                                                  columns=sids))

        # update trading day counters
        # we may get events from non-trading sources which occurr on
        # non-trading days. The book-keeping for market close and
        # trading day counting should only consider trading days.
        if trading.environment.is_trading_day(event.dt):
            _, mkt_close = trading.environment.get_open_and_close(event.dt)
            if self.bars == 'daily':
                # Daily bars have their dt set to midnight.
                mkt_close = trading.environment.normalize_date(mkt_close)
            if event.dt == mkt_close:
                if self.downsample:
                    downsample_panel(self.rolling_panel,
                                     self.daily_rolling_panel,
                                     mkt_close
                                     )
                self.trading_days_total += 1
            self.mkt_close = mkt_close

        self.last_dt = event.dt

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
        # 3. Is this a downsampled batch, and is the last event mkt close?
        downsample_ready = not self.downsample or \
            self.last_dt == self.mkt_close

        recalculate_needed = downsample_ready and \
            (args_updated or (period_signals_update and self.updated))
        ###################################################

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
        if self.downsample:
            data = self.daily_rolling_panel.get_current()
        else:
            data = self.rolling_panel.get_current()

        if self.supplemental_data is not None:
            for item in data.items:
                if item not in self.supplemental_data.items:
                    continue
                for dt in data.major_axis:
                    try:
                        supplemental_for_dt = self.supplemental_data.ix[
                            item, dt, :]
                    except KeyError:
                        # Only filling in data available in supplemental data.
                        supplemental_for_dt = None

                    if supplemental_for_dt is not None:
                        data[item].ix[dt] = \
                            supplemental_for_dt.combine_first(
                                data[item].ix[dt])

        # screen out sids no longer in the multiverse
        data = data.ix[:, :, self.latest_sids]
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
        for sid in itervalues(event.data):
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
        unwanted_fields = {
            'portfolio',
            'sid',
            'dt',
            'type',
            'source_id',
            '_initial_len',
        }
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
