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
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd


def _ensure_index(x):
    if not isinstance(x, pd.Index):
        x = pd.Index(sorted(x))

    return x


class RollingPanel(object):
    """
    Preallocation strategies for rolling window over expanding data set

    Restrictions: major_axis can only be a DatetimeIndex for now
    """

    def __init__(self,
                 window,
                 items,
                 sids,
                 cap_multiple=2,
                 dtype=np.float64,
                 initial_dates=None):

        self._pos = window
        self._window = window

        self.items = _ensure_index(items)
        self.minor_axis = _ensure_index(sids)

        self.cap_multiple = cap_multiple

        self.dtype = dtype
        if initial_dates is None:
            self.date_buf = np.empty(self.cap, dtype='M8[ns]') * pd.NaT
        elif len(initial_dates) != window:
            raise ValueError('initial_dates must be of length window')
        else:
            self.date_buf = np.hstack(
                (
                    initial_dates,
                    np.empty(
                        window * (cap_multiple - 1),
                        dtype='datetime64[ns]',
                    ),
                ),
            )

        self.buffer = self._create_buffer()

    @property
    def cap(self):
        return self.cap_multiple * self._window

    @property
    def _start_index(self):
        return self._pos - self._window

    @property
    def start_date(self):
        return self.date_buf[self._start_index]

    def oldest_frame(self, raw=False):
        """
        Get the oldest frame in the panel.
        """
        if raw:
            return self.buffer.values[:, self._start_index, :]
        return self.buffer.iloc[:, self._start_index, :]

    def set_minor_axis(self, minor_axis):
        self.minor_axis = _ensure_index(minor_axis)
        self.buffer = self.buffer.reindex(minor_axis=self.minor_axis)

    def set_items(self, items):
        self.items = _ensure_index(items)
        self.buffer = self.buffer.reindex(items=self.items)

    def _create_buffer(self):
        panel = pd.Panel(
            items=self.items,
            minor_axis=self.minor_axis,
            major_axis=range(self.cap),
            dtype=self.dtype,
        )
        return panel

    def extend_back(self, missing_dts):
        """
        Resizes the buffer to hold a new window with a new cap_multiple.
        If cap_multiple is None, then the old cap_multiple is used.
        """
        delta = len(missing_dts)

        if not delta:
            raise ValueError(
                'missing_dts must be a non-empty index',
            )

        self._window += delta

        self._pos += delta

        self.date_buf = self.date_buf.copy()
        self.date_buf.resize(self.cap)
        self.date_buf = np.roll(self.date_buf, delta)

        old_vals = self.buffer.values
        shape = old_vals.shape
        nan_arr = np.empty((shape[0], delta, shape[2]))
        nan_arr.fill(np.nan)

        new_vals = np.column_stack(
            (nan_arr,
             old_vals,
             np.empty((shape[0], delta * (self.cap_multiple - 1), shape[2]))),
        )

        self.buffer = pd.Panel(
            data=new_vals,
            items=self.items,
            minor_axis=self.minor_axis,
            major_axis=np.arange(self.cap),
            dtype=self.dtype,
        )

        # Fill the delta with the dates we calculated.
        where = slice(self._start_index, self._start_index + delta)
        self.date_buf[where] = missing_dts

    def add_frame(self, tick, frame, minor_axis=None, items=None):
        """
        """
        if self._pos == self.cap:
            self._roll_data()

        values = frame
        if isinstance(frame, pd.DataFrame):
            values = frame.values

        self.buffer.values[:, self._pos, :] = values.astype(self.dtype)
        self.date_buf[self._pos] = tick

        self._pos += 1

    def get_current(self, item=None, raw=False, start=None, end=None):
        """
        Get a Panel that is the current data in view. It is not safe to persist
        these objects because internal data might change
        """
        item_indexer = slice(None)
        if item:
            item_indexer = self.items.get_loc(item)

        start_index = self._start_index
        end_index = self._pos

        # get inital date window
        where = slice(start_index, end_index)
        current_dates = self.date_buf[where]

        def convert_datelike_to_long(dt):
            if isinstance(dt, pd.Timestamp):
                return dt.asm8
            if isinstance(dt, datetime.datetime):
                return np.datetime64(dt)
            return dt

        # constrict further by date
        if start:
            start = convert_datelike_to_long(start)
            start_index += current_dates.searchsorted(start)

        if end:
            end = convert_datelike_to_long(end)
            _end = current_dates.searchsorted(end, 'right')
            end_index -= len(current_dates) - _end

        where = slice(start_index, end_index)

        values = self.buffer.values[item_indexer, where, :]
        current_dates = self.date_buf[where]

        if raw:
            # return copy so we can change it without side effects here
            return values.copy()

        major_axis = pd.DatetimeIndex(deepcopy(current_dates), tz='utc')
        if values.ndim == 3:
            return pd.Panel(values, self.items, major_axis, self.minor_axis,
                            dtype=self.dtype)

        elif values.ndim == 2:
            return pd.DataFrame(values, major_axis, self.minor_axis,
                                dtype=self.dtype)

    def set_current(self, panel):
        """
        Set the values stored in our current in-view data to be values of the
        passed panel.  The passed panel must have the same indices as the panel
        that would be returned by self.get_current.
        """
        where = slice(self._start_index, self._pos)
        self.buffer.values[:, where, :] = panel.values

    def current_dates(self):
        where = slice(self._start_index, self._pos)
        return pd.DatetimeIndex(deepcopy(self.date_buf[where]), tz='utc')

    def _roll_data(self):
        """
        Roll window worth of data up to position zero.
        Save the effort of having to expensively roll at each iteration
        """

        self.buffer.values[:, :self._window, :] = \
            self.buffer.values[:, -self._window:, :]
        self.date_buf[:self._window] = self.date_buf[-self._window:]
        self._pos = self._window

    @property
    def window_length(self):
        return self._window


class MutableIndexRollingPanel(object):
    """
    A version of RollingPanel that exists for backwards compatibility with
    batch_transform. This is a copy to allow behavior of RollingPanel to drift
    away from this without breaking this class.

    This code should be considered frozen, and should not be used in the
    future. Instead, see RollingPanel.
    """
    def __init__(self, window, items, sids, cap_multiple=2, dtype=np.float64):

        self._pos = 0
        self._window = window

        self.items = _ensure_index(items)
        self.minor_axis = _ensure_index(sids)

        self.cap_multiple = cap_multiple
        self.cap = cap_multiple * window

        self.dtype = dtype
        self.date_buf = np.empty(self.cap, dtype='M8[ns]')

        self.buffer = self._create_buffer()

    def _oldest_frame_idx(self):
        return max(self._pos - self._window, 0)

    def oldest_frame(self, raw=False):
        """
        Get the oldest frame in the panel.
        """
        if raw:
            return self.buffer.values[:, self._oldest_frame_idx(), :]
        return self.buffer.iloc[:, self._oldest_frame_idx(), :]

    def set_sids(self, sids):
        self.minor_axis = _ensure_index(sids)
        self.buffer = self.buffer.reindex(minor_axis=self.minor_axis)

    def _create_buffer(self):
        panel = pd.Panel(
            items=self.items,
            minor_axis=self.minor_axis,
            major_axis=range(self.cap),
            dtype=self.dtype,
        )
        return panel

    def get_current(self):
        """
        Get a Panel that is the current data in view. It is not safe to persist
        these objects because internal data might change
        """

        where = slice(self._oldest_frame_idx(), self._pos)
        major_axis = pd.DatetimeIndex(deepcopy(self.date_buf[where]), tz='utc')
        return pd.Panel(self.buffer.values[:, where, :], self.items,
                        major_axis, self.minor_axis, dtype=self.dtype)

    def set_current(self, panel):
        """
        Set the values stored in our current in-view data to be values of the
        passed panel.  The passed panel must have the same indices as the panel
        that would be returned by self.get_current.
        """
        where = slice(self._oldest_frame_idx(), self._pos)
        self.buffer.values[:, where, :] = panel.values

    def current_dates(self):
        where = slice(self._oldest_frame_idx(), self._pos)
        return pd.DatetimeIndex(deepcopy(self.date_buf[where]), tz='utc')

    def _roll_data(self):
        """
        Roll window worth of data up to position zero.
        Save the effort of having to expensively roll at each iteration
        """

        self.buffer.values[:, :self._window, :] = \
            self.buffer.values[:, -self._window:, :]
        self.date_buf[:self._window] = self.date_buf[-self._window:]
        self._pos = self._window

    def add_frame(self, tick, frame, minor_axis=None, items=None):
        """
        """
        if self._pos == self.cap:
            self._roll_data()

        if isinstance(frame, pd.DataFrame):
            minor_axis = frame.columns
            items = frame.index

        if set(minor_axis).difference(set(self.minor_axis)) or \
                set(items).difference(set(self.items)):
            self._update_buffer(frame)

        vals = frame.T.astype(self.dtype)
        self.buffer.loc[:, self._pos, :] = vals
        self.date_buf[self._pos] = tick

        self._pos += 1

    def _update_buffer(self, frame):

        # Get current frame as we only need to care about the data that is in
        # the active window
        old_buffer = self.get_current()
        if self._pos >= self._window:
            # Don't count the last major_axis entry if we're past our window,
            # since it's about to roll off the end of the panel.
            old_buffer = old_buffer.iloc[:, 1:, :]

        nans = pd.isnull(old_buffer)

        # Find minor_axes that have only nans
        # Note that minor is axis 2
        non_nan_cols = set(old_buffer.minor_axis[~np.all(nans, axis=(0, 1))])
        # Determine new columns to be added
        new_cols = set(frame.columns).difference(non_nan_cols)
        # Update internal minor axis
        self.minor_axis = _ensure_index(new_cols.union(non_nan_cols))

        # Same for items (fields)
        # Find items axes that have only nans
        # Note that items is axis 0
        non_nan_items = set(old_buffer.items[~np.all(nans, axis=(1, 2))])
        new_items = set(frame.index).difference(non_nan_items)
        self.items = _ensure_index(new_items.union(non_nan_items))

        # :NOTE:
        # There is a simpler and 10x faster way to do this:
        #
        # Reindex buffer to update axes (automatically adds nans)
        # self.buffer = self.buffer.reindex(items=self.items,
        #                                   major_axis=np.arange(self.cap),
        #                                   minor_axis=self.minor_axis)
        #
        # However, pandas==0.12.0, for which we remain backwards compatible,
        # has a bug in .reindex() that this triggers. Using .update() as before
        # seems to work fine.

        new_buffer = self._create_buffer()
        new_buffer.update(
            self.buffer.loc[non_nan_items, :, non_nan_cols])

        self.buffer = new_buffer
