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

import numpy as np
import pandas as pd
from copy import deepcopy


def _ensure_index(x):
    if not isinstance(x, pd.Index):
        x = pd.Index(sorted(x))

    return x


class RollingPanel(object):
    """
    Preallocation strategies for rolling window over expanding data set

    Restrictions: major_axis can only be a DatetimeIndex for now
    """

    def __init__(self, window, items, sids, cap_multiple=2, dtype=np.float64):

        self.pos = 0
        self.window = window

        self.items = _ensure_index(items)
        self.minor_axis = _ensure_index(sids)

        self.cap_multiple = cap_multiple
        self.cap = cap_multiple * window

        self.dtype = dtype
        self.index_buf = np.empty(self.cap, dtype='M8[ns]')

        self.buffer = self._create_buffer()

    def _create_buffer(self):
        panel = pd.Panel(
            items=self.items,
            minor_axis=self.minor_axis,
            major_axis=range(self.cap),
            dtype=self.dtype,
        )

        return panel

    def _update_buffer(self, frame):
        # Get current frame as we only need to care about the data that is in
        # the active window
        # Note that we have to increase pos so that we get the current frame as
        # self.pos is increased _after_ this call
        old_buffer = self.get_current(self.pos + 1)

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

    def add_frame(self, tick, frame):
        """
        """
        if self.pos == self.cap:
            self._roll_data()

        if set(frame.columns).difference(set(self.minor_axis)) or \
                set(frame.index).difference(set(self.items)):
            self._update_buffer(frame)

        self.buffer.loc[:, self.pos, :] = \
            frame.ix[self.items].T.astype(self.dtype)

        self.index_buf[self.pos] = tick

        self.pos += 1

    def get_current(self, pos=None):
        """
        Get a Panel that is the current data in view. It is not safe to persist
        these objects because internal data might change
        """
        if pos is None:
            pos = self.pos

        where = slice(max(pos - self.window, 0), pos)
        major_axis = pd.DatetimeIndex(deepcopy(self.index_buf[where]),
                                      tz='utc')

        return pd.Panel(self.buffer.values[:, where, :], self.items,
                        major_axis, self.minor_axis, dtype=self.dtype)

    def _roll_data(self):
        """
        Roll window worth of data up to position zero.
        Save the effort of having to expensively roll at each iteration
        """

        self.buffer.values[:, :self.window, :] = \
            self.buffer.values[:, -self.window:, :]
        # Clean out nans so that they get dropped in _update_buffer()
        self.buffer.values[:, -self.window:, :] = np.nan
        self.index_buf[:self.window] = self.index_buf[-self.window:]
        self.pos = self.window
