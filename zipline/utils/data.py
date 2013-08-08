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
        x = pd.Index(x)

    return x


class RollingPanel(object):
    """
    Preallocation strategies for rolling window over expanding data set

    Restrictions: major_axis can only be a DatetimeIndex for now
    """

    def __init__(self, window, items, sids, cap_multiple=2,
                 dtype=np.float64):
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
        return pd.Panel(items=self.items, minor_axis=self.minor_axis,
                        major_axis=range(self.cap),
                        dtype=self.dtype)

    def _update_buffer(self, frame):
        # Drop outdated, nan-filled minors (sids) and items (fields)
        non_nan_cols = set(self.buffer.dropna(axis=1).minor_axis)
        new_cols = set(frame.columns)
        self.minor_axis = _ensure_index(new_cols.union(non_nan_cols))

        non_nan_items = set(self.buffer.dropna(axis=1).items)
        new_items = set(frame.index)
        self.items = _ensure_index(new_items.union(non_nan_items))

        new_buffer = self._create_buffer()
        # Copy old values we want to keep
        # .update() is pretty slow. Ideally we would be using
        # new_buffer.loc[non_nan_items, :, non_nan_cols] =
        # but this triggers a bug in Pandas 0.11. Update
        # this when 0.12 is released.
        # https://github.com/pydata/pandas/issues/3777
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

        self.buffer.loc[:, self.pos, :] = frame.ix[self.items].T

        self.index_buf[self.pos] = tick

        self.pos += 1

    def get_current(self):
        """
        Get a Panel that is the current data in view. It is not safe to persist
        these objects because internal data might change
        """
        where = slice(max(self.pos - self.window, 0), self.pos)
        major_axis = pd.DatetimeIndex(deepcopy(self.index_buf[where]),
                                      tz='utc')

        return pd.Panel(self.buffer.values[:, where, :], self.items,
                        major_axis, self.minor_axis)

    def _roll_data(self):
        """
        Roll window worth of data up to position zero.
        Save the effort of having to expensively roll at each iteration
        """
        self.buffer.values[:, :self.window, :] = \
            self.buffer.values[:, -self.window:]
        self.index_buf[:self.window] = self.index_buf[-self.window:]
        self.pos = self.window
