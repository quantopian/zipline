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

    def __init__(self, window, items, minor_axis, cap_multiple=2,
                 dtype=np.float64):
        self.pos = 0
        self.window = window

        self.items = _ensure_index(items)
        self.minor_axis = _ensure_index(minor_axis)

        self.cap_multiple = cap_multiple
        self.cap = cap_multiple * window

        self.dtype = dtype
        self.index_buf = np.empty(self.cap, dtype='M8[ns]')
        self.buffer = pd.Panel(items=items, minor_axis=minor_axis,
                               major_axis=range(self.cap),
                               dtype=dtype)

    def add_frame(self, tick, frame):
        """
        """
        if self.pos == self.cap:
            self._roll_data()
        self.buffer.values[:, self.pos, :] = frame.ix[self.items].values
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


class NaiveRollingPanel(object):

    def __init__(self, window, items, minor_axis, cap_multiple=2):
        pass
