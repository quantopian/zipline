from collections import deque

import numpy as np
import pandas as pd

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
        self.buffer = np.empty((len(items), self.cap, len(minor_axis)),
                               dtype=dtype)
        self.index_buf = np.empty(self.cap, dtype='M8[ns]')

    def add_frame(self, tick, frame):
        """

        TODO: this assumes the DataFrame has the right shape
        """
        if self.pos == self.cap:
            self._roll_data()

        self.buffer[:, self.pos, :] = frame.values
        self.index_buf[self.pos] = tick

        self.pos += 1

    def get_current(self):
        """
        Get a Panel that is the current data in view. It is not safe to persist
        these objects because internal data might change
        """
        where = slice(max(self.pos - self.window, 0), self.pos)
        major_axis = pd.DatetimeIndex(self.index_buf[where], tz='utc')
        return pd.Panel(self.buffer[:, where, :], self.items, major_axis,
                        self.minor_axis)

    def _roll_data(self):
        """
        Roll window worth of data up to position zero. Save the effort of having
        to expensively roll at each iteration
        """
        self.buffer[:, :self.window, :] = self.buffer[:, -self.window:]
        self.index_buf[:self.window] = self.index_buf[-self.window:]
        self.pos = self.window

class NaiveRollingPanel(object):

    def __init__(self, window, items, minor_axis, cap_multiple=2):
        pass
