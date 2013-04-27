import unittest

from collections import deque

import numpy as np

import pandas as pd
import pandas.util.testing as tm

from zipline.utils.data import RollingPanel


class TestRollingPanel(unittest.TestCase):
    def test_basics(self):
        items = ['foo', 'bar', 'baz']
        minor = ['A', 'B', 'C', 'D']

        window = 10

        rp = RollingPanel(window, items, minor, cap_multiple=2)

        dates = pd.date_range('2000-01-01', periods=30, tz='utc')

        major_deque = deque()

        frames = {}

        for i in range(30):
            frame = pd.DataFrame(np.random.randn(3, 4), index=items,
                                 columns=minor)
            date = dates[i]

            rp.add_frame(date, frame)

            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                major_deque.popleft()

            result = rp.get_current()
            expected = pd.Panel(frames, items=list(major_deque),
                                major_axis=items, minor_axis=minor)
            tm.assert_panel_equal(result, expected.swapaxes(0, 1))


def f(option='clever', n=500, copy=False):
    items = range(5)
    minor = range(20)
    window = 100
    periods = n

    dates = pd.date_range('2000-01-01', periods=periods, tz='utc')
    frames = {}

    if option == 'clever':
        rp = RollingPanel(window, items, minor, cap_multiple=2)
        major_deque = deque()
        dummy = pd.DataFrame(np.random.randn(len(items), len(minor)),
                             index=items, columns=minor)

        for i in range(periods):
            frame = dummy * (1 + 0.001 * i)
            date = dates[i]

            rp.add_frame(date, frame)

            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                del frames[major_deque.popleft()]

            result = rp.get_current()
            if copy:
                result = result.copy()
    else:
        major_deque = deque()
        dummy = pd.DataFrame(np.random.randn(len(items), len(minor)),
                             index=items, columns=minor)

        for i in range(periods):
            frame = dummy * (1 + 0.001 * i)
            date = dates[i]
            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                del frames[major_deque.popleft()]

            result = pd.Panel(frames, items=list(major_deque),
                              major_axis=items, minor_axis=minor)
