#
# Copyright 2014 Quantopian, Inc.
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

import unittest

from collections import deque

import numpy as np

import pandas as pd
import pandas.util.testing as tm

from zipline.utils.data import RollingPanel


class TestRollingPanel(unittest.TestCase):

    def test_basics(self, window=10):
        items = ['bar', 'baz', 'foo']
        minor = ['A', 'B', 'C', 'D']

        rp = RollingPanel(window, items, minor, cap_multiple=2)

        dates = pd.date_range('2000-01-01', periods=30, tz='utc')

        major_deque = deque(maxlen=window)

        frames = {}

        for i, date in enumerate(dates):
            frame = pd.DataFrame(np.random.randn(3, 4), index=items,
                                 columns=minor)

            rp.add_frame(date, frame)

            frames[date] = frame
            major_deque.append(date)

            result = rp.get_current()
            expected = pd.Panel(frames, items=list(major_deque),
                                major_axis=items, minor_axis=minor)

            tm.assert_panel_equal(result, expected.swapaxes(0, 1))

    def test_adding_and_dropping_items(self, n_items=5, n_minor=10, window=10):
        items = list(range(n_items))
        minor = list(range(n_minor))

        expected_items = list(range(n_items))
        expected_minor = list(range(n_minor))

        rp = RollingPanel(window, items, minor, cap_multiple=2)

        dates = pd.date_range('2000-01-01', periods=30, tz='utc')

        frames = {}

        expected_frames = deque(maxlen=window)
        expected_dates = deque()
        j = 0
        for i, date in enumerate(dates):
            frame = pd.DataFrame(np.random.randn(n_items, n_minor),
                                 index=items, columns=minor)

            if i >= window:
                # Old labels and dates should start to get dropped at every
                # call
                del frames[expected_dates.popleft()]
                expected_minor = expected_minor[1:]
                expected_items = expected_items[1:]

            expected_frames.append(frame)
            expected_dates.append(date)

            rp.add_frame(date, frame)

            frames[date] = frame

            result = rp.get_current()
            np.testing.assert_array_equal(sorted(result.minor_axis.values),
                                          sorted(expected_minor))
            np.testing.assert_array_equal(sorted(result.items.values),
                                          sorted(expected_items))
            tm.assert_frame_equal(frame.T,
                                  result.ix[frame.index, -1, frame.columns])
            expected_result = pd.Panel(frames).swapaxes(0, 1)
            tm.assert_panel_equal(expected_result,
                                  result)

            # shift minor and items to trigger updating of underlying data
            # structure
            if i < window:
                # Insert new items
                minor = minor[1:]
                minor.append(minor[-1] + 1)
                items = items[1:]
                items.append(items[-1] + 1)

                expected_minor.append(expected_minor[-1] + 1)
                expected_items.append(expected_items[-1] + 1)
            else:
                # Start inserting old items out of order to make sure sorting
                # works.
                j += 1
                minor = minor[1:]
                minor.append(j)
                items = items[1:]
                items.append(j)

                expected_minor.append(j)
                expected_items.append(j)


def run_history_implementations(option='clever', n=500, change_fields=False,
                                copy=False, n_items=15, n_minor=20,
                                change_freq=5, window=100):
    items = range(n_items)
    minor = range(n_minor)
    periods = n

    dates = pd.date_range('2000-01-01', periods=periods, tz='utc')
    frames = {}

    if option == 'clever':
        rp = RollingPanel(window, items, minor, cap_multiple=2)
        major_deque = deque()

        for i in range(periods):
            # Add a new and drop an field every change_freq iterations
            if change_fields and (i % change_freq) == 0:
                minor = minor[1:]
                minor.append(minor[-1] + 1)
                items = items[1:]
                items.append(items[-1] + 1)

            dummy = pd.DataFrame(np.random.randn(len(items), len(minor)),
                                 index=items, columns=minor)

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
