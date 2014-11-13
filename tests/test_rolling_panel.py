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

from zipline.utils.data import MutableIndexRollingPanel, RollingPanel
from zipline.finance.trading import with_environment


class TestRollingPanel(unittest.TestCase):
    @with_environment()
    def test_alignment(self, env):
        items = ('a', 'b')
        sids = (1, 2)

        dts = env.market_minute_window(
            env.open_and_closes.market_open[0], 4,
        ).values
        rp = RollingPanel(2, items, sids, initial_dates=dts[1:-1])

        frame = pd.DataFrame(
            data=np.arange(4).reshape((2, 2)),
            columns=sids,
            index=items,
        )

        nan_arr = np.empty((2, 6))
        nan_arr.fill(np.nan)

        rp.add_frame(dts[-1], frame)

        cur = rp.get_current()
        data = np.array((((np.nan, np.nan),
                          (0, 1)),
                         ((np.nan, np.nan),
                          (2, 3))),
                        float)
        expected = pd.Panel(
            data,
            major_axis=dts[2:],
            minor_axis=sids,
            items=items,
        )
        expected.major_axis = expected.major_axis.tz_localize('utc')
        tm.assert_panel_equal(
            cur,
            expected,
        )

        rp.extend_back(dts[:-2])

        cur = rp.get_current()
        data = np.array((((np.nan, np.nan),
                          (np.nan, np.nan),
                          (np.nan, np.nan),
                          (0, 1)),
                         ((np.nan, np.nan),
                          (np.nan, np.nan),
                          (np.nan, np.nan),
                          (2, 3))),
                        float)
        expected = pd.Panel(
            data,
            major_axis=dts,
            minor_axis=sids,
            items=items,
        )
        expected.major_axis = expected.major_axis.tz_localize('utc')
        tm.assert_panel_equal(
            cur,
            expected,
        )


class TestMutableIndexRollingPanel(unittest.TestCase):

    def test_basics(self, window=10):
        items = ['bar', 'baz', 'foo']
        minor = ['A', 'B', 'C', 'D']

        rp = MutableIndexRollingPanel(window, items, minor, cap_multiple=2)

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

    def test_adding_and_dropping_items(self, n_items=5, n_minor=10, window=10,
                                       periods=30):
        np.random.seed(123)

        items = deque(range(n_items))
        minor = deque(range(n_minor))

        expected_items = deque(range(n_items))
        expected_minor = deque(range(n_minor))

        first_non_existant = max(n_items, n_minor) + 1
        # We want to add new columns with random order
        add_items = np.arange(first_non_existant, first_non_existant + periods)
        np.random.shuffle(add_items)

        rp = MutableIndexRollingPanel(window, items, minor, cap_multiple=2)

        dates = pd.date_range('2000-01-01', periods=periods, tz='utc')

        frames = {}

        expected_frames = deque(maxlen=window)
        expected_dates = deque()

        for i, (date, add_item) in enumerate(zip(dates, add_items)):
            frame = pd.DataFrame(np.random.randn(n_items, n_minor),
                                 index=items, columns=minor)

            if i >= window:
                # Old labels and dates should start to get dropped at every
                # call
                del frames[expected_dates.popleft()]
                expected_minor.popleft()
                expected_items.popleft()

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

            # Insert new items
            minor.popleft()
            minor.append(add_item)
            items.popleft()
            items.append(add_item)

            expected_minor.append(add_item)
            expected_items.append(add_item)
