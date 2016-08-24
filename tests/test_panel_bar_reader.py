#
# Copyright 2016 Quantopian, Inc.
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

from itertools import permutations, product

import numpy as np
import pandas as pd

from zipline.data.us_equity_pricing import PanelBarReader
from zipline.testing import ExplodingObject
from zipline.testing.fixtures import (
    WithAssetFinder,
    ZiplineTestCase,
)
from zipline.utils.calendars import get_calendar


class WithPanelBarReader(WithAssetFinder):

    @classmethod
    def init_class_fixtures(cls):
        super(WithPanelBarReader, cls).init_class_fixtures()

        finder = cls.asset_finder
        trading_calendar = get_calendar('NYSE')

        items = finder.retrieve_all(finder.sids)
        major_axis = (
            trading_calendar.sessions_in_range if cls.FREQUENCY == 'daily'
            else trading_calendar.minutes_for_sessions_in_range
        )(cls.START_DATE, cls.END_DATE)
        minor_axis = ['open', 'high', 'low', 'close', 'volume']

        shape = tuple(map(len, [items, major_axis, minor_axis]))
        raw_data = np.arange(shape[0] * shape[1] * shape[2]).reshape(shape)

        cls.panel = pd.Panel(
            raw_data,
            items=items,
            major_axis=major_axis,
            minor_axis=minor_axis,
        )

        cls.reader = PanelBarReader(trading_calendar, cls.panel, cls.FREQUENCY)

    def test_get_value(self):
        panel = self.panel
        reader = self.reader

        for asset, date, field in product(*panel.axes):
            self.assertEqual(
                panel.loc[asset, date, field],
                reader.get_value(asset, date, field),
            )

    def test_duplicate_values(self):
        UNIMPORTANT_VALUE = 57

        panel = pd.Panel(
            UNIMPORTANT_VALUE,
            items=['a', 'b', 'b', 'a'],
            major_axis=['c'],
            minor_axis=['d'],
        )
        unused = ExplodingObject()

        axis_names = ['items', 'major_axis', 'minor_axis']

        for axis_order in permutations((0, 1, 2)):
            transposed = panel.transpose(*axis_order)
            with self.assertRaises(ValueError) as e:
                PanelBarReader(unused, transposed, 'daily')

            expected = (
                "Duplicate entries in Panel.{name}: ['a', 'b'].".format(
                    name=axis_names[axis_order.index(0)],
                )
            )
            self.assertEqual(str(e.exception), expected)

    def test_sessions(self):
        sessions = self.reader.sessions

        self.assertEqual(self.NUM_SESSIONS, len(sessions))
        self.assertEqual(self.START_DATE, sessions[0])
        self.assertEqual(self.END_DATE, sessions[-1])


class TestPanelDailyBarReader(WithPanelBarReader,
                              ZiplineTestCase):

    FREQUENCY = 'daily'

    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-02-01', tz='utc')

    NUM_SESSIONS = 21


class TestPanelMinuteBarReader(WithPanelBarReader,
                               ZiplineTestCase):

    FREQUENCY = 'minute'

    START_DATE = pd.Timestamp('2015-12-23', tz='utc')
    END_DATE = pd.Timestamp('2015-12-24', tz='utc')

    NUM_SESSIONS = 2
