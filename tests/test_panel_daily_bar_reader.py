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

from zipline.data.us_equity_pricing import PanelDailyBarReader
from zipline.testing import ExplodingObject
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithNYSETradingDays,
    ZiplineTestCase,
)


class TestPanelDailyBarReader(WithAssetFinder,
                              WithNYSETradingDays,
                              ZiplineTestCase):

    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-02-01', tz='utc')

    @classmethod
    def init_class_fixtures(cls):
        super(TestPanelDailyBarReader, cls).init_class_fixtures()

        finder = cls.asset_finder
        days = cls.trading_days

        items = finder.retrieve_all(finder.sids)
        major_axis = days
        minor_axis = ['open', 'high', 'low', 'close', 'volume']

        shape = tuple(map(len, [items, major_axis, minor_axis]))
        raw_data = np.arange(shape[0] * shape[1] * shape[2]).reshape(shape)

        cls.panel = pd.Panel(
            raw_data,
            items=items,
            major_axis=major_axis,
            minor_axis=minor_axis,
        )

        cls.reader = PanelDailyBarReader(days, cls.panel)

    def test_spot_price(self):
        panel = self.panel
        reader = self.reader

        for asset, date, field in product(*panel.axes):
            self.assertEqual(
                panel.loc[asset, date, field],
                reader.spot_price(asset, date, field),
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
                PanelDailyBarReader(unused, transposed)

            expected = (
                "Duplicate entries in Panel.{name}: ['a', 'b'].".format(
                    name=axis_names[axis_order.index(0)],
                )
            )
            self.assertEqual(str(e.exception), expected)
