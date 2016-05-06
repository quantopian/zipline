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

from itertools import permutations

import pandas as pd

from zipline.data.us_equity_pricing import PanelDailyBarReader
from zipline.testing import ExplodingObject
from zipline.testing.fixtures import ZiplineTestCase


class TestPanelDailyBarReader(ZiplineTestCase):
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
            with self.assertRaises(ValueError) as e:
                PanelDailyBarReader(unused, panel.transpose(*axis_order))

            expected = (
                "Duplicate entries in Panel.{name}: ['a', 'b'].".format(
                    name=axis_names[axis_order.index(0)],
                )
            )
            self.assertEqual(str(e.exception), expected)
