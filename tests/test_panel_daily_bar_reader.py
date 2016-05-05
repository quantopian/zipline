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

import pandas as pd

from zipline.data.us_equity_pricing import PanelDailyBarReader
from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase


class TestPanelDailyBarReader(WithTradingEnvironment, ZiplineTestCase):
    def test_duplicate_values(self):
        df = pd.DataFrame()
        panel = pd.concat([pd.Panel({"X": df}), pd.Panel({"X": df})])

        with self.assertRaises(ValueError) as e:
            # panel's items has duplicates
            PanelDailyBarReader(None, panel)

        self.assertEqual("Duplicated items found: ['X']",
                         e.exception.message)

        with self.assertRaises(ValueError) as e:
            # panel's major axis has duplicates
            PanelDailyBarReader(None, panel.swapaxes(0, 1))

        self.assertEqual("Duplicated items found: ['X']",
                         e.exception.message)

        with self.assertRaises(ValueError) as e:
            # panel's minor axis has duplicates
            PanelDailyBarReader(None, panel.swapaxes(0, 2))

        self.assertEqual("Duplicated items found: ['X']",
                         e.exception.message)
