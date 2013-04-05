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
import pandas as pd
import pytz
from itertools import cycle

from unittest import TestCase

import zipline.utils.factory as factory
from zipline.sources import DataFrameSource, DataPanelSource


class TestDataFrameSource(TestCase):
    def test_df_source(self):
        source, df = factory.create_test_df_source()
        assert isinstance(source.start, pd.lib.Timestamp)
        assert isinstance(source.end, pd.lib.Timestamp)

        for expected_dt, expected_price in df.iterrows():
            sid0 = source.next()

            assert expected_dt == sid0.dt
            assert expected_price[0] == sid0.price

    def test_df_sid_filtering(self):
        _, df = factory.create_test_df_source()
        source = DataFrameSource(df, sids=[0])
        assert 1 not in [event.sid for event in source], \
            "DataFrameSource should only stream selected sid 0, not sid 1."

    def test_panel_source(self):
        source, panel = factory.create_test_panel_source()
        assert isinstance(source.start, pd.lib.Timestamp)
        assert isinstance(source.end, pd.lib.Timestamp)
        for event in source:
            self.assertTrue('sid' in event)
            self.assertTrue('arbitrary' in event)
            self.assertTrue('volume' in event)
            self.assertTrue('price' in event)
            self.assertEquals(event['arbitrary'], 1.)
            self.assertEquals(event['volume'], 1000)
            self.assertEquals(event['sid'], 0)
            self.assertTrue(isinstance(event['volume'], int))
            self.assertTrue(isinstance(event['arbitrary'], float))

    def test_yahoo_bars_to_panel_source(self):
        stocks = ['AAPL', 'GE']
        start = pd.datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
        data = factory.load_bars_from_yahoo(stocks=stocks,
                                            indexes={},
                                            start=start,
                                            end=end)

        check_fields = ['sid', 'open', 'high', 'low', 'close',
                        'volume', 'price']
        source = DataPanelSource(data)
        stocks_iter = cycle(stocks)
        for event in source:
            for check_field in check_fields:
                self.assertIn(check_field, event)
            self.assertTrue(isinstance(event['volume'], (int, long)))
            self.assertEqual(stocks_iter.next(), event['sid'])
