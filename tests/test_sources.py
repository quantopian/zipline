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
import numpy as np

from six import integer_types

from unittest import TestCase

import zipline.utils.factory as factory
from zipline.sources import (DataFrameSource,
                             DataPanelSource,
                             RandomWalkSource)
from zipline.utils import tradingcalendar as calendar_nyse


class TestDataFrameSource(TestCase):
    def test_df_source(self):
        source, df = factory.create_test_df_source()
        assert isinstance(source.start, pd.lib.Timestamp)
        assert isinstance(source.end, pd.lib.Timestamp)

        for expected_dt, expected_price in df.iterrows():
            sid0 = next(source)

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
            self.assertTrue(isinstance(event['volume'], (integer_types)))
            self.assertEqual(next(stocks_iter), event['sid'])


class TestRandomWalkSource(TestCase):
    def test_minute(self):
        np.random.seed(123)
        start_prices = {0: 100,
                        1: 500}
        start = pd.Timestamp('1990-01-01', tz='UTC')
        end = pd.Timestamp('1991-01-01', tz='UTC')
        source = RandomWalkSource(start_prices=start_prices,
                                  calendar=calendar_nyse, start=start,
                                  end=end)
        self.assertIsInstance(source.start, pd.lib.Timestamp)
        self.assertIsInstance(source.end, pd.lib.Timestamp)

        for event in source:
            self.assertIn(event.sid, start_prices.keys())
            self.assertIn(event.dt.replace(minute=0, hour=0),
                          calendar_nyse.trading_days)
            self.assertGreater(event.dt, start)
            self.assertLess(event.dt, end)
            self.assertGreater(event.price, 0,
                               "price should never go negative.")
            self.assertEqual(event.volume, 1000)
            self.assertTrue(13 <= event.dt.hour <= 21,
                            "event.dt.hour == %i, not during market \
                            hours." % event.dt.hour)

    def test_day(self):
        np.random.seed(123)
        start_prices = {0: 100,
                        1: 500}
        start = pd.Timestamp('1990-01-01', tz='UTC')
        end = pd.Timestamp('1992-01-01', tz='UTC')
        source = RandomWalkSource(start_prices=start_prices,
                                  calendar=calendar_nyse, start=start,
                                  end=end, freq='day')
        self.assertIsInstance(source.start, pd.lib.Timestamp)
        self.assertIsInstance(source.end, pd.lib.Timestamp)

        for event in source:
            self.assertIn(event.sid, start_prices.keys())
            self.assertIn(event.dt.replace(minute=0, hour=0),
                          calendar_nyse.trading_days)
            self.assertGreater(event.dt, start)
            self.assertLess(event.dt, end)
            self.assertGreater(event.price, 0,
                               "price should never go negative.")
            self.assertEqual(event.volume, 1000)
            self.assertTrue(13 <= event.dt.hour <= 21,
                            "event.dt.hour == %i, not during market \
                            hours." % event.dt.hour)
