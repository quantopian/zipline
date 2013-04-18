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

from collections import deque

import pytz
import numpy as np
import pandas as pd

from datetime import datetime
from unittest import TestCase

from zipline.utils.test_utils import setup_logger

from zipline.sources.data_source import DataSource
import zipline.utils.factory as factory

from zipline.test_algorithms import (BatchTransformAlgorithm,
                                     batch_transform,
                                     ReturnPriceBatchTransform)

from zipline.algorithm import TradingAlgorithm
from zipline.utils.tradingcalendar import trading_days
from copy import deepcopy


@batch_transform
def return_price(data):
    return data.price


class BatchTransformAlgorithmSetSid(TradingAlgorithm):
    def initialize(self, sids):
        self.history = []

        self.batch_transform = return_price(
            refresh_period=1,
            window_length=10,
            clean_nans=False,
            sids=sids,
            compute_only_full=False
        )

    def handle_data(self, data):
        self.history.append(
            deepcopy(self.batch_transform.handle_data(data)))


class DifferentSidSource(DataSource):
    def __init__(self):
        self.dates = pd.date_range('1990-01-01', periods=180, tz='utc')
        self.start = self.dates[0]
        self.end = self.dates[-1]
        self._raw_data = None
        self.sids = range(90)
        self.sid = 0
        self.trading_days = []

    @property
    def instance_hash(self):
        return '1234'

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data

    @property
    def mapping(self):
        return {
            'dt': (lambda x: x, 'dt'),
            'sid': (lambda x: x, 'sid'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
        }

    def raw_data_gen(self):
        # Create differente sid for each event
        for date in self.dates:
            if date not in trading_days:
                continue
            event = {'dt': date,
                     'sid': self.sid,
                     'price': self.sid,
                     'volume': self.sid}
            self.sid += 1
            self.trading_days.append(date)
            yield event


class TestChangeOfSids(TestCase):
    def setUp(self):
        self.sids = range(90)
        self.sim_params = factory.create_simulation_parameters(
            start=datetime(1990, 1, 1, tzinfo=pytz.utc),
            end=datetime(1990, 1, 8, tzinfo=pytz.utc)
        )

    def test_all_sids_passed(self):
        algo = BatchTransformAlgorithmSetSid(self.sids,
                                             sim_params=self.sim_params)
        source = DifferentSidSource()
        algo.run(source)
        for df, date in zip(algo.history, source.trading_days):
            self.assertEqual(df.index[-1], date, "Newest event doesn't \
                             match.")

            for sid in self.sids:
                self.assertIn(sid, df.columns)

            last_elem = len(df) - 1
            self.assertEqual(df[last_elem][last_elem], last_elem)


class TestBatchTransform(TestCase):
    def setUp(self):
        self.sim_params = factory.create_simulation_parameters(
            start=datetime(1990, 1, 1, tzinfo=pytz.utc),
            end=datetime(1990, 1, 8, tzinfo=pytz.utc)
        )
        setup_logger(self)
        self.source, self.df = \
            factory.create_test_df_source(self.sim_params)

    def test_core_functionality(self):
        algo = BatchTransformAlgorithm(sim_params=self.sim_params)
        algo.run(self.source)
        wl = algo.window_length
        # The following assertion depend on window length of 3
        self.assertEqual(wl, 3)
        # If window_length is 3, there should be 2 None events, as the
        # window fills up on the 3rd day.
        n_none_events = 2
        self.assertEqual(algo.history_return_price_class[:n_none_events],
                         [None] * n_none_events,
                         "First two iterations should return None." + "\n" +
                         "i.e. no returned values until window is full'" +
                         "%s" % (algo.history_return_price_class,))
        self.assertEqual(algo.history_return_price_decorator[:n_none_events],
                         [None] * n_none_events,
                         "First two iterations should return None." + "\n" +
                         "i.e. no returned values until window is full'" +
                         "%s" % (algo.history_return_price_decorator,))

        # After three Nones, the next value should be a data frame
        self.assertTrue(isinstance(
            algo.history_return_price_class[wl],
            pd.DataFrame)
        )

        # Test whether arbitrary fields can be added to datapanel
        field = algo.history_return_arbitrary_fields[-1]
        self.assertTrue(
            'arbitrary' in field.items,
            'datapanel should contain column arbitrary'
        )

        self.assertTrue(all(
            field['arbitrary'].values.flatten() ==
            [123] * algo.window_length),
            'arbitrary dataframe should contain only "test"'
        )

        for data in algo.history_return_sid_filter[wl:]:
            self.assertIn(0, data.columns)
            self.assertNotIn(1, data.columns)

        for data in algo.history_return_field_filter[wl:]:
            self.assertIn('price', data.items)
            self.assertNotIn('ignore', data.items)

        for data in algo.history_return_field_no_filter[wl:]:
            self.assertIn('price', data.items)
            self.assertIn('ignore', data.items)

        for data in algo.history_return_ticks[wl:]:
            self.assertTrue(isinstance(data, deque))

        for data in algo.history_return_not_full:
            self.assertIsNot(data, None)

        # test overloaded class
        for test_history in [algo.history_return_price_class,
                             algo.history_return_price_decorator]:
            # starting at window length, the window should contain
            # consecutive (of window length) numbers up till the end.
            for i in range(algo.window_length, len(test_history)):
                np.testing.assert_array_equal(
                    range(i - algo.window_length + 1, i + 1),
                    test_history[i].values.flatten()
                )

    def test_passing_of_args(self):
        algo = BatchTransformAlgorithm(1, kwarg='str',
                                       sim_params=self.sim_params)
        self.assertEqual(algo.args, (1,))
        self.assertEqual(algo.kwargs, {'kwarg': 'str'})

        algo.run(self.source)
        expected_item = ((1, ), {'kwarg': 'str'})
        self.assertEqual(
            algo.history_return_args,
            [
                # 1990-01-01 - market holiday, no event
                # 1990-01-02 - window not full
                None,
                # 1990-01-03 - window not full
                None,
                # 1990-01-04 - window now full, 3rd event
                expected_item,
                # 1990-01-05 - window now full
                expected_item,
                # 1990-01-08 - window now full
                expected_item
            ])


def run_batchtransform(window_length=10):
    sim_params = factory.create_simulation_parameters(
        start=datetime(1990, 1, 1, tzinfo=pytz.utc),
        end=datetime(1995, 1, 8, tzinfo=pytz.utc)
    )
    source, df = factory.create_test_df_source(sim_params)

    return_price_class = ReturnPriceBatchTransform(
        refresh_period=1,
        window_length=window_length,
        clean_nans=False
    )

    for raw_event in source:
        raw_event['datetime'] = raw_event.dt
        event = {0: raw_event}
        return_price_class.handle_data(event)
