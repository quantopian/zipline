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

from datetime import timedelta, datetime
from unittest import TestCase

from zipline import ndict

from zipline.utils.test_utils import setup_logger

from zipline.sources import SpecificEquityTrades
from zipline.transforms.utils import StatefulTransform, EventWindow
from zipline.transforms import MovingVWAP
from zipline.transforms import MovingAverage
from zipline.transforms import MovingStandardDev
from zipline.transforms import Returns
import zipline.utils.factory as factory

from zipline.test_algorithms import BatchTransformAlgorithm


def to_dt(msg):
    return ndict({'dt': msg})


class NoopEventWindow(EventWindow):
    """
    A no-op EventWindow subclass for testing the base EventWindow logic.
    Keeps lists of all added and dropped events.
    """
    def __init__(self, market_aware, days, delta):
        EventWindow.__init__(self, market_aware, days, delta)

        self.added = []
        self.removed = []

    def handle_add(self, event):
        self.added.append(event)

    def handle_remove(self, event):
        self.removed.append(event)


class TestEventWindow(TestCase):
    def setUp(self):
        self.sim_params = factory.create_simulation_parameters()

        setup_logger(self)

        self.monday = datetime(2012, 7, 9, 16, tzinfo=pytz.utc)
        self.eleven_normal_days = [self.monday + i * timedelta(days=1)
                                   for i in xrange(11)]

        # Modify the end of the period slightly to exercise the
        # incomplete day logic.
        self.eleven_normal_days[-1] -= timedelta(minutes=1)
        self.eleven_normal_days.append(self.monday +
                                       timedelta(days=11, seconds=1))

        # Second set of dates to test holiday handling.
        self.jul4_monday = datetime(2012, 7, 2, 16, tzinfo=pytz.utc)
        self.week_of_jul4 = [self.jul4_monday + i * timedelta(days=1)
                             for i in xrange(5)]

    def test_market_aware_window_normal_week(self):
        window = NoopEventWindow(
            market_aware=True,
            delta=None,
            days=3
        )
        events = [to_dt(date) for date in self.eleven_normal_days]
        lengths = []
        # Run the events.
        for event in events:
            window.update(event)
            # Record the length of the window after each event.
            lengths.append(len(window.ticks))

        # The window stretches out during the weekend because we wait
        # to drop events until the weekend ends. The last window is
        # briefly longer because it doesn't complete a full day.  The
        # window then shrinks once the day completes
        self.assertEquals(lengths, [1, 2, 3, 3, 3, 4, 5, 5, 5, 3, 4, 3])
        self.assertEquals(window.added, events)
        self.assertEquals(window.removed, events[:-3])

    def test_market_aware_window_holiday(self):
        window = NoopEventWindow(
            market_aware=True,
            delta=None,
            days=2
        )
        events = [to_dt(date) for date in self.week_of_jul4]
        lengths = []

        # Run the events.
        for event in events:
            window.update(event)
            # Record the length of the window after each event.
            lengths.append(len(window.ticks))

        self.assertEquals(lengths, [1, 2, 3, 3, 2])
        self.assertEquals(window.added, events)
        self.assertEquals(window.removed, events[:-2])

    def tearDown(self):
        setup_logger(self)


class TestFinanceTransforms(TestCase):

    def setUp(self):
        self.sim_params = factory.create_simulation_parameters()
        setup_logger(self)

        trade_history = factory.create_trade_history(
            133,
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            self.sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

    def tearDown(self):
        self.log_handler.pop_application()

    def test_vwap(self):
        vwap = MovingVWAP(
            market_aware=True,
            window_length=2
        )
        transformed = list(vwap.transform(self.source))

        # Output values
        tnfm_vals = [message[vwap.get_hash()] for message in transformed]
        # "Hand calculated" values.
        expected = [
            (10.0 * 100) / 100.0,
            ((10.0 * 100) + (10.0 * 100)) / (200.0),
            # We should drop the first event here.
            ((10.0 * 100) + (11.0 * 100)) / (200.0),
            # We should drop the second event here.
            ((11.0 * 100) + (11.0 * 300)) / (400.0)
        ]

        # Output should match the expected.
        self.assertEquals(tnfm_vals, expected)

    def test_returns(self):
        # Daily returns.
        returns = Returns(1)

        transformed = list(returns.transform(self.source))
        tnfm_vals = [message[returns.get_hash()] for message in transformed]

        # No returns for the first event because we don't have a
        # previous close.
        expected = [0.0, 0.0, 0.1, 0.0]

        self.assertEquals(tnfm_vals, expected)

        # Two-day returns.  An extra kink here is that the
        # factory will automatically skip a weekend for the
        # last event. Results shouldn't notice this blip.

        trade_history = factory.create_trade_history(
            133,
            [10.0, 15.0, 13.0, 12.0, 13.0],
            [100, 100, 100, 300, 100],
            timedelta(days=1),
            self.sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        returns = StatefulTransform(Returns, 2)

        transformed = list(returns.transform(self.source))
        tnfm_vals = [message[returns.get_hash()] for message in transformed]

        expected = [
            0.0,
            0.0,
            (13.0 - 10.0) / 10.0,
            (12.0 - 15.0) / 15.0,
            (13.0 - 13.0) / 13.0
        ]

        self.assertEquals(tnfm_vals, expected)

    def test_moving_average(self):

        mavg = MovingAverage(
            market_aware=True,
            fields=['price', 'volume'],
            window_length=2
        )

        transformed = list(mavg.transform(self.source))
        # Output values.
        tnfm_prices = [message[mavg.get_hash()].price
                       for message in transformed]
        tnfm_volumes = [message[mavg.get_hash()].volume
                        for message in transformed]

        # "Hand-calculated" values
        expected_prices = [
            ((10.0) / 1.0),
            ((10.0 + 10.0) / 2.0),
            # First event should get dropped here.
            ((10.0 + 11.0) / 2.0),
            # Second event should get dropped here.
            ((11.0 + 11.0) / 2.0)
        ]
        expected_volumes = [
            ((100.0) / 1.0),
            ((100.0 + 100.0) / 2.0),
            # First event should get dropped here.
            ((100.0 + 100.0) / 2.0),
            # Second event should get dropped here.
            ((100.0 + 300.0) / 2.0)
        ]

        self.assertEquals(tnfm_prices, expected_prices)
        self.assertEquals(tnfm_volumes, expected_volumes)

    def test_moving_stddev(self):
        trade_history = factory.create_trade_history(
            133,
            [10.0, 15.0, 13.0, 12.0],
            [100, 100, 100, 100],
            timedelta(days=1),
            self.sim_params
        )

        stddev = MovingStandardDev(
            market_aware=True,
            window_length=3,
        )

        self.source = SpecificEquityTrades(event_list=trade_history)

        transformed = list(stddev.transform(self.source))

        vals = [message[stddev.get_hash()] for message in transformed]

        expected = [
            None,
            np.std([10.0, 15.0], ddof=1),
            np.std([10.0, 15.0, 13.0], ddof=1),
            np.std([15.0, 13.0, 12.0], ddof=1),
        ]

        # np has odd rounding behavior, cf.
        # http://docs.scipy.org/doc/np/reference/generated/np.std.html
        for v1, v2 in zip(vals, expected):

            if v1 is None:
                self.assertIsNone(v2)
                continue
            self.assertEquals(round(v1, 5), round(v2, 5))


############################################################
# Test BatchTransform

class TestBatchTransform(TestCase):
    def setUp(self):
        self.sim_params = factory.create_simulation_parameters(
            start=datetime(1990, 1, 1, tzinfo=pytz.utc),
            end=datetime(1990, 1, 8, tzinfo=pytz.utc)
        )
        setup_logger(self)
        self.source, self.df = \
            factory.create_test_df_source(self.sim_params)

    def test_event_window(self):
        algo = BatchTransformAlgorithm()
        algo.run(self.source)
        wl = algo.window_length
        # The following assertion depend on window length of 3
        self.assertEqual(wl, 3)
        self.assertEqual(algo.history_return_price_class[:wl],
                         [None] * wl,
                         "First three iterations should return None." + "\n" +
                         "i.e. no returned values until window is full'" +
                         "%s" % (algo.history_return_price_class,))
        self.assertEqual(algo.history_return_price_decorator[:wl],
                         [None] * wl,
                         "First three iterations should return None." + "\n" +
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
        algo = BatchTransformAlgorithm(1, kwarg='str')
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
                # 1990-01-04 - window not full, 3rd event
                None,
                # 1990-01-05 - window now full
                expected_item,
                # 1990-01-08 - window now full
                expected_item
            ])
