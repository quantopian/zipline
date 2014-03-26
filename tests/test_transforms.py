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

import pytz
import numpy as np
import pandas as pd

from datetime import timedelta, datetime
from unittest import TestCase, skip

from six.moves import range

from zipline.utils.test_utils import setup_logger

from zipline.protocol import Event
from zipline.sources import SpecificEquityTrades
from zipline.transforms.utils import StatefulTransform, EventWindow
from zipline.transforms import MovingVWAP
from zipline.transforms import MovingAverage
from zipline.transforms import MovingStandardDev
from zipline.transforms import Returns
import zipline.utils.factory as factory

from zipline.test_algorithms import TALIBAlgorithm


def to_dt(msg):
    return Event({'dt': msg})


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
                                   for i in range(11)]

        # Modify the end of the period slightly to exercise the
        # incomplete day logic.
        self.eleven_normal_days[-1] -= timedelta(minutes=1)
        self.eleven_normal_days.append(self.monday +
                                       timedelta(days=11, seconds=1))

        # Second set of dates to test holiday handling.
        self.jul4_monday = datetime(2012, 7, 2, 16, tzinfo=pytz.utc)
        self.week_of_jul4 = [self.jul4_monday + i * timedelta(days=1)
                             for i in range(5)]

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
# Test TALIB

import talib
import zipline.transforms.ta as ta


class TestTALIB(TestCase):
    def setUp(self):
        setup_logger(self)
        sim_params = factory.create_simulation_parameters(
            start=datetime(1990, 1, 1, tzinfo=pytz.utc),
            end=datetime(1990, 3, 30, tzinfo=pytz.utc))
        self.source, self.panel = \
            factory.create_test_panel_ohlc_source(sim_params)

    @skip
    def test_talib_with_default_params(self):
        BLACKLIST = ['make_transform', 'BatchTransform',
                     # TODO: Figure out why MAVP generates a KeyError
                     'MAVP']
        names = [name for name in dir(ta) if name[0].isupper()
                 and name not in BLACKLIST]

        for name in names:
            print(name)
            zipline_transform = getattr(ta, name)(sid=0)
            talib_fn = getattr(talib.abstract, name)

            start = datetime(1990, 1, 1, tzinfo=pytz.utc)
            end = start + timedelta(days=zipline_transform.lookback + 10)
            sim_params = factory.create_simulation_parameters(
                start=start, end=end)
            source, panel = \
                factory.create_test_panel_ohlc_source(sim_params)

            algo = TALIBAlgorithm(talib=zipline_transform)
            algo.run(source)

            zipline_result = np.array(
                algo.talib_results[zipline_transform][-1])

            talib_data = dict()
            data = zipline_transform.window
            # TODO: Figure out if we are clobbering the tests by this
            # protection against empty windows
            if not data:
                continue
            for key in ['open', 'high', 'low', 'volume']:
                if key in data:
                    talib_data[key] = data[key][0].values
            talib_data['close'] = data['price'][0].values
            expected_result = talib_fn(talib_data)

            if isinstance(expected_result, list):
                expected_result = np.array([e[-1] for e in expected_result])
            else:
                expected_result = np.array(expected_result[-1])
            if not (np.all(np.isnan(zipline_result))
                    and np.all(np.isnan(expected_result))):
                self.assertTrue(np.allclose(zipline_result, expected_result))
            else:
                print('--- NAN')

            # reset generator so next iteration has data
            # self.source, self.panel = \
                # factory.create_test_panel_ohlc_source(self.sim_params)

    def test_multiple_talib_with_args(self):
        zipline_transforms = [ta.MA(timeperiod=10),
                              ta.MA(timeperiod=25)]
        talib_fn = talib.abstract.MA
        algo = TALIBAlgorithm(talib=zipline_transforms)
        algo.run(self.source)
        # Test if computed values match those computed by pandas rolling mean.
        sid = 0
        talib_values = np.array([x[sid] for x in
                                 algo.talib_results[zipline_transforms[0]]])
        np.testing.assert_array_equal(talib_values,
                                      pd.rolling_mean(self.panel[0]['price'],
                                                      10).values)
        talib_values = np.array([x[sid] for x in
                                 algo.talib_results[zipline_transforms[1]]])
        np.testing.assert_array_equal(talib_values,
                                      pd.rolling_mean(self.panel[0]['price'],
                                                      25).values)
        for t in zipline_transforms:
            talib_result = np.array(algo.talib_results[t][-1])
            talib_data = dict()
            data = t.window
            # TODO: Figure out if we are clobbering the tests by this
            # protection against empty windows
            if not data:
                continue
            for key in ['open', 'high', 'low', 'volume']:
                if key in data:
                    talib_data[key] = data[key][0].values
            talib_data['close'] = data['price'][0].values
            expected_result = talib_fn(talib_data, **t.call_kwargs)[-1]
            np.testing.assert_allclose(talib_result, expected_result)

    def test_talib_with_minute_data(self):

        ma_one_day_minutes = ta.MA(timeperiod=10, bars='minute')

        # Assert that the BatchTransform window length is enough to cover
        # the amount of minutes in the timeperiod.

        # Here, 10 minutes only needs a window length of 1.
        self.assertEquals(1, ma_one_day_minutes.window_length)

        # With minutes greater than the 390, i.e. one trading day, we should
        # have a window_length of two days.
        ma_two_day_minutes = ta.MA(timeperiod=490, bars='minute')
        self.assertEquals(2, ma_two_day_minutes.window_length)

        # TODO: Ensure that the lookback into the datapanel is returning
        # expected results.
        # Requires supplying minute instead of day data to the unit test.
        # When adding test data, should add more minute events than the
        # timeperiod to ensure that lookback is behaving properly.
