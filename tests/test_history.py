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

from unittest import TestCase
from itertools import product
from textwrap import dedent
import warnings

from nose_parameterized import parameterized
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from pandas.tseries.tools import normalize_date

from .history_cases import (
    HISTORY_CONTAINER_TEST_CASES,
)
from zipline import TradingAlgorithm
from zipline.errors import HistoryInInitialize, IncompatibleHistoryFrequency
from zipline.finance import trading
from zipline.finance.trading import (
    SimulationParameters,
    TradingEnvironment,
)
from zipline.history import history
from zipline.history.history_container import HistoryContainer
from zipline.protocol import BarData
from zipline.sources import RandomWalkSource, DataFrameSource
from zipline.testing import subtest
import zipline.utils.factory as factory


class TestHistoryContainer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def bar_data_dt(self, bar_data, require_unique=True):
        """
        Get a dt to associate with the given BarData object.

        If require_unique == True, throw an error if multiple unique dt's are
        encountered.  Otherwise, return the earliest dt encountered.
        """
        dts = {sid_data['dt'] for sid_data in bar_data.values()}
        if require_unique and len(dts) > 1:
            self.fail("Multiple unique dts ({0}) in {1}".format(dts, bar_data))

        return sorted(dts)[0]

    @parameterized.expand(
        [(name,
          case['specs'],
          case['sids'],
          case['dt'],
          case['updates'],
          case['expected'])
         for name, case in HISTORY_CONTAINER_TEST_CASES.items()]
    )
    def test_history_container(self,
                               name,
                               specs,
                               sids,
                               dt,
                               updates,
                               expected):

        for spec in specs:
            # Sanity check on test input.
            self.assertEqual(len(expected[spec.key_str]), len(updates))

        container = HistoryContainer(
            {spec.key_str: spec for spec in specs}, sids, dt, 'minute',
            env=self.env,
        )

        for update_count, update in enumerate(updates):

            bar_dt = self.bar_data_dt(update)
            container.update(update, bar_dt)

            for spec in specs:
                pd.util.testing.assert_frame_equal(
                    container.get_history(spec, bar_dt),
                    expected[spec.key_str][update_count],
                    check_dtype=False,
                    check_column_type=True,
                    check_index_type=True,
                    check_frame_type=True,
                )

    def test_multiple_specs_on_same_bar(self):
        """
        Test that a ffill and non ffill spec both get
        the correct results when called on the same tick
        """
        spec = history.HistorySpec(
            bar_count=3,
            frequency='1m',
            field='price',
            ffill=True,
            data_frequency='minute',
            env=self.env,
        )
        no_fill_spec = history.HistorySpec(
            bar_count=3,
            frequency='1m',
            field='price',
            ffill=False,
            data_frequency='minute',
            env=self.env,
        )

        specs = {spec.key_str: spec, no_fill_spec.key_str: no_fill_spec}
        initial_sids = [1, ]
        initial_dt = pd.Timestamp(
            '2013-06-28 9:31AM', tz='US/Eastern').tz_convert('UTC')

        container = HistoryContainer(
            specs, initial_sids, initial_dt, 'minute', env=self.env,
        )

        bar_data = BarData()
        container.update(bar_data, initial_dt)
        # Add data on bar two of first day.
        second_bar_dt = pd.Timestamp(
            '2013-06-28 9:32AM', tz='US/Eastern').tz_convert('UTC')
        bar_data[1] = {
            'price': 10,
            'dt': second_bar_dt
        }
        container.update(bar_data, second_bar_dt)

        third_bar_dt = pd.Timestamp(
            '2013-06-28 9:33AM', tz='US/Eastern').tz_convert('UTC')

        del bar_data[1]

        # add nan for 3rd bar
        container.update(bar_data, third_bar_dt)
        prices = container.get_history(spec, third_bar_dt)
        no_fill_prices = container.get_history(no_fill_spec, third_bar_dt)
        self.assertEqual(prices.values[-1], 10)
        self.assertTrue(np.isnan(no_fill_prices.values[-1]),
                        "Last price should be np.nan")

    def test_container_nans_and_daily_roll(self):

        spec = history.HistorySpec(
            bar_count=3,
            frequency='1d',
            field='price',
            ffill=True,
            data_frequency='minute',
            env=self.env,
        )
        specs = {spec.key_str: spec}
        initial_sids = [1, ]
        initial_dt = pd.Timestamp(
            '2013-06-28 9:31AM', tz='US/Eastern').tz_convert('UTC')

        container = HistoryContainer(
            specs, initial_sids, initial_dt, 'minute', env=self.env,
        )

        bar_data = BarData()
        container.update(bar_data, initial_dt)
        # Since there was no backfill because of no db.
        # And no first bar of data, so all values should be nans.
        prices = container.get_history(spec, initial_dt)
        nan_values = np.isnan(prices[1])
        self.assertTrue(all(nan_values), nan_values)

        # Add data on bar two of first day.
        second_bar_dt = pd.Timestamp(
            '2013-06-28 9:32AM', tz='US/Eastern').tz_convert('UTC')

        bar_data[1] = {
            'price': 10,
            'dt': second_bar_dt
        }
        container.update(bar_data, second_bar_dt)

        prices = container.get_history(spec, second_bar_dt)
        # Prices should be
        #                             1
        # 2013-06-26 20:00:00+00:00 NaN
        # 2013-06-27 20:00:00+00:00 NaN
        # 2013-06-28 13:32:00+00:00  10

        self.assertTrue(np.isnan(prices[1].ix[0]))
        self.assertTrue(np.isnan(prices[1].ix[1]))
        self.assertEqual(prices[1].ix[2], 10)

        third_bar_dt = pd.Timestamp(
            '2013-06-28 9:33AM', tz='US/Eastern').tz_convert('UTC')

        del bar_data[1]

        container.update(bar_data, third_bar_dt)

        prices = container.get_history(spec, third_bar_dt)
        # The one should be forward filled

        # Prices should be
        #                             1
        # 2013-06-26 20:00:00+00:00 NaN
        # 2013-06-27 20:00:00+00:00 NaN
        # 2013-06-28 13:33:00+00:00  10

        self.assertEquals(prices[1][third_bar_dt], 10)

        # Note that we did not fill in data at the close.
        # There was a bug where a nan was being introduced because of the
        # last value of 'raw' data was used, instead of a ffilled close price.

        day_two_first_bar_dt = pd.Timestamp(
            '2013-07-01 9:31AM', tz='US/Eastern').tz_convert('UTC')

        bar_data[1] = {
            'price': 20,
            'dt': day_two_first_bar_dt
        }

        container.update(bar_data, day_two_first_bar_dt)

        prices = container.get_history(spec, day_two_first_bar_dt)

        # Prices Should Be

        #                              1
        # 2013-06-27 20:00:00+00:00  nan
        # 2013-06-28 20:00:00+00:00   10
        # 2013-07-01 13:31:00+00:00   20

        self.assertTrue(np.isnan(prices[1].ix[0]))
        self.assertEqual(prices[1].ix[1], 10)
        self.assertEqual(prices[1].ix[2], 20)

        # Clear out the bar data

        del bar_data[1]

        day_three_first_bar_dt = pd.Timestamp(
            '2013-07-02 9:31AM', tz='US/Eastern').tz_convert('UTC')

        container.update(bar_data, day_three_first_bar_dt)

        prices = container.get_history(spec, day_three_first_bar_dt)

        #                             1
        # 2013-06-28 20:00:00+00:00  10
        # 2013-07-01 20:00:00+00:00  20
        # 2013-07-02 13:31:00+00:00  20

        self.assertTrue(prices[1].ix[0], 10)
        self.assertTrue(prices[1].ix[1], 20)
        self.assertTrue(prices[1].ix[2], 20)

        day_four_first_bar_dt = pd.Timestamp(
            '2013-07-03 9:31AM', tz='US/Eastern').tz_convert('UTC')

        container.update(bar_data, day_four_first_bar_dt)

        prices = container.get_history(spec, day_four_first_bar_dt)

        #                             1
        # 2013-07-01 20:00:00+00:00  20
        # 2013-07-02 20:00:00+00:00  20
        # 2013-07-03 13:31:00+00:00  20

        self.assertEqual(prices[1].ix[0], 20)
        self.assertEqual(prices[1].ix[1], 20)
        self.assertEqual(prices[1].ix[2], 20)


class TestHistoryAlgo(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = trading.TradingEnvironment()
        cls.env.write_data(equities_identifiers=[0, 1])

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def setUp(self):
        np.random.seed(123)

    def test_history_daily(self):
        bar_count = 3
        algo_text = """
from zipline.api import history, add_history

def initialize(context):
    add_history(bar_count={bar_count}, frequency='1d', field='price')
    context.history_trace = []

def handle_data(context, data):
    prices = history(bar_count={bar_count}, frequency='1d', field='price')
    context.history_trace.append(prices)
""".format(bar_count=bar_count).strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-30', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end, data_frequency='daily', env=self.env,
        )

        _, df = factory.create_test_df_source(sim_params, self.env)
        df = df.astype(np.float64)
        source = DataFrameSource(df)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='daily',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        output = test_algo.run(source)
        self.assertIsNotNone(output)

        df.columns = self.env.asset_finder.retrieve_all(df.columns)

        for i, received in enumerate(test_algo.history_trace[bar_count - 1:]):
            expected = df.iloc[i:i + bar_count]
            assert_frame_equal(expected, received)

    def test_history_daily_data_1m_window(self):
        algo_text = """
from zipline.api import history, add_history

def initialize(context):
    add_history(bar_count=1, frequency='1m', field='price')

def handle_data(context, data):
    prices = history(bar_count=3, frequency='1d', field='price')
""".strip()

        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-30', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end)

        with self.assertRaises(IncompatibleHistoryFrequency):
            algo = TradingAlgorithm(
                script=algo_text,
                data_frequency='daily',
                sim_params=sim_params,
                env=TestHistoryAlgo.env,
            )
            source = RandomWalkSource(start=start, end=end)
            algo.run(source)

    def test_basic_history(self):
        algo_text = """
from zipline.api import history, add_history

def initialize(context):
    add_history(bar_count=2, frequency='1d', field='price')

def handle_data(context, data):
    prices = history(bar_count=2, frequency='1d', field='price')
    prices['prices_times_two'] = prices[1] * 2
    context.last_prices = prices
""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31
        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-21', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)
        self.assertIsNotNone(output)

        last_prices = test_algo.last_prices[0]
        oldest_dt = pd.Timestamp(
            '2006-03-20 4:00 PM', tz='US/Eastern').tz_convert('UTC')
        newest_dt = pd.Timestamp(
            '2006-03-21 4:00 PM', tz='US/Eastern').tz_convert('UTC')

        self.assertEquals(oldest_dt, last_prices.index[0])
        self.assertEquals(newest_dt, last_prices.index[-1])

        # Random, depends on seed
        self.assertEquals(139.36946942498648, last_prices[oldest_dt])
        self.assertEquals(180.15661995395106, last_prices[newest_dt])

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_history_in_bts_price_days(self, data_freq):
        """
        Test calling history() in before_trading_start()
        with daily price bars.
        """
        algo_text = """
from zipline.api import history

def initialize(context):
    context.first_bts_call = True

def before_trading_start(context, data):
    if not context.first_bts_call:
        prices_bts = history(bar_count=3, frequency='1d', field='price')
        context.prices_bts = prices_bts
    context.first_bts_call = False

def handle_data(context, data):
    prices_hd = history(bar_count=3, frequency='1d', field='price')
    context.prices_hd = prices_hd
""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31
        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-22', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end, data_frequency=data_freq)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency=data_freq,
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start, end=end, freq=data_freq)
        output = test_algo.run(source)
        self.assertIsNotNone(output)

        # Get the prices recorded by history() within handle_data()
        prices_hd = test_algo.prices_hd[0]
        # Get the prices recorded by history() within BTS
        prices_bts = test_algo.prices_bts[0]

        # before_trading_start() is timestamp'd to midnight prior to
        # the day's trading. Since no equity trades occur at midnight,
        # the price recorded for this time is forward filled from the
        # last trade - typically ~4pm the previous day. This results
        # in the OHLCV data recorded by history() in BTS lagging
        # that recorded by history in handle_data().
        # The trace of the pricing data from history() called within
        # handle_data() vs. BTS in the above algo is as follows:

        #  When called within handle_data()
        # ---------------------------------
        # 2006-03-20 21:00:00    139.369469
        # 2006-03-21 21:00:00    180.156620
        # 2006-03-22 21:00:00    221.344654

        #       When called within BTS
        # ---------------------------------
        # 2006-03-17 21:00:00           NaN
        # 2006-03-20 21:00:00    139.369469
        # 2006-03-22 00:00:00    180.156620

        # Get relevant Timestamps for the history() call within handle_data()
        oldest_hd_dt = pd.Timestamp(
            '2006-03-20 4:00 PM', tz='US/Eastern').tz_convert('UTC')
        penultimate_hd_dt = pd.Timestamp(
            '2006-03-21 4:00 PM', tz='US/Eastern').tz_convert('UTC')

        # Get relevant Timestamps for the history() call within BTS
        penultimate_bts_dt = pd.Timestamp(
            '2006-03-20 4:00 PM', tz='US/Eastern').tz_convert('UTC')
        newest_bts_dt = normalize_date(pd.Timestamp(
            '2006-03-22 04:00 PM', tz='US/Eastern').tz_convert('UTC'))

        if data_freq == 'daily':
            # If we're dealing with daily data, then we record
            # canonicalized timestamps, so make conversion here:
            oldest_hd_dt = normalize_date(oldest_hd_dt)
            penultimate_hd_dt = normalize_date(penultimate_hd_dt)
            penultimate_bts_dt = normalize_date(penultimate_bts_dt)

        self.assertEquals(prices_hd[oldest_hd_dt],
                          prices_bts[penultimate_bts_dt])
        self.assertEquals(prices_hd[penultimate_hd_dt],
                          prices_bts[newest_bts_dt])

    def test_history_in_bts_price_minutes(self):
        """
        Test calling history() in before_trading_start()
        with minutely price bars.
        """
        algo_text = """
from zipline.api import history

def initialize(context):
    context.first_bts_call = True

def before_trading_start(context, data):
    if not context.first_bts_call:
        price_bts = history(bar_count=1, frequency='1m', field='price')
        context.price_bts = price_bts
    context.first_bts_call = False

def handle_data(context, data):
    pass

""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31
        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-22', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start, end=end)
        output = test_algo.run(source)
        self.assertIsNotNone(output)

        # Get the prices recorded by history() within BTS
        price_bts_0 = test_algo.price_bts[0]
        price_bts_1 = test_algo.price_bts[1]

        # The prices recorded by history() in BTS should
        # be the closing price of the previous day, which are:
        #
        #          sid | close on 2006-03-21
        #         ----------------------------
        #           0  | 180.15661995395106
        #           1  | 578.41665003444723

        # These are not 'real' price values. They are the product of
        # RandonWalkSource, which produces random walk OHLCV timeseries. For a
        # given seed these values are deterministc.
        self.assertEquals(180.15661995395106, price_bts_0.ix[0])
        self.assertEquals(578.41665003444723, price_bts_1.ix[0])

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_history_in_bts_volume_days(self, data_freq):
        """
        Test calling history() in before_trading_start()
        with daily volume bars.
        """
        algo_text = """
from zipline.api import history

def initialize(context):
    context.first_bts_call = True

def before_trading_start(context, data):
    if not context.first_bts_call:
        volume_bts = history(bar_count=2, frequency='1d', field='volume')
        context.volume_bts = volume_bts
    context.first_bts_call = False

def handle_data(context, data):
    volume_hd = history(bar_count=2, frequency='1d', field='volume')
    context.volume_hd = volume_hd
""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31
        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-22', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end, data_frequency=data_freq)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency=data_freq,
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start, end=end, freq=data_freq)
        output = test_algo.run(source)
        self.assertIsNotNone(output)

        # Get the volume recorded by history() within handle_data()
        volume_hd_0 = test_algo.volume_hd[0]
        volume_hd_1 = test_algo.volume_hd[1]
        # Get the volume recorded by history() within BTS
        volume_bts_0 = test_algo.volume_bts[0]
        volume_bts_1 = test_algo.volume_bts[1]

        penultimate_hd_dt = pd.Timestamp(
            '2006-03-21 4:00 PM', tz='US/Eastern').tz_convert('UTC')
        # Midnight of the day on which BTS is invoked.
        newest_bts_dt = normalize_date(pd.Timestamp(
            '2006-03-22 04:00 PM', tz='US/Eastern').tz_convert('UTC'))

        if data_freq == 'daily':
            # If we're dealing with daily data, then we record
            # canonicalized timestamps, so make conversion here:
            penultimate_hd_dt = normalize_date(penultimate_hd_dt)

        # When history() is called in BTS, its 'current' volume value
        # should equal the sum of the previous day.
        self.assertEquals(volume_hd_0[penultimate_hd_dt],
                          volume_bts_0[newest_bts_dt])
        self.assertEquals(volume_hd_1[penultimate_hd_dt],
                          volume_bts_1[newest_bts_dt])

    def test_history_in_bts_volume_minutes(self):
        """
        Test calling history() in before_trading_start()
        with minutely volume bars.
        """
        algo_text = """
from zipline.api import history

def initialize(context):
    context.first_bts_call = True

def before_trading_start(context, data):
    if not context.first_bts_call:
        volume_bts = history(bar_count=2, frequency='1m', field='volume')
        context.volume_bts = volume_bts
    context.first_bts_call = False

def handle_data(context, data):
    pass
""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31
        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-22', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start, end=end)
        output = test_algo.run(source)
        self.assertIsNotNone(output)

        # Get the volumes recorded for sid 0 by history() within BTS
        volume_bts_0 = test_algo.volume_bts[0]
        # Get the volumes recorded for sid 1 by history() within BTS
        volume_bts_1 = test_algo.volume_bts[1]

        # The values recorded on 2006-03-22 by history() in BTS
        # should equal the final volume values for the trading
        # day 2006-03-21:
        #                             0       1
        #   2006-03-21 20:59:00  215548  439908
        #   2006-03-21 21:00:00  985645  664313
        #
        # Note: These are not 'real' volume values. They are the product of
        # RandonWalkSource, which produces random walk OHLCV timeseries. For a
        # given seed these values are deterministc.
        self.assertEquals(215548, volume_bts_0.ix[0])
        self.assertEquals(985645, volume_bts_0.ix[1])
        self.assertEquals(439908, volume_bts_1.ix[0])
        self.assertEquals(664313, volume_bts_1.ix[1])

    def test_basic_history_one_day(self):
        algo_text = """
from zipline.api import history, add_history

def initialize(context):
    add_history(bar_count=1, frequency='1d', field='price')

def handle_data(context, data):
    prices = history(bar_count=1, frequency='1d', field='price')
    context.last_prices = prices
""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-21', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)

        self.assertIsNotNone(output)

        last_prices = test_algo.last_prices[0]
        # oldest and newest should be the same if there is only 1 bar
        oldest_dt = pd.Timestamp(
            '2006-03-21 4:00 PM', tz='US/Eastern').tz_convert('UTC')
        newest_dt = pd.Timestamp(
            '2006-03-21 4:00 PM', tz='US/Eastern').tz_convert('UTC')

        self.assertEquals(oldest_dt, last_prices.index[0])
        self.assertEquals(newest_dt, last_prices.index[-1])

        # Random, depends on seed
        self.assertEquals(180.15661995395106, last_prices[oldest_dt])
        self.assertEquals(180.15661995395106, last_prices[newest_dt])

    def test_basic_history_positional_args(self):
        """
        Ensure that positional args work.
        """
        algo_text = """
from zipline.api import history, add_history

def initialize(context):
    add_history(2, '1d', 'price')

def handle_data(context, data):

    prices = history(2, '1d', 'price')
    context.last_prices = prices
""".strip()

        #      March 2006
        # Su Mo Tu We Th Fr Sa
        #          1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        start = pd.Timestamp('2006-03-20', tz='UTC')
        end = pd.Timestamp('2006-03-21', tz='UTC')

        sim_params = factory.create_simulation_parameters(
            start=start, end=end)

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)
        self.assertIsNotNone(output)

        last_prices = test_algo.last_prices[0]
        oldest_dt = pd.Timestamp(
            '2006-03-20 4:00 PM', tz='US/Eastern').tz_convert('UTC')
        newest_dt = pd.Timestamp(
            '2006-03-21 4:00 PM', tz='US/Eastern').tz_convert('UTC')

        self.assertEquals(oldest_dt, last_prices.index[0])
        self.assertEquals(newest_dt, last_prices.index[-1])

        self.assertEquals(139.36946942498648, last_prices[oldest_dt])
        self.assertEquals(180.15661995395106, last_prices[newest_dt])

    def test_history_with_volume(self):
        algo_text = """
from zipline.api import history, add_history, record

def initialize(context):
    add_history(3, '1d', 'volume')

def handle_data(context, data):
    volume = history(3, '1d', 'volume')

    record(current_volume=volume[0].ix[-1])
""".strip()

        #      April 2007
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        start = pd.Timestamp('2007-04-10', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='minute'
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)

        np.testing.assert_equal(output.ix[0, 'current_volume'],
                                212218404.0)

    def test_history_with_high(self):
        algo_text = """
from zipline.api import history, add_history, record

def initialize(context):
    add_history(3, '1d', 'high')

def handle_data(context, data):
    highs = history(3, '1d', 'high')

    record(current_high=highs[0].ix[-1])
""".strip()

        #      April 2007
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        start = pd.Timestamp('2007-04-10', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='minute'
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)

        np.testing.assert_equal(output.ix[0, 'current_high'],
                                139.5370641791925)

    def test_history_with_low(self):
        algo_text = """
from zipline.api import history, add_history, record

def initialize(context):
    add_history(3, '1d', 'low')

def handle_data(context, data):
    lows = history(3, '1d', 'low')

    record(current_low=lows[0].ix[-1])
""".strip()

        #      April 2007
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        start = pd.Timestamp('2007-04-10', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='minute'
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)

        np.testing.assert_equal(output.ix[0, 'current_low'],
                                99.891436939669944)

    def test_history_with_open(self):
        algo_text = """
from zipline.api import history, add_history, record

def initialize(context):
    add_history(3, '1d', 'open_price')

def handle_data(context, data):
    opens = history(3, '1d', 'open_price')

    record(current_open=opens[0].ix[-1])
""".strip()

        #      April 2007
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        start = pd.Timestamp('2007-04-10', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='minute'
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)

        np.testing.assert_equal(output.ix[0, 'current_open'],
                                99.991436939669939)

    def test_history_passed_to_func(self):
        """
        Had an issue where MagicMock was causing errors during validation
        with rolling mean.
        """
        algo_text = """
from zipline.api import history, add_history
import pandas as pd

def initialize(context):
    add_history(2, '1d', 'price')

def handle_data(context, data):
    prices = history(2, '1d', 'price')

    pd.rolling_mean(prices, 2)
""".strip()

        #      April 2007
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        start = pd.Timestamp('2007-04-10', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='minute'
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)

        # At this point, just ensure that there is no crash.
        self.assertIsNotNone(output)

    def test_history_passed_to_talib(self):
        """
        Had an issue where MagicMock was causing errors during validation
        with talib.

        We don't officially support a talib integration, yet.
        But using talib directly should work.
        """
        algo_text = """
import talib
import numpy as np

from zipline.api import history, add_history, record

def initialize(context):
    add_history(2, '1d', 'price')

def handle_data(context, data):
    prices = history(2, '1d', 'price')

    ma_result = talib.MA(np.asarray(prices[0]), timeperiod=2)
    record(ma=ma_result[-1])
""".strip()

        #      April 2007
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        # Eddie: this was set to 04-10 but I don't see how that makes
        # sense as it does not generate enough data to get at -2 index
        # below.
        start = pd.Timestamp('2007-04-05', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='daily'
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start,
                                  end=end)
        output = test_algo.run(source)
        # At this point, just ensure that there is no crash.
        self.assertIsNotNone(output)

        recorded_ma = output.ix[-2, 'ma']

        self.assertFalse(pd.isnull(recorded_ma))
        # Depends on seed
        np.testing.assert_almost_equal(recorded_ma,
                                       159.76304468946876)

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_history_container_constructed_at_runtime(self, data_freq):
        algo_text = dedent(
            """\
            from zipline.api import history
            def handle_data(context, data):
                context.prices = history(2, '1d', 'price')
            """
        )
        start = pd.Timestamp('2007-04-05', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency=data_freq,
            emission_rate=data_freq
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency=data_freq,
            sim_params=sim_params,
            env=TestHistoryAlgo.env,
        )

        source = RandomWalkSource(start=start, end=end, freq=data_freq)

        self.assertIsNone(test_algo.history_container)
        test_algo.run(source)
        self.assertIsNotNone(
            test_algo.history_container,
            msg='HistoryContainer was not constructed at runtime',
        )

        container = test_algo.history_container
        self.assertEqual(
            len(container.digest_panels),
            1,
            msg='The HistoryContainer created too many digest panels',
        )

        freq, digest = list(container.digest_panels.items())[0]
        self.assertEqual(
            freq.unit_str,
            'd',
        )

        self.assertEqual(
            digest.window_length,
            1,
            msg='The digest panel is not large enough to service the given'
            ' HistorySpec',
        )

    def test_history_in_initialize(self):
        algo_text = dedent(
            """\
            from zipline.api import history

            def initialize(context):
                history(10, '1d', 'price')

            def handle_data(context, data):
                pass
            """
        )

        start = pd.Timestamp('2007-04-05', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='daily',
            env=self.env,
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=self.env,
        )

        with self.assertRaises(HistoryInInitialize):
            test_algo.initialize()

    @parameterized.expand([
        (1,),
        (2,),
    ])
    def test_history_grow_length_inter_bar(self, incr):
        """
        Tests growing the length of a digest panel with different date_buf
        deltas once per bar.
        """
        algo_text = dedent(
            """\
            from zipline.api import history


            def initialize(context):
                context.bar_count = 1


            def handle_data(context, data):
                prices = history(context.bar_count, '1d', 'price')
                context.test_case.assertEqual(len(prices), context.bar_count)
                context.bar_count += {incr}
            """
        ).format(incr=incr)
        start = pd.Timestamp('2007-04-05', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='daily',
            env=self.env,
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=self.env,
        )
        test_algo.test_case = self

        source = RandomWalkSource(start=start, end=end)

        self.assertIsNone(test_algo.history_container)
        test_algo.run(source)

    @parameterized.expand([
        (1,),
        (2,),
    ])
    def test_history_grow_length_intra_bar(self, incr):
        """
        Tests growing the length of a digest panel with different date_buf
        deltas in a single bar.
        """
        algo_text = dedent(
            """\
            from zipline.api import history


            def initialize(context):
                context.bar_count = 1


            def handle_data(context, data):
                prices = history(context.bar_count, '1d', 'price')
                context.test_case.assertEqual(len(prices), context.bar_count)
                context.bar_count += {incr}
                prices = history(context.bar_count, '1d', 'price')
                context.test_case.assertEqual(len(prices), context.bar_count)
            """
        ).format(incr=incr)
        start = pd.Timestamp('2007-04-05', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='daily',
            env=self.env,
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=self.env,
        )
        test_algo.test_case = self

        source = RandomWalkSource(start=start, end=end)

        self.assertIsNone(test_algo.history_container)
        test_algo.run(source)


class TestHistoryContainerResize(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()

    @classmethod
    def tearDownClass(cls):
        del cls.env

    @subtest(
        ((freq, field, data_frequency, construct_digest)
         for freq in ('1m', '1d')
         for field in HistoryContainer.VALID_FIELDS
         for data_frequency in ('minute', 'daily')
         for construct_digest in (True, False)
         if not (freq == '1m' and data_frequency == 'daily')),
        'freq',
        'field',
        'data_frequency',
        'construct_digest',
    )
    def test_history_grow_length(self,
                                 freq,
                                 field,
                                 data_frequency,
                                 construct_digest):
        bar_count = 2 if construct_digest else 1
        spec = history.HistorySpec(
            bar_count=bar_count,
            frequency=freq,
            field=field,
            ffill=True,
            data_frequency=data_frequency,
            env=self.env,
        )
        specs = {spec.key_str: spec}
        initial_sids = [1]
        initial_dt = pd.Timestamp(
            '2013-06-28 13:31'
            if data_frequency == 'minute'
            else '2013-06-28 12:00AM',
            tz='UTC',
        )

        container = HistoryContainer(
            specs, initial_sids, initial_dt, data_frequency, env=self.env,
        )

        if construct_digest:
            self.assertEqual(
                container.digest_panels[spec.frequency].window_length, 1,
            )

        bar_data = BarData()
        container.update(bar_data, initial_dt)

        to_add = (
            history.HistorySpec(
                bar_count=bar_count + 1,
                frequency=freq,
                field=field,
                ffill=True,
                data_frequency=data_frequency,
                env=self.env,
            ),
            history.HistorySpec(
                bar_count=bar_count + 2,
                frequency=freq,
                field=field,
                ffill=True,
                data_frequency=data_frequency,
                env=self.env,
            ),
        )

        for spec in to_add:
            container.ensure_spec(spec, initial_dt, bar_data)

            self.assertEqual(
                container.digest_panels[spec.frequency].window_length,
                spec.bar_count - 1,
            )

            self.assert_history(container, spec, initial_dt)

    @subtest(
        ((bar_count, freq, pair, data_frequency)
         for bar_count in (1, 2)
         for freq in ('1m', '1d')
         for pair in product(HistoryContainer.VALID_FIELDS, repeat=2)
         for data_frequency in ('minute', 'daily')
         if not (freq == '1m' and data_frequency == 'daily')),
        'bar_count',
        'freq',
        'pair',
        'data_frequency',
    )
    def test_history_add_field(self, bar_count, freq, pair, data_frequency):
        first, second = pair
        spec = history.HistorySpec(
            bar_count=bar_count,
            frequency=freq,
            field=first,
            ffill=True,
            data_frequency=data_frequency,
            env=self.env,
        )
        specs = {spec.key_str: spec}
        initial_sids = [1]
        initial_dt = pd.Timestamp(
            '2013-06-28 13:31'
            if data_frequency == 'minute'
            else '2013-06-28 12:00AM',
            tz='UTC',
        )

        container = HistoryContainer(
            specs, initial_sids, initial_dt, data_frequency, env=self.env
        )

        if bar_count > 1:
            self.assertEqual(
                container.digest_panels[spec.frequency].window_length, 1,
            )

        bar_data = BarData()
        container.update(bar_data, initial_dt)

        new_spec = history.HistorySpec(
            bar_count,
            frequency=freq,
            field=second,
            ffill=True,
            data_frequency=data_frequency,
            env=self.env,
        )

        container.ensure_spec(new_spec, initial_dt, bar_data)

        if bar_count > 1:
            digest_panel = container.digest_panels[new_spec.frequency]
            self.assertEqual(digest_panel.window_length, bar_count - 1)
            self.assertIn(second, digest_panel.items)
        else:
            self.assertNotIn(new_spec.frequency, container.digest_panels)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            self.assert_history(container, new_spec, initial_dt)

    @subtest(
        ((bar_count, pair, field, data_frequency)
         for bar_count in (1, 2)
         for pair in product(('1m', '1d'), repeat=2)
         for field in HistoryContainer.VALID_FIELDS
         for data_frequency in ('minute', 'daily')
         if not ('1m' in pair and data_frequency == 'daily')),
        'bar_count',
        'pair',
        'field',
        'data_frequency',
    )
    def test_history_add_freq(self, bar_count, pair, field, data_frequency):
        first, second = pair
        spec = history.HistorySpec(
            bar_count=bar_count,
            frequency=first,
            field=field,
            ffill=True,
            data_frequency=data_frequency,
            env=self.env,
        )
        specs = {spec.key_str: spec}
        initial_sids = [1]
        initial_dt = pd.Timestamp(
            '2013-06-28 13:31'
            if data_frequency == 'minute'
            else '2013-06-28 12:00AM',
            tz='UTC',
        )

        container = HistoryContainer(
            specs, initial_sids, initial_dt, data_frequency, env=self.env,
        )

        if bar_count > 1:
            self.assertEqual(
                container.digest_panels[spec.frequency].window_length, 1,
            )

        bar_data = BarData()
        container.update(bar_data, initial_dt)

        new_spec = history.HistorySpec(
            bar_count,
            frequency=second,
            field=field,
            ffill=True,
            data_frequency=data_frequency,
            env=self.env,
        )

        container.ensure_spec(new_spec, initial_dt, bar_data)

        if bar_count > 1:
            digest_panel = container.digest_panels[new_spec.frequency]
            self.assertEqual(digest_panel.window_length, bar_count - 1)
        else:
            self.assertNotIn(new_spec.frequency, container.digest_panels)

        self.assert_history(container, new_spec, initial_dt)

    def assert_history(self, container, spec, dt):
        hst = container.get_history(spec, dt)

        self.assertEqual(len(hst), spec.bar_count)

        back = spec.frequency.prev_bar
        for n in reversed(hst.index):
            self.assertEqual(dt, n)
            dt = back(dt)
