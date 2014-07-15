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

from nose_parameterized import parameterized
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from zipline.history import history
from zipline.history.history_container import HistoryContainer
from zipline.protocol import BarData
import zipline.utils.factory as factory
from zipline import TradingAlgorithm
from zipline.finance.trading import SimulationParameters, TradingEnvironment

from zipline.sources import RandomWalkSource, DataFrameSource

from .history_cases import (
    HISTORY_CONTAINER_TEST_CASES,
)

# Cases are over the July 4th holiday, to ensure use of trading calendar.

#      March 2013
# Su Mo Tu We Th Fr Sa
#                 1  2
#  3  4  5  6  7  8  9
# 10 11 12 13 14 15 16
# 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30
# 31
#      April 2013
# Su Mo Tu We Th Fr Sa
#     1  2  3  4  5  6
#  7  8  9 10 11 12 13
# 14 15 16 17 18 19 20
# 21 22 23 24 25 26 27
# 28 29 30
#
#       May 2013
# Su Mo Tu We Th Fr Sa
#           1  2  3  4
#  5  6  7  8  9 10 11
# 12 13 14 15 16 17 18
# 19 20 21 22 23 24 25
# 26 27 28 29 30 31
#
#      June 2013
# Su Mo Tu We Th Fr Sa
#                    1
#  2  3  4  5  6  7  8
#  9 10 11 12 13 14 15
# 16 17 18 19 20 21 22
# 23 24 25 26 27 28 29
# 30
#      July 2013
# Su Mo Tu We Th Fr Sa
#     1  2  3  4  5  6
#  7  8  9 10 11 12 13
# 14 15 16 17 18 19 20
# 21 22 23 24 25 26 27
# 28 29 30 31
#
# Times to be converted via:
# pd.Timestamp('2013-07-05 9:31', tz='US/Eastern').tz_convert('UTC')},

INDEX_TEST_CASES_RAW = {
    'week of daily data': {
        'input': {'bar_count': 5,
                  'frequency': '1d',
                  'algo_dt': '2013-07-05 9:31AM'},
        'expected': [
            '2013-06-28 4:00PM',
            '2013-07-01 4:00PM',
            '2013-07-02 4:00PM',
            '2013-07-03 1:00PM',
            '2013-07-05 9:31AM',
        ]
    },
    'five minutes on july 5th open': {
        'input': {'bar_count': 5,
                  'frequency': '1m',
                  'algo_dt': '2013-07-05 9:31AM'},
        'expected': [
            '2013-07-03 12:57PM',
            '2013-07-03 12:58PM',
            '2013-07-03 12:59PM',
            '2013-07-03 1:00PM',
            '2013-07-05 9:31AM',
        ]
    },
}


def to_timestamp(dt_str):
    return pd.Timestamp(dt_str, tz='US/Eastern').tz_convert('UTC')


def convert_cases(cases):
    """
    Convert raw strings to values comparable with system data.
    """
    cases = cases.copy()
    for case in cases.values():
        case['input']['algo_dt'] = to_timestamp(case['input']['algo_dt'])
        case['expected'] = pd.DatetimeIndex([to_timestamp(dt_str) for dt_str
                                             in case['expected']])
    return cases

INDEX_TEST_CASES = convert_cases(INDEX_TEST_CASES_RAW)


def get_index_at_dt(case_input):
    history_spec = history.HistorySpec(
        case_input['bar_count'],
        case_input['frequency'],
        None,
        False,
        daily_at_midnight=False
    )
    return history.index_at_dt(history_spec, case_input['algo_dt'])


class TestHistoryIndex(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.environment = TradingEnvironment.instance()

    @parameterized.expand(
        [(name, case['input'], case['expected'])
         for name, case in INDEX_TEST_CASES.items()]
    )
    def test_index_at_dt(self, name, case_input, expected):
        history_index = get_index_at_dt(case_input)

        history_series = pd.Series(index=history_index)
        expected_series = pd.Series(index=expected)

        pd.util.testing.assert_series_equal(history_series, expected_series)


class TestHistoryContainer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment.instance()

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
            {spec.key_str: spec for spec in specs}, sids, dt
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

    def test_container_nans_and_daily_roll(self):

        spec = history.HistorySpec(
            bar_count=3,
            frequency='1d',
            field='price',
            ffill=True,
            daily_at_midnight=False
        )
        specs = {spec.key_str: spec}
        initial_sids = [1, ]
        initial_dt = pd.Timestamp(
            '2013-06-28 9:31AM', tz='US/Eastern').tz_convert('UTC')

        container = HistoryContainer(
            specs, initial_sids, initial_dt)

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
    def setUp(self):
        np.random.seed(123)

    def test_history_daily(self):
        bar_count = 3
        algo_text = """
from zipline.api import history, add_history
from copy import deepcopy

def initialize(context):
    add_history(bar_count={bar_count}, frequency='1d', field='price')
    context.history_trace = []

def handle_data(context, data):
    prices = history(bar_count={bar_count}, frequency='1d', field='price')
    context.history_trace.append(deepcopy(prices))
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
            start=start, end=end)

        _, df = factory.create_test_df_source(sim_params)
        df = df.astype(np.float64)
        source = DataFrameSource(df, sids=[0])

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='daily',
            sim_params=sim_params
        )

        output = test_algo.run(source)
        self.assertIsNotNone(output)

        history_trace = test_algo.history_trace

        for i, received in enumerate(history_trace[bar_count - 1:]):
            expected = df.iloc[i:i + bar_count]
            assert_frame_equal(expected, received)

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
            sim_params=sim_params
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
            sim_params=sim_params
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
import copy
from zipline.api import history, add_history

def initialize(context):
    add_history(2, '1d', 'price')

def handle_data(context, data):

    prices = history(2, '1d', 'price')
    context.last_prices = copy.deepcopy(prices)
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
            sim_params=sim_params
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
            sim_params=sim_params
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
            sim_params=sim_params
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
            sim_params=sim_params
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
            sim_params=sim_params
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
            sim_params=sim_params
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
            sim_params=sim_params
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
