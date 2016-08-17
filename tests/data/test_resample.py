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
from collections import OrderedDict
from numbers import Real

from nose_parameterized import parameterized
from numpy.testing import assert_almost_equal
from numpy import nan, array
import pandas as pd
from pandas import DataFrame
from six import iteritems

from zipline.data.resample import (
    minute_to_session,
    DailyHistoryAggregator,
    MinuteResampleSessionBarReader,
)

from zipline.testing.fixtures import (
    WithEquityMinuteBarData,
    WithBcolzEquityMinuteBarReader,
    WithBcolzFutureMinuteBarReader,
    ZiplineTestCase,
)

OHLC = ['open', 'high', 'low', 'close']
OHLCV = OHLC + ['volume']


NYSE_MINUTES = OrderedDict((
    ('day_0_front', pd.date_range('2016-03-15 9:31',
                                  '2016-03-15 9:33',
                                  freq='min',
                                  tz='US/Eastern').tz_convert('UTC')),
    ('day_0_back', pd.date_range('2016-03-15 15:58',
                                 '2016-03-15 16:00',
                                 freq='min',
                                 tz='US/Eastern').tz_convert('UTC')),
    ('day_1_front', pd.date_range('2016-03-16 9:31',
                                  '2016-03-16 9:33',
                                  freq='min',
                                  tz='US/Eastern').tz_convert('UTC')),
    ('day_1_back', pd.date_range('2016-03-16 15:58',
                                 '2016-03-16 16:00',
                                 freq='min',
                                 tz='US/Eastern').tz_convert('UTC')),
))


CME_MINUTES = OrderedDict((
    ('day_0_front', pd.date_range('2016-03-15 18:01',
                                  '2016-03-15 18:03',
                                  freq='min',
                                  tz='US/Eastern').tz_convert('UTC')),
    ('day_0_back', pd.date_range('2016-03-16 17:58',
                                 '2016-03-16 18:00',
                                 freq='min',
                                 tz='US/Eastern').tz_convert('UTC')),
    ('day_1_front', pd.date_range('2016-03-16 18:01',
                                  '2016-03-16 18:03',
                                  freq='min',
                                  tz='US/Eastern').tz_convert('UTC')),
    ('day_1_back', pd.date_range('2016-03-17 17:58',
                                 '2016-03-17 18:00',
                                 freq='min',
                                 tz='US/Eastern').tz_convert('UTC')),
))


SCENARIOS = OrderedDict((
    ('none_missing', array([
        [101.5, 101.9, 101.1, 101.3, 1001],
        [103.5, 103.9, 103.1, 103.3, 1003],
        [102.5, 102.9, 102.1, 102.3, 1002],
    ])),
    ('all_missing', array([
        [nan, nan, nan, nan, 0],
        [nan, nan, nan, nan, 0],
        [nan, nan, nan, nan, 0],
    ])),
    ('missing_first', array([
        [nan,     nan,   nan,   nan,     0],
        [103.5, 103.9, 103.1, 103.3,  1003],
        [102.5, 102.9, 102.1, 102.3,  1002],
    ])),
    ('missing_last', array([
        [107.5, 107.9, 107.1, 107.3,  1007],
        [108.5, 108.9, 108.1, 108.3,  1008],
        [nan,     nan,   nan,   nan,     0],
    ])),
    ('missing_middle', array([
        [103.5, 103.9, 103.1, 103.3, 1003],
        [nan,     nan,   nan,   nan,    0],
        [102.5, 102.5, 102.1, 102.3, 1002],
    ])),
))

OHLCV = ('open', 'high', 'low', 'close', 'volume')

_EQUITY_CASES = (
    (1, (('none_missing', 'day_0_front'),
         ('none_missing', 'day_0_back'))),
    (2, (('missing_first', 'day_0_front'),
         ('none_missing', 'day_0_back'))),
    (3, (('missing_last', 'day_0_back'),
         ('missing_first', 'day_1_front'))),
)

EQUITY_CASES = OrderedDict()

for sid, combos in _EQUITY_CASES:
    frames = [DataFrame(SCENARIOS[s], columns=OHLCV).
              set_index(NYSE_MINUTES[m])
              for s, m in combos]
    EQUITY_CASES[sid] = pd.concat(frames)

_FUTURE_CASES = (
    (1001, (('none_missing', 'day_0_front'),
            ('none_missing', 'day_0_back'))),
    (1002, (('missing_first', 'day_0_front'),
            ('none_missing', 'day_0_back'))),
    (1003, (('missing_last', 'day_0_back'),
            ('missing_first', 'day_1_front'))),
)

FUTURE_CASES = OrderedDict()

for sid, combos in _FUTURE_CASES:
    frames = [DataFrame(SCENARIOS[s], columns=OHLCV).
              set_index(CME_MINUTES[m])
              for s, m in combos]
    FUTURE_CASES[sid] = pd.concat(frames)


EXPECTED_AGGREGATION = {
    1: DataFrame({
        'open': [101.5, 101.5, 101.5, 101.5, 101.5, 101.5],
        'high': [101.9, 103.9, 103.9, 103.9, 103.9, 103.9],
        'low': [101.1, 101.1, 101.1, 101.1, 101.1, 101.1],
        'close': [101.3, 103.3, 102.3, 101.3, 103.3, 102.3],
        'volume': [1001, 2004, 3006, 4007, 5010, 6012],
    }, columns=OHLCV),
    2: DataFrame({
        'open': [nan, 103.5, 103.5, 103.5, 103.5, 103.5],
        'high': [nan, 103.9, 103.9, 103.9, 103.9, 103.9],
        'low': [nan, 103.1, 102.1, 101.1, 101.1, 101.1],
        'close': [nan, 103.3, 102.3, 101.3, 103.3, 102.3],
        'volume': [0, 1003, 2005, 3006, 4009, 5011],
    }, columns=OHLCV),
    # Equity 3 straddles two days.
    3: DataFrame({
        'open': [107.5, 107.5, 107.5, nan, 103.5, 103.5],
        'high': [107.9, 108.9, 108.9, nan, 103.9, 103.9],
        'low': [107.1, 107.1, 107.1, nan, 103.1, 102.1],
        'close': [107.3, 108.3, 108.3, nan, 103.3, 102.3],
        'volume': [1007, 2015, 2015, 0, 1003, 2005],
    }, columns=OHLCV),
    1001: DataFrame({
        'open': [101.5, 101.5, 101.5, 101.5, 101.5, 101.5],
        'high': [101.9, 103.9, 103.9, 103.9, 103.9, 103.9],
        'low': [101.1, 101.1, 101.1, 101.1, 101.1, 101.1],
        'close': [101.3, 103.3, 102.3, 101.3, 103.3, 102.3],
        'volume': [1001, 2004, 3006, 4007, 5010, 6012],
    }, columns=OHLCV),
    1002: DataFrame({
        'open': [nan, 103.5, 103.5, 103.5, 103.5, 103.5],
        'high': [nan, 103.9, 103.9, 103.9, 103.9, 103.9],
        'low': [nan, 103.1, 102.1, 101.1, 101.1, 101.1],
        'close': [nan, 103.3, 102.3, 101.3, 103.3, 102.3],
        'volume': [0, 1003, 2005, 3006, 4009, 5011],
    }, columns=OHLCV),
    # Equity 3 straddles two days.
    1003: DataFrame({
        'open': [107.5, 107.5, 107.5, nan, 103.5, 103.5],
        'high': [107.9, 108.9, 108.9, nan, 103.9, 103.9],
        'low': [107.1, 107.1, 107.1, nan, 103.1, 102.1],
        'close': [107.3, 108.3, 108.3, nan, 103.3, 102.3],
        'volume': [1007, 2015, 2015, 0, 1003, 2005],
    }, columns=OHLCV),
}

EXPECTED_SESSIONS = {
    1: DataFrame([EXPECTED_AGGREGATION[1].iloc[-1].values],
                 columns=OHLCV,
                 index=pd.to_datetime(['2016-03-15'], utc=True)),
    2: DataFrame([EXPECTED_AGGREGATION[2].iloc[-1].values],
                 columns=OHLCV,
                 index=pd.to_datetime(['2016-03-15'], utc=True)),
    3: DataFrame(EXPECTED_AGGREGATION[3].iloc[[2, 5]].values,
                 columns=OHLCV,
                 index=pd.to_datetime(['2016-03-15', '2016-03-16'], utc=True)),
    1001: DataFrame([EXPECTED_AGGREGATION[1001].iloc[-1].values],
                    columns=OHLCV,
                    index=pd.to_datetime(['2016-03-16'], utc=True)),
    1002: DataFrame([EXPECTED_AGGREGATION[1002].iloc[-1].values],
                    columns=OHLCV,
                    index=pd.to_datetime(['2016-03-16'], utc=True)),
    1003: DataFrame(EXPECTED_AGGREGATION[1003].iloc[[2, 5]].values,
                    columns=OHLCV,
                    index=pd.to_datetime(['2016-03-16', '2016-03-17'],
                                         utc=True))
}


class MinuteToDailyAggregationTestCase(WithBcolzEquityMinuteBarReader,
                                       ZiplineTestCase):

    #    March 2016
    # Su Mo Tu We Th Fr Sa
    #        1  2  3  4  5
    #  6  7  8  9 10 11 12
    # 13 14 15 16 17 18 19
    # 20 21 22 23 24 25 26
    # 27 28 29 30 31

    TRADING_ENV_MIN_DATE = START_DATE = pd.Timestamp(
        '2016-03-01', tz='UTC',
    )
    TRADING_ENV_MAX_DATE = END_DATE = pd.Timestamp(
        '2016-03-31', tz='UTC',
    )
    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    @classmethod
    def make_equity_minute_bar_data(cls):
        for sid in cls.ASSET_FINDER_EQUITY_SIDS:
            frame = EQUITY_CASES[sid]
            yield sid, frame

    def init_instance_fixtures(self):
        super(MinuteToDailyAggregationTestCase, self).init_instance_fixtures()
        # Set up a fresh data portal for each test, since order of calling
        # needs to be tested.
        self.equity_daily_aggregator = DailyHistoryAggregator(
            self.trading_calendar.schedule.market_open,
            self.bcolz_equity_minute_bar_reader,
            self.trading_calendar
        )

    @parameterized.expand([
        ('open_sid_1', 'open', 1),
        ('high_1', 'high', 1),
        ('low_1', 'low', 1),
        ('close_1', 'close', 1),
        ('volume_1', 'volume', 1),
        ('open_2', 'open', 2),
        ('high_2', 'high', 2),
        ('low_2', 'low', 2),
        ('close_2', 'close', 2),
        ('volume_2', 'volume', 2),
        ('open_3', 'open', 3),
        ('high_3', 'high', 3),
        ('low_3', 'low', 3),
        ('close_3', 'close', 3),
        ('volume_3', 'volume', 3),
    ])
    def test_contiguous_minutes_individual(self, name, field, sid):
        # First test each minute in order.
        method_name = field + 's'
        results = []
        repeat_results = []
        asset = self.asset_finder.retrieve_asset(sid)
        minutes = EQUITY_CASES[asset].index
        for minute in minutes:
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            results.append(value)

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            repeat_results.append(value)

        assert_almost_equal(results, EXPECTED_AGGREGATION[asset][field],
                            err_msg='sid={0} field={1}'.format(asset, field))
        assert_almost_equal(repeat_results, EXPECTED_AGGREGATION[asset][field],
                            err_msg='sid={0} field={1}'.format(asset, field))

    @parameterized.expand([
        ('open_sid_1', 'open', 1),
        ('high_1', 'high', 1),
        ('low_1', 'low', 1),
        ('close_1', 'close', 1),
        ('volume_1', 'volume', 1),
        ('open_2', 'open', 2),
        ('high_2', 'high', 2),
        ('low_2', 'low', 2),
        ('close_2', 'close', 2),
        ('volume_2', 'volume', 2),
        ('open_3', 'open', 3),
        ('high_3', 'high', 3),
        ('low_3', 'low', 3),
        ('close_3', 'close', 3),
        ('volume_3', 'volume', 3),
    ])
    def test_skip_minutes_individual(self, name, field, sid):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        asset = self.asset_finder.retrieve_asset(sid)
        minutes = EQUITY_CASES[asset].index
        for i in [1, 5]:
            minute = minutes[i]
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            assert_almost_equal(value,
                                EXPECTED_AGGREGATION[sid][field][i],
                                err_msg='sid={0} field={1} dt={2}'.format(
                                    sid, field, minute))

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            assert_almost_equal(value,
                                EXPECTED_AGGREGATION[sid][field][i],
                                err_msg='sid={0} field={1} dt={2}'.format(
                                    sid, field, minute))

    @parameterized.expand(OHLCV)
    def test_contiguous_minutes_multiple(self, field):
        # First test each minute in order.
        method_name = field + 's'
        assets = self.asset_finder.retrieve_all([1, 2])
        results = {asset: [] for asset in assets}
        repeat_results = {asset: [] for asset in assets}
        minutes = EQUITY_CASES[1].index
        for minute in minutes:
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                results[asset].append(value)

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                repeat_results[asset].append(value)
        for asset in assets:
            assert_almost_equal(results[asset],
                                EXPECTED_AGGREGATION[asset][field],
                                err_msg='sid={0} field={1}'.format(
                                    asset, field))
            assert_almost_equal(repeat_results[asset],
                                EXPECTED_AGGREGATION[asset][field],
                                err_msg='sid={0} field={1}'.format(
                                    asset, field))

    @parameterized.expand(OHLCV)
    def test_skip_minutes_multiple(self, field):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        assets = self.asset_finder.retrieve_all([1, 2])
        minutes = EQUITY_CASES[1].index
        for i in [1, 5]:
            minute = minutes[i]
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                assert_almost_equal(
                    value,
                    EXPECTED_AGGREGATION[asset][field][i],
                    err_msg='sid={0} field={1} dt={2}'.format(
                        asset, field, minute))

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                assert_almost_equal(
                    value,
                    EXPECTED_AGGREGATION[asset][field][i],
                    err_msg='sid={0} field={1} dt={2}'.format(
                        asset, field, minute))


class TestMinuteToSession(WithEquityMinuteBarData,
                          ZiplineTestCase):

    #    March 2016
    # Su Mo Tu We Th Fr Sa
    #        1  2  3  4  5
    #  6  7  8  9 10 11 12
    # 13 14 15 16 17 18 19
    # 20 21 22 23 24 25 26
    # 27 28 29 30 31

    START_DATE = pd.Timestamp(
        '2016-03-15', tz='UTC',
    )
    END_DATE = pd.Timestamp(
        '2016-03-15', tz='UTC',
    )
    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    @classmethod
    def make_equity_minute_bar_data(cls):
        for sid, frame in iteritems(EQUITY_CASES):
            yield sid, frame

    @classmethod
    def init_class_fixtures(cls):
        super(TestMinuteToSession, cls).init_class_fixtures()
        cls.equity_frames = {
            sid: frame for sid, frame in cls.make_equity_minute_bar_data()}

    def test_minute_to_session(self):
        for sid in self.ASSET_FINDER_EQUITY_SIDS:
            frame = self.equity_frames[sid]
            expected = EXPECTED_SESSIONS[sid]
            result = minute_to_session(frame, self.nyse_calendar)
            assert_almost_equal(expected.values,
                                result.values,
                                err_msg='sid={0}'.format(sid))


class TestResampleSessionBars(WithBcolzFutureMinuteBarReader,
                              ZiplineTestCase):

    TRADING_CALENDAR_STRS = ('CME',)
    TRADING_CALENDAR_PRIMARY_CAL = 'CME'

    ASSET_FINDER_FUTURE_SIDS = 1001, 1002, 1003

    START_DATE = pd.Timestamp('2016-03-16', tz='UTC')
    END_DATE = pd.Timestamp('2016-03-17', tz='UTC')
    NUM_SESSIONS = 2

    @classmethod
    def make_future_minute_bar_data(cls):
        for sid in cls.ASSET_FINDER_FUTURE_SIDS:
            frame = FUTURE_CASES[sid]
            yield sid, frame

    def test_resample(self):
        calendar = self.trading_calendar
        session_bar_reader = MinuteResampleSessionBarReader(
            calendar,
            self.bcolz_future_minute_bar_reader
        )
        for sid in self.ASSET_FINDER_FUTURE_SIDS:
            case_frame = FUTURE_CASES[sid]
            first = case_frame.index[0]
            last = case_frame.index[-1]
            result = session_bar_reader.load_raw_arrays(
                ['open', 'high', 'low', 'close', 'volume'],
                first, last, [sid])
            assert_almost_equal(result, EXPECTED_SESSIONS[sid],
                                err_msg="sid={0}".format(sid))

    def test_sessions(self):
        calendar = self.trading_calendar
        session_bar_reader = MinuteResampleSessionBarReader(
            calendar,
            self.bcolz_future_minute_bar_reader
        )
        sessions = session_bar_reader.sessions

        self.assertEqual(self.NUM_SESSIONS, len(sessions))
        self.assertEqual(self.START_DATE, sessions[0])
        self.assertEqual(self.END_DATE, sessions[-1])

    def test_last_available_dt(self):
        calendar = self.trading_calendar
        session_bar_reader = MinuteResampleSessionBarReader(
            calendar,
            self.bcolz_future_minute_bar_reader
        )

        self.assertEqual(self.END_DATE, session_bar_reader.last_available_dt)

    def test_spot_price(self):
        calendar = self.trading_calendar
        session_bar_reader = MinuteResampleSessionBarReader(
            calendar,
            self.bcolz_future_minute_bar_reader
        )
        for sid in self.ASSET_FINDER_FUTURE_SIDS:
            expected = EXPECTED_SESSIONS[sid]
            for dt_str, values in expected.iterrows():
                dt = pd.Timestamp(dt_str, tz='UTC')
                for col in OHLCV:
                    result = session_bar_reader.spot_price(sid, dt, col)
                    assert_almost_equal(values[col], result,
                                        err_msg="sid={0} col={1} dt={2}".
                                        format(sid, col, dt))
