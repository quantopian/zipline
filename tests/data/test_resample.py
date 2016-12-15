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
from numpy import nan, array, full
import pandas as pd
from pandas import DataFrame
from six import iteritems

from zipline.data.bar_reader import NoDataOnDate
from zipline.data.resample import (
    minute_frame_to_session_frame,
    DailyHistoryAggregator,
    MinuteResampleSessionBarReader,
    ReindexMinuteBarReader,
    ReindexSessionBarReader,
)

from zipline.testing.fixtures import (
    WithEquityMinuteBarData,
    WithBcolzEquityMinuteBarReader,
    WithBcolzEquityDailyBarReader,
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


FUT_MINUTES = OrderedDict((
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
         ('missing_last', 'day_0_back'))),
    (2, (('missing_first', 'day_0_front'),
         ('none_missing', 'day_0_back'))),
    (3, (('missing_last', 'day_0_back'),
         ('missing_first', 'day_1_front'))),
    # Asset 4 has a start date on day 1
    (4, (('all_missing', 'day_0_back'),
         ('none_missing', 'day_1_front'))),
    # Asset 5 has a start date before day_0, but does not have data on that
    # day.
    (5, (('all_missing', 'day_0_back'),
         ('none_missing', 'day_1_front'))),
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
              set_index(FUT_MINUTES[m])
              for s, m in combos]
    FUTURE_CASES[sid] = pd.concat(frames)


EXPECTED_AGGREGATION = {
    1: DataFrame({
        'open': [101.5, 101.5, 101.5, 101.5, 101.5, 101.5],
        'high': [101.9, 103.9, 103.9, 107.9, 108.9, 108.9],
        'low': [101.1, 101.1, 101.1, 101.1, 101.1, 101.1],
        'close': [101.3, 103.3, 102.3, 107.3, 108.3, 108.3],
        'volume': [1001, 2004, 3006, 4013, 5021, 5021],
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
    # Equity 4 straddles two days and is not active the first day.
    4: DataFrame({
        'open': [nan, nan, nan, 101.5, 101.5, 101.5],
        'high': [nan, nan, nan, 101.9, 103.9, 103.9],
        'low': [nan, nan, nan, 101.1, 101.1, 101.1],
        'close': [nan, nan, nan, 101.3, 103.3, 102.3],
        'volume': [0, 0, 0, 1001, 2004, 3006],
    }, columns=OHLCV),
    # Equity 5 straddles two days and does not have data the first day.
    5: DataFrame({
        'open': [nan, nan, nan, 101.5, 101.5, 101.5],
        'high': [nan, nan, nan, 101.9, 103.9, 103.9],
        'low': [nan, nan, nan, 101.1, 101.1, 101.1],
        'close': [nan, nan, nan, 101.3, 103.3, 102.3],
        'volume': [0, 0, 0, 1001, 2004, 3006],
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
    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3, 4, 5

    @classmethod
    def make_equity_info(cls):
        frame = super(MinuteToDailyAggregationTestCase, cls).make_equity_info()
        # Make equity 4 start a day behind the data start to exercise assets
        # which not alive for the session.
        frame.loc[[4], 'start_date'] = pd.Timestamp('2016-03-16', tz='UTC')
        return frame

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
        ('open_4', 'open', 4),
        ('high_4', 'high', 4),
        ('low_4', 'low', 4),
        ('close_4', 'close', 4),
        ('volume_4', 'volume', 4),
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
        ('open_4', 'open', 4),
        ('high_4', 'high', 4),
        ('low_4', 'low', 4),
        ('close_4', 'close', 4),
        ('volume_4', 'volume', 4),
        ('open_5', 'open', 5),
        ('high_5', 'high', 5),
        ('low_5', 'low', 5),
        ('close_5', 'close', 5),
        ('volume_5', 'volume', 5),
    ])
    def test_skip_minutes_individual(self, name, field, sid):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        asset = self.asset_finder.retrieve_asset(sid)
        minutes = EQUITY_CASES[asset].index
        for i in [0, 2, 3, 5]:
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
            result = minute_frame_to_session_frame(frame, self.nyse_calendar)
            assert_almost_equal(expected.values,
                                result.values,
                                err_msg='sid={0}'.format(sid))


class TestResampleSessionBars(WithBcolzFutureMinuteBarReader,
                              ZiplineTestCase):

    TRADING_CALENDAR_STRS = ('us_futures',)
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

    ASSET_FINDER_FUTURE_SIDS = 1001, 1002, 1003

    START_DATE = pd.Timestamp('2016-03-16', tz='UTC')
    END_DATE = pd.Timestamp('2016-03-17', tz='UTC')
    NUM_SESSIONS = 2

    @classmethod
    def make_futures_info(cls):
        future_dict = {}

        for future_sid in cls.ASSET_FINDER_FUTURE_SIDS:
            future_dict[future_sid] = {
                'multiplier': 1000,
                'exchange': 'CME',
                'root_symbol': "ABC"
            }

        return pd.DataFrame.from_dict(future_dict, orient='index')

    @classmethod
    def make_future_minute_bar_data(cls):
        for sid in cls.ASSET_FINDER_FUTURE_SIDS:
            frame = FUTURE_CASES[sid]
            yield sid, frame

    def init_instance_fixtures(self):
        super(TestResampleSessionBars, self).init_instance_fixtures()
        self.session_bar_reader = MinuteResampleSessionBarReader(
            self.trading_calendar,
            self.bcolz_future_minute_bar_reader
        )

    def test_resample(self):
        calendar = self.trading_calendar
        for sid in self.ASSET_FINDER_FUTURE_SIDS:
            case_frame = FUTURE_CASES[sid]
            first = calendar.minute_to_session_label(
                case_frame.index[0])
            last = calendar.minute_to_session_label(
                case_frame.index[-1])
            result = self.session_bar_reader.load_raw_arrays(
                OHLCV, first, last, [sid])
            for i, field in enumerate(OHLCV):
                assert_almost_equal(
                    EXPECTED_SESSIONS[sid][[field]],
                    result[i],
                    err_msg="sid={0} field={1}".format(sid, field))

    def test_sessions(self):
        sessions = self.session_bar_reader.sessions

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

    def test_get_value(self):
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
                    result = session_bar_reader.get_value(sid, dt, col)
                    assert_almost_equal(result,
                                        values[col],
                                        err_msg="sid={0} col={1} dt={2}".
                                        format(sid, col, dt))

    def test_first_trading_day(self):
        self.assertEqual(self.START_DATE,
                         self.session_bar_reader.first_trading_day)

    def test_get_last_traded_dt(self):
        future = self.asset_finder.retrieve_asset(
            self.ASSET_FINDER_FUTURE_SIDS[0]
        )

        self.assertEqual(
            self.trading_calendar.previous_session_label(self.END_DATE),
            self.session_bar_reader.get_last_traded_dt(future, self.END_DATE)
        )


class TestReindexMinuteBars(WithBcolzEquityMinuteBarReader,
                            ZiplineTestCase):

    TRADING_CALENDAR_STRS = ('us_futures', 'NYSE')
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    START_DATE = pd.Timestamp('2015-12-01', tz='UTC')
    END_DATE = pd.Timestamp('2015-12-31', tz='UTC')

    def test_load_raw_arrays(self):
        reindex_reader = ReindexMinuteBarReader(
            self.trading_calendar,
            self.bcolz_equity_minute_bar_reader,
            self.START_DATE,
            self.END_DATE,
        )
        m_open, m_close = self.trading_calendar.open_and_close_for_session(
            self.START_DATE)
        outer_minutes = self.trading_calendar.minutes_in_range(m_open, m_close)
        result = reindex_reader.load_raw_arrays(
            OHLCV, m_open, m_close, [1, 2])

        opens = DataFrame(data=result[0], index=outer_minutes,
                          columns=[1, 2])
        opens_with_price = opens.dropna()

        self.assertEqual(
            1440,
            len(opens),
            "The result should have 1440 bars, the number of minutes in a "
            "trading session on the target calendar."
        )

        self.assertEqual(
            390,
            len(opens_with_price),
            "The result, after dropping nans, should have 390 bars, the "
            " number of bars in a trading session in the reader's calendar."
        )

        slicer = outer_minutes.slice_indexer(
            end=pd.Timestamp('2015-12-01 14:30', tz='UTC'))

        assert_almost_equal(
            opens[1][slicer],
            full(slicer.stop, nan),
            err_msg="All values before the NYSE market open should be nan.")

        slicer = outer_minutes.slice_indexer(
            start=pd.Timestamp('2015-12-01 21:01', tz='UTC'))

        assert_almost_equal(
            opens[1][slicer],
            full(slicer.stop - slicer.start, nan),
            err_msg="All values after the NYSE market close should be nan.")

        first_minute_loc = outer_minutes.get_loc(pd.Timestamp(
            '2015-12-01 14:31', tz='UTC'))

        # Spot check a value.
        # The value is the autogenerated value from test fixtures.
        assert_almost_equal(
            10.0,
            opens[1][first_minute_loc],
            err_msg="The value for Equity 1, should be 10.0, at NYSE open.")


class TestReindexSessionBars(WithBcolzEquityDailyBarReader,
                             ZiplineTestCase):

    TRADING_CALENDAR_STRS = ('us_futures', 'NYSE')
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    # Dates are chosen to span Thanksgiving, which is not a Holiday on
    # us_futures.
    START_DATE = pd.Timestamp('2015-11-02', tz='UTC')
    END_DATE = pd.Timestamp('2015-11-30', tz='UTC')
    #     November 2015
    # Su Mo Tu We Th Fr Sa
    #  1  2  3  4  5  6  7
    #  8  9 10 11 12 13 14
    # 15 16 17 18 19 20 21
    # 22 23 24 25 26 27 28
    # 29 30

    def init_instance_fixtures(self):
        super(TestReindexSessionBars, self).init_instance_fixtures()

        self.reader = ReindexSessionBarReader(
            self.trading_calendar,
            self.bcolz_equity_daily_bar_reader,
            self.START_DATE,
            self.END_DATE,
        )

    def test_load_raw_arrays(self):
        outer_sessions = self.trading_calendar.sessions_in_range(
            self.START_DATE, self.END_DATE)

        result = self.reader.load_raw_arrays(
            OHLCV, self.START_DATE, self.END_DATE, [1, 2])

        opens = DataFrame(data=result[0], index=outer_sessions,
                          columns=[1, 2])
        opens_with_price = opens.dropna()

        self.assertEqual(
            21,
            len(opens),
            "The reindexed result should have 21 days, which is the number of "
            "business days in 2015-11")
        self.assertEqual(
            20,
            len(opens_with_price),
            "The reindexed result after dropping nans should have 20 days, "
            "because Thanksgiving is a NYSE holiday.")

        tday = pd.Timestamp('2015-11-26', tz='UTC')

        # Thanksgiving, 2015-11-26.
        # Is a holiday in NYSE, but not in us_futures.
        tday_loc = outer_sessions.get_loc(tday)

        assert_almost_equal(
            nan,
            opens[1][tday_loc],
            err_msg="2015-11-26 should be `nan`, since Thanksgiving is a "
            "holiday in the reader's calendar.")

        # Thanksgiving, 2015-11-26.
        # Is a holiday in NYSE, but not in us_futures.
        tday_loc = outer_sessions.get_loc(pd.Timestamp('2015-11-26', tz='UTC'))

        assert_almost_equal(
            nan,
            opens[1][tday_loc],
            err_msg="2015-11-26 should be `nan`, since Thanksgiving is a "
            "holiday in the reader's calendar.")

    def test_load_raw_arrays_holiday_start(self):
        tday = pd.Timestamp('2015-11-26', tz='UTC')
        outer_sessions = self.trading_calendar.sessions_in_range(
            tday, self.END_DATE)

        result = self.reader.load_raw_arrays(
            OHLCV, tday, self.END_DATE, [1, 2])

        opens = DataFrame(data=result[0], index=outer_sessions,
                          columns=[1, 2])
        opens_with_price = opens.dropna()

        self.assertEqual(
            3,
            len(opens),
            "The reindexed result should have 3 days, which is the number of "
            "business days in from Thanksgiving to end of 2015-11.")
        self.assertEqual(
            2,
            len(opens_with_price),
            "The reindexed result after dropping nans should have 2 days, "
            "because Thanksgiving is a NYSE holiday.")

    def test_load_raw_arrays_holiday_end(self):
        tday = pd.Timestamp('2015-11-26', tz='UTC')
        outer_sessions = self.trading_calendar.sessions_in_range(
            self.START_DATE, tday)

        result = self.reader.load_raw_arrays(
            OHLCV, self.START_DATE, tday, [1, 2])

        opens = DataFrame(data=result[0], index=outer_sessions,
                          columns=[1, 2])
        opens_with_price = opens.dropna()

        self.assertEqual(
            19,
            len(opens),
            "The reindexed result should have 19 days, which is the number of "
            "business days in from start of 2015-11 up to Thanksgiving.")
        self.assertEqual(
            18,
            len(opens_with_price),
            "The reindexed result after dropping nans should have 18 days, "
            "because Thanksgiving is a NYSE holiday.")

    def test_get_value(self):
        assert_almost_equal(self.reader.get_value(1, self.START_DATE, 'open'),
                            10.0,
                            err_msg="The open of the fixture data on the "
                            "first session should be 10.")
        tday = pd.Timestamp('2015-11-26', tz='UTC')

        with self.assertRaises(NoDataOnDate):
            self.reader.get_value(1, tday, 'close')

        with self.assertRaises(NoDataOnDate):
            self.reader.get_value(1, tday, 'volume')

    def test_last_availabe_dt(self):
        self.assertEqual(self.reader.last_available_dt, self.END_DATE)

    def test_get_last_traded_dt(self):
        asset = self.asset_finder.retrieve_asset(1)
        self.assertEqual(self.reader.get_last_traded_dt(asset,
                                                        self.END_DATE),
                         self.END_DATE)

    def test_sessions(self):
        sessions = self.reader.sessions
        self.assertEqual(21, len(sessions),
                         "There should be 21 sessions in 2015-11.")
        self.assertEqual(pd.Timestamp('2015-11-02', tz='UTC'),
                         sessions[0])
        self.assertEqual(pd.Timestamp('2015-11-30', tz='UTC'),
                         sessions[-1])

    def test_first_trading_day(self):
        self.assertEqual(self.reader.first_trading_day, self.START_DATE)

    def test_trading_calendar(self):
        self.assertEqual('us_futures',
                         self.reader.trading_calendar.name,
                         "The calendar for the reindex reader should be the "
                         "specified futures calendar.")
