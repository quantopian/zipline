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
from numpy import array, nan
from numpy.testing import assert_almost_equal
from pandas import DataFrame, Timestamp

from zipline.assets import Equity, Future

from zipline.data.dispatch_bar_reader import (
    AssetDispatchMinuteBarReader,
    AssetDispatchSessionBarReader,
)
from zipline.data.resample import (
    MinuteResampleSessionBarReader,
    ReindexMinuteBarReader,
    ReindexSessionBarReader,
)
from zipline.testing.fixtures import (
    WithBcolzEquityMinuteBarReader,
    WithBcolzEquityDailyBarReader,
    WithBcolzFutureMinuteBarReader,
    WithTradingSessions,
    ZiplineTestCase,
)

OHLC = ["open", "high", "low", "close"]


class AssetDispatchSessionBarTestCase(
    WithBcolzEquityDailyBarReader,
    WithBcolzFutureMinuteBarReader,
    WithTradingSessions,
    ZiplineTestCase,
):
    TRADING_CALENDAR_STRS = ("us_futures", "NYSE")
    TRADING_CALENDAR_PRIMARY_CAL = "us_futures"

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    START_DATE = Timestamp("2016-08-22")
    END_DATE = Timestamp("2016-08-24")

    @classmethod
    def make_future_minute_bar_data(cls):
        m_opens = [
            cls.trading_calendar.session_first_minute(session)
            for session in cls.trading_sessions["us_futures"]
        ]
        yield 10001, DataFrame(
            {
                "open": [10000.5, 10001.5, nan],
                "high": [10000.9, 10001.9, nan],
                "low": [10000.1, 10001.1, nan],
                "close": [10000.3, 10001.3, nan],
                "volume": [1000, 1001, 0],
            },
            index=m_opens,
        )
        yield 10002, DataFrame(
            {
                "open": [20000.5, nan, 20002.5],
                "high": [20000.9, nan, 20002.9],
                "low": [20000.1, nan, 20002.1],
                "close": [20000.3, nan, 20002.3],
                "volume": [2000, 0, 2002],
            },
            index=m_opens,
        )
        yield 10003, DataFrame(
            {
                "open": [nan, 30001.5, 30002.5],
                "high": [nan, 30001.9, 30002.9],
                "low": [nan, 30001.1, 30002.1],
                "close": [nan, 30001.3, 30002.3],
                "volume": [0, 3001, 3002],
            },
            index=m_opens,
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        sessions = cls.trading_sessions["NYSE"]
        yield 1, DataFrame(
            {
                "open": [100.5, 101.5, nan],
                "high": [100.9, 101.9, nan],
                "low": [100.1, 101.1, nan],
                "close": [100.3, 101.3, nan],
                "volume": [1000, 1001, 0],
            },
            index=sessions,
        )
        yield 2, DataFrame(
            {
                "open": [200.5, nan, 202.5],
                "high": [200.9, nan, 202.9],
                "low": [200.1, nan, 202.1],
                "close": [200.3, nan, 202.3],
                "volume": [2000, 0, 2002],
            },
            index=sessions,
        )
        yield 3, DataFrame(
            {
                "open": [301.5, 302.5, nan],
                "high": [301.9, 302.9, nan],
                "low": [301.1, 302.1, nan],
                "close": [301.3, 302.3, nan],
                "volume": [3001, 3002, 0],
            },
            index=sessions,
        )

    @classmethod
    def make_futures_info(cls):
        return DataFrame(
            {
                "sid": [10001, 10002, 10003],
                "root_symbol": ["FOO", "BAR", "BAZ"],
                "symbol": ["FOOA", "BARA", "BAZA"],
                "start_date": [cls.START_DATE] * 3,
                "end_date": [cls.END_DATE] * 3,
                # TODO: Make separate from 'end_date'
                "notice_date": [cls.END_DATE] * 3,
                "expiration_date": [cls.END_DATE] * 3,
                "multiplier": [500] * 3,
                "exchange": ["CMES"] * 3,
            }
        )

    @classmethod
    def init_class_fixtures(cls):
        super(AssetDispatchSessionBarTestCase, cls).init_class_fixtures()

        readers = {
            Equity: ReindexSessionBarReader(
                cls.trading_calendar,
                cls.bcolz_equity_daily_bar_reader,
                cls.START_DATE,
                cls.END_DATE,
            ),
            Future: MinuteResampleSessionBarReader(
                cls.trading_calendar,
                cls.bcolz_future_minute_bar_reader,
            ),
        }
        cls.dispatch_reader = AssetDispatchSessionBarReader(
            cls.trading_calendar, cls.asset_finder, readers
        )

    def test_load_raw_arrays(self):
        sessions = self.trading_calendar.sessions_in_range(
            self.START_DATE, self.END_DATE
        )

        results = self.dispatch_reader.load_raw_arrays(
            ["high", "volume"], sessions[0], sessions[2], [2, 10003, 1, 10001]
        )

        expected_per_sid = (
            (
                2,
                [array([200.9, nan, 202.9]), array([2000, 0, 2002])],
                "sid=2 should have values on the first and third sessions.",
            ),
            (
                10003,
                [array([nan, 30001.9, 30002.9]), array([0, 3001, 3002])],
                "sid=10003 should have values on the second and third sessions.",
            ),
            (
                1,
                [array([100.9, 101.90, nan]), array([1000, 1001, 0])],
                "sid=1 should have values on the first and second sessions.",
            ),
            (
                10001,
                [array([10000.9, 10001.9, nan]), array([1000, 1001, 0])],
                "sid=10001 should have a values on the first and second " "sessions.",
            ),
        )

        for i, (_sid, expected, msg) in enumerate(expected_per_sid):
            for j, result in enumerate(results):
                assert_almost_equal(result[:, i], expected[j], err_msg=msg)


class AssetDispatchMinuteBarTestCase(
    WithBcolzEquityMinuteBarReader, WithBcolzFutureMinuteBarReader, ZiplineTestCase
):
    TRADING_CALENDAR_STRS = ("us_futures", "NYSE")
    TRADING_CALENDAR_PRIMARY_CAL = "us_futures"

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    START_DATE = Timestamp("2016-08-24")
    END_DATE = Timestamp("2016-08-24")

    @classmethod
    def make_equity_minute_bar_data(cls):
        minutes = cls.trading_calendars[Equity].session_minutes(cls.START_DATE)
        yield 1, DataFrame(
            {
                "open": [100.5, 101.5],
                "high": [100.9, 101.9],
                "low": [100.1, 101.1],
                "close": [100.3, 101.3],
                "volume": [1000, 1001],
            },
            index=minutes[[0, 1]],
        )
        yield 2, DataFrame(
            {
                "open": [200.5, 202.5],
                "high": [200.9, 202.9],
                "low": [200.1, 202.1],
                "close": [200.3, 202.3],
                "volume": [2000, 2002],
            },
            index=minutes[[0, 2]],
        )
        yield 3, DataFrame(
            {
                "open": [301.5, 302.5],
                "high": [301.9, 302.9],
                "low": [301.1, 302.1],
                "close": [301.3, 302.3],
                "volume": [3001, 3002],
            },
            index=minutes[[1, 2]],
        )

    @classmethod
    def make_future_minute_bar_data(cls):
        e_m = cls.trading_calendars[Equity].session_minutes(cls.START_DATE)
        f_m = cls.trading_calendar.session_minutes(cls.START_DATE)
        # Equity market open occurs at loc 930 in Future minutes.
        minutes = [f_m[0], e_m[0], e_m[1]]
        yield 10001, DataFrame(
            {
                "open": [10000.5, 10930.5, 10931.5],
                "high": [10000.9, 10930.9, 10931.9],
                "low": [10000.1, 10930.1, 10931.1],
                "close": [10000.3, 10930.3, 10931.3],
                "volume": [1000, 1930, 1931],
            },
            index=minutes,
        )
        minutes = [f_m[1], e_m[1], e_m[2]]
        yield 10002, DataFrame(
            {
                "open": [20001.5, 20931.5, 20932.5],
                "high": [20001.9, 20931.9, 20932.9],
                "low": [20001.1, 20931.1, 20932.1],
                "close": [20001.3, 20931.3, 20932.3],
                "volume": [2001, 2931, 2932],
            },
            index=minutes,
        )
        minutes = [f_m[2], e_m[0], e_m[2]]
        yield 10003, DataFrame(
            {
                "open": [30002.5, 30930.5, 30932.5],
                "high": [30002.9, 30930.9, 30932.9],
                "low": [30002.1, 30930.1, 30932.1],
                "close": [30002.3, 30930.3, 30932.3],
                "volume": [3002, 3930, 3932],
            },
            index=minutes,
        )

    @classmethod
    def make_futures_info(cls):
        return DataFrame(
            {
                "sid": [10001, 10002, 10003],
                "root_symbol": ["FOO", "BAR", "BAZ"],
                "symbol": ["FOOA", "BARA", "BAZA"],
                "start_date": [cls.START_DATE] * 3,
                "end_date": [cls.END_DATE] * 3,
                # TODO: Make separate from 'end_date'
                "notice_date": [cls.END_DATE] * 3,
                "expiration_date": [cls.END_DATE] * 3,
                "multiplier": [500] * 3,
                "exchange": ["CMES"] * 3,
            }
        )

    @classmethod
    def init_class_fixtures(cls):
        super(AssetDispatchMinuteBarTestCase, cls).init_class_fixtures()

        readers = {
            Equity: ReindexMinuteBarReader(
                cls.trading_calendar,
                cls.bcolz_equity_minute_bar_reader,
                cls.START_DATE,
                cls.END_DATE,
            ),
            Future: cls.bcolz_future_minute_bar_reader,
        }
        cls.dispatch_reader = AssetDispatchMinuteBarReader(
            cls.trading_calendar, cls.asset_finder, readers
        )

    def test_load_raw_arrays_at_future_session_open(self):
        f_minutes = self.trading_calendar.session_minutes(self.START_DATE)

        results = self.dispatch_reader.load_raw_arrays(
            ["open", "close"], f_minutes[0], f_minutes[2], [2, 10003, 1, 10001]
        )

        expected_per_sid = (
            (
                2,
                [array([nan, nan, nan]), array([nan, nan, nan])],
                "Before Equity market open, sid=2 should have no values.",
            ),
            (
                10003,
                [array([nan, nan, 30002.5]), array([nan, nan, 30002.3])],
                "sid=10003 should have a value at the 22:03 occurring "
                "before the session label, which will be the third minute.",
            ),
            (
                1,
                [array([nan, nan, nan]), array([nan, nan, nan])],
                "Before Equity market open, sid=1 should have no values.",
            ),
            (
                10001,
                [array([10000.5, nan, nan]), array([10000.3, nan, nan])],
                "sid=10001 should have a value at the market open.",
            ),
        )

        for i, (sid, expected, msg) in enumerate(expected_per_sid):
            for j, result in enumerate(results):
                assert_almost_equal(result[:, i], expected[j], err_msg=msg)

        results = self.dispatch_reader.load_raw_arrays(
            ["open"], f_minutes[0], f_minutes[2], [2, 10003, 1, 10001]
        )

    def test_load_raw_arrays_at_equity_session_open(self):
        e_minutes = self.trading_calendars[Equity].session_minutes(self.START_DATE)

        results = self.dispatch_reader.load_raw_arrays(
            ["open", "high"], e_minutes[0], e_minutes[2], [10002, 1, 3, 10001]
        )

        expected_per_sid = (
            (
                10002,
                [array([nan, 20931.5, 20932.5]), array([nan, 20931.9, 20932.9])],
                "At Equity market open, sid=10002 should have values at the "
                "second and third minute.",
            ),
            (
                1,
                [array([100.5, 101.5, nan]), array([100.9, 101.9, nan])],
                "At Equity market open, sid=1 should have values at the first "
                "and second minute.",
            ),
            (
                3,
                [array([nan, 301.5, 302.5]), array([nan, 301.9, 302.9])],
                "At Equity market open, sid=3 should have a values at the second "
                "and third minute.",
            ),
            (
                10001,
                [array([10930.5, 10931.5, nan]), array([10930.9, 10931.9, nan])],
                "At Equity market open, sid=10001 should have a values at the "
                "first and second minute.",
            ),
        )

        for i, (sid, expected, msg) in enumerate(expected_per_sid):
            for j, result in enumerate(results):
                assert_almost_equal(result[:, i], expected[j], err_msg=msg)
