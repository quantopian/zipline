#
# Copyright 2015 Quantopian, Inc.
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
import numpy as np
import pandas as pd

from zipline.data.data_portal import DataPortal
from zipline.errors import (
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
    InvalidBenchmarkAsset)

from zipline.sources.benchmark_source import BenchmarkSource
from zipline.testing import (
    MockDailyBarReader,
    create_minute_bar_data,
    tmp_bcolz_equity_minute_bar_reader,
)
from zipline.testing.fixtures import (
    WithDataPortal,
    WithSimParams,
    WithTradingCalendars,
    ZiplineTestCase,
)


class TestBenchmark(WithDataPortal, WithSimParams, WithTradingCalendars,
                    ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-12-29', tz='utc')

    @classmethod
    def make_equity_info(cls):
        return pd.DataFrame.from_dict(
            {
                1: {
                    'symbol': 'A',
                    'start_date': cls.START_DATE,
                    'end_date': cls.END_DATE + pd.Timedelta(days=1),
                    "exchange": "TEST",
                },
                2: {
                    'symbol': 'B',
                    'start_date': cls.START_DATE,
                    'end_date': cls.END_DATE + pd.Timedelta(days=1),
                    "exchange": "TEST",
                },
                3: {
                    'symbol': 'C',
                    'start_date': pd.Timestamp('2006-05-26', tz='utc'),
                    'end_date': pd.Timestamp('2006-08-09', tz='utc'),
                    "exchange": "TEST",
                },
                4: {
                    'symbol': 'D',
                    'start_date': cls.START_DATE,
                    'end_date': cls.END_DATE + pd.Timedelta(days=1),
                    "exchange": "TEST",
                },
            },
            orient='index',
        )

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return MockDailyBarReader()

    @classmethod
    def make_stock_dividends_data(cls):
        declared_date = cls.sim_params.sessions[45]
        ex_date = cls.sim_params.sessions[50]
        record_date = pay_date = cls.sim_params.sessions[55]
        return pd.DataFrame({
            'sid': np.array([4], dtype=np.uint32),
            'payment_sid': np.array([5], dtype=np.uint32),
            'ratio': np.array([2], dtype=np.float64),
            'declared_date': np.array([declared_date], dtype='datetime64[ns]'),
            'ex_date': np.array([ex_date], dtype='datetime64[ns]'),
            'record_date': np.array([record_date], dtype='datetime64[ns]'),
            'pay_date': np.array([pay_date], dtype='datetime64[ns]'),
        })

    def test_normal(self):
        days_to_use = self.sim_params.sessions[1:]

        source = BenchmarkSource(
            1, self.env, self.trading_calendar, days_to_use, self.data_portal
        )

        # should be the equivalent of getting the price history, then doing
        # a pct_change on it
        manually_calculated = self.data_portal.get_history_window(
            [1], days_to_use[-1], len(days_to_use), "1d", "close"
        )[1].pct_change()

        # compare all the fields except the first one, for which we don't have
        # data in manually_calculated
        for idx, day in enumerate(days_to_use[1:]):
            self.assertEqual(
                source.get_value(day),
                manually_calculated[idx + 1]
            )

    def test_asset_not_trading(self):
        benchmark = self.env.asset_finder.retrieve_asset(3)
        benchmark_start = benchmark.start_date
        benchmark_end = benchmark.end_date

        with self.assertRaises(BenchmarkAssetNotAvailableTooEarly) as exc:
            BenchmarkSource(
                3,
                self.env,
                self.trading_calendar,
                self.sim_params.sessions[1:],
                self.data_portal
            )

        self.assertEqual(
            '3 does not exist on %s. It started trading on %s.' %
            (self.sim_params.sessions[1], benchmark_start),
            exc.exception.message
        )

        with self.assertRaises(BenchmarkAssetNotAvailableTooLate) as exc2:
            BenchmarkSource(
                3,
                self.env,
                self.trading_calendar,
                self.sim_params.sessions[120:],
                self.data_portal
            )

        self.assertEqual(
            '3 does not exist on %s. It stopped trading on %s.' %
            (self.sim_params.sessions[-1], benchmark_end),
            exc2.exception.message
        )

    def test_asset_IPOed_same_day(self):
        # gotta get some minute data up in here.
        # add sid 4 for a couple of days
        minutes = self.trading_calendar.minutes_for_sessions_in_range(
            self.sim_params.sessions[0],
            self.sim_params.sessions[5]
        )

        tmp_reader = tmp_bcolz_equity_minute_bar_reader(
            self.trading_calendar,
            self.trading_calendar.all_sessions,
            create_minute_bar_data(minutes, [2]),
        )
        with tmp_reader as reader:
            data_portal = DataPortal(
                self.env.asset_finder, self.trading_calendar,
                first_trading_day=reader.first_trading_day,
                equity_minute_reader=reader,
                equity_daily_reader=self.bcolz_equity_daily_bar_reader,
                adjustment_reader=self.adjustment_reader,
            )

            source = BenchmarkSource(
                2,
                self.env,
                self.trading_calendar,
                self.sim_params.sessions,
                data_portal
            )

            days_to_use = self.sim_params.sessions

            # first value should be 0.0, coming from daily data
            self.assertAlmostEquals(0.0, source.get_value(days_to_use[0]))

            manually_calculated = data_portal.get_history_window(
                [2], days_to_use[-1],
                len(days_to_use),
                "1d",
                "close",
            )[2].pct_change()

            for idx, day in enumerate(days_to_use[1:]):
                self.assertEqual(
                    source.get_value(day),
                    manually_calculated[idx + 1]
                )

    def test_no_stock_dividends_allowed(self):
        # try to use sid(4) as benchmark, should blow up due to the presence
        # of a stock dividend

        with self.assertRaises(InvalidBenchmarkAsset) as exc:
            BenchmarkSource(
                4, self.env, self.trading_calendar,
                self.sim_params.sessions, self.data_portal
            )

        self.assertEqual("4 cannot be used as the benchmark because it has a "
                         "stock dividend on 2006-03-16 00:00:00.  Choose "
                         "another asset to use as the benchmark.",
                         exc.exception.message)
