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
import logbook
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal

from zipline.data.data_portal import DataPortal
from zipline.errors import (
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
    InvalidBenchmarkAsset)

from zipline.sources.benchmark_source import BenchmarkSource
from zipline.utils.run_algo import BenchmarkSpec

from zipline.testing import (
    MockDailyBarReader,
    create_minute_bar_data,
    parameter_space,
    tmp_bcolz_equity_minute_bar_reader,
)
from zipline.testing.predicates import assert_equal
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithDataPortal,
    WithSimParams,
    WithTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
)
from zipline.testing.core import make_test_handler


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
        return MockDailyBarReader(
            dates=cls.trading_calendar.sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )

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
            self.asset_finder.retrieve_asset(1),
            self.trading_calendar,
            days_to_use,
            self.data_portal
        )

        # should be the equivalent of getting the price history, then doing
        # a pct_change on it
        manually_calculated = self.data_portal.get_history_window(
            [1],
            days_to_use[-1],
            len(days_to_use),
            "1d",
            "close",
            "daily",
        )[1].pct_change()

        # compare all the fields except the first one, for which we don't have
        # data in manually_calculated
        for idx, day in enumerate(days_to_use[1:]):
            self.assertEqual(
                source.get_value(day),
                manually_calculated[idx + 1]
            )

        # compare a slice of the data
        assert_series_equal(
            source.get_range(days_to_use[1], days_to_use[10]),
            manually_calculated[1:11]
        )

    def test_asset_not_trading(self):
        benchmark = self.asset_finder.retrieve_asset(3)
        benchmark_start = benchmark.start_date
        benchmark_end = benchmark.end_date

        with self.assertRaises(BenchmarkAssetNotAvailableTooEarly) as exc:
            BenchmarkSource(
                benchmark,
                self.trading_calendar,
                self.sim_params.sessions[1:],
                self.data_portal
            )

        self.assertEqual(
            'Equity(3 [C]) does not exist on %s. It started trading on %s.' %
            (self.sim_params.sessions[1], benchmark_start),
            exc.exception.message
        )

        with self.assertRaises(BenchmarkAssetNotAvailableTooLate) as exc2:
            BenchmarkSource(
                benchmark,
                self.trading_calendar,
                self.sim_params.sessions[120:],
                self.data_portal
            )

        self.assertEqual(
            'Equity(3 [C]) does not exist on %s. It stopped trading on %s.' %
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
                self.asset_finder, self.trading_calendar,
                first_trading_day=reader.first_trading_day,
                equity_minute_reader=reader,
                equity_daily_reader=self.bcolz_equity_daily_bar_reader,
                adjustment_reader=self.adjustment_reader,
            )

            source = BenchmarkSource(
                self.asset_finder.retrieve_asset(2),
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
                "daily",
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
                self.asset_finder.retrieve_asset(4),
                self.trading_calendar,
                self.sim_params.sessions,
                self.data_portal
            )

        self.assertEqual("Equity(4 [D]) cannot be used as the benchmark "
                         "because it has a stock dividend on 2006-03-16 "
                         "00:00:00.  Choose another asset to use as the "
                         "benchmark.",
                         exc.exception.message)


class BenchmarkSpecTestCase(WithTmpDir,
                            WithAssetFinder,
                            ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(BenchmarkSpecTestCase, cls).init_class_fixtures()

        zero_returns_index = pd.date_range(
            cls.START_DATE,
            cls.END_DATE,
            freq='D',
            tz='utc',
        )
        cls.zero_returns = pd.Series(index=zero_returns_index, data=0.0)

    def init_instance_fixtures(self):
        super(BenchmarkSpecTestCase, self).init_instance_fixtures()
        self.log_handler = self.enter_instance_context(make_test_handler(self))

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
                }
            },
            orient='index',
        )

    def logs_at_level(self, level):
        return [
            r.message for r in self.log_handler.records if r.level == level
        ]

    def resolve_spec(self, spec):
        return spec.resolve(self.asset_finder, self.START_DATE, self.END_DATE)

    def test_no_benchmark(self):
        """Test running with no benchmark provided.

        We should have no benchmark sid and have a returns series of all zeros.
        """
        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=None,
            benchmark_symbol=None,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        self.assertIs(sid, None)
        self.assertIs(returns, None)

        warnings = self.logs_at_level(logbook.WARNING)
        expected = [
            'No benchmark configured. Assuming algorithm calls set_benchmark.',
            'Pass --benchmark-sid, --benchmark-symbol, or --benchmark-file to set a source of benchmark returns.',  # noqa
            "Pass --no-benchmark to use a dummy benchmark of zero returns.",
        ]
        assert_equal(warnings, expected)

    def test_no_benchmark_explicitly_disabled(self):
        """Test running with no benchmark provided, with no_benchmark flag.
        """
        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=True,
            benchmark_sid=None,
            benchmark_symbol=None,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        self.assertIs(sid, None)
        assert_series_equal(returns, self.zero_returns)

        warnings = self.logs_at_level(logbook.WARNING)
        expected = []
        assert_equal(warnings, expected)

    @parameter_space(case=[('A', 1), ('B', 2)])
    def test_benchmark_symbol(self, case):
        """Test running with no benchmark provided, with no_benchmark flag.
        """
        symbol, expected_sid = case

        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=None,
            benchmark_symbol=symbol,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        assert_equal(sid, expected_sid)
        self.assertIs(returns, None)

        warnings = self.logs_at_level(logbook.WARNING)
        expected = []
        assert_equal(warnings, expected)

    @parameter_space(input_sid=[1, 2])
    def test_benchmark_sid(self, input_sid):
        """Test running with no benchmark provided, with no_benchmark flag.
        """
        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=input_sid,
            benchmark_symbol=None,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        assert_equal(sid, input_sid)
        self.assertIs(returns, None)

        warnings = self.logs_at_level(logbook.WARNING)
        expected = []
        assert_equal(warnings, expected)

    def test_benchmark_file(self):
        """Test running with a benchmark file.
        """
        csv_file_path = self.tmpdir.getpath('b.csv')
        with open(csv_file_path, 'w') as csv_file:
            csv_file.write("date,return\n"
                           "2020-01-03 00:00:00+00:00,-0.1\n"
                           "2020-01-06 00:00:00+00:00,0.333\n"
                           "2020-01-07 00:00:00+00:00,0.167\n"
                           "2020-01-08 00:00:00+00:00,0.143\n"
                           "2020-01-09 00:00:00+00:00,6.375\n")

        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=None,
            benchmark_symbol=None,
            benchmark_file=csv_file_path,
        )

        sid, returns = self.resolve_spec(spec)

        self.assertIs(sid, None)

        expected_dates = pd.to_datetime(
            ['2020-01-03', '2020-01-06', '2020-01-07',
             '2020-01-08', '2020-01-09'],
            utc=True,
        )
        expected_values = [-0.1, 0.333, 0.167, 0.143, 6.375]
        expected_returns = pd.Series(index=expected_dates,
                                     data=expected_values)

        assert_series_equal(returns, expected_returns, check_names=False)

        warnings = self.logs_at_level(logbook.WARNING)
        expected = []
        assert_equal(warnings, expected)
