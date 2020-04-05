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
import os
import mock
import warnings

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from textwrap import dedent

from zipline.data.data_portal import DataPortal
from zipline.errors import (
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
    InvalidBenchmarkAsset)

from zipline.sources.benchmark_source import BenchmarkSource
from zipline.utils.run_algo import _run

from zipline.testing import (
    MockDailyBarReader,
    create_minute_bar_data,
    tmp_bcolz_equity_minute_bar_reader
)
from zipline.testing.fixtures import (
    WithDataPortal,
    WithSimParams,
    WithTradingCalendars,
    ZiplineTestCase,
    WithLogger,
    WithTmpDir
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


class TestBenchmarkParameters(
    WithLogger,
    WithDataPortal,
    WithTradingCalendars,
    WithTmpDir,
    ZiplineTestCase,
):
    START_DATE = pd.Timestamp('2020-01-02', tz='utc')
    END_DATE = pd.Timestamp('2020-01-09', tz='utc')
    PARS = {
        'initialize': None,
        'handle_data': None,
        'before_trading_start': None,
        'analyze': None,
        'algofile': None,
        'defines': (),
        'data_frequency': 'daily',
        'capital_base': 10000000,
        'bundle': 'csvdir',
        'bundle_timestamp': False,
        'start': START_DATE,
        'end': END_DATE,
        'output': os.devnull,
        'print_algo': False,
        'metrics_set': 'default',
        'local_namespace': False,
        'environ': os.environ,
        'blotter': 'default',
    }

    SCRIPT = dedent("""
                    from zipline.api import order, symbol
                    from zipline.finance import commission, slippage

                    def initialize(context):
                        context.stocks = symbol('a')
                        context.has_ordered = False
                        context.set_commission(commission.NoCommission())
                        context.set_slippage(slippage.NoSlippage())



                    def handle_data(context, data):
                        if not context.has_ordered:
                            order(context.stocks, 1000)
                            context.has_ordered = True
                    """)

    SCRIPT_SET_BENCHMARK = dedent("""
                        from zipline.api import order, symbol
                        from zipline.finance import commission, slippage

                        def initialize(context):
                            context.stocks = symbol('a')
                            context.set_benchmark(symbol('b'))
                            context.has_ordered = False
                            context.set_commission(commission.NoCommission())
                            context.set_slippage(slippage.NoSlippage())



                        def handle_data(context, data):
                            if not context.has_ordered:
                                order(context.stocks, 1000)
                                context.has_ordered = True
                        """)

    @classmethod
    def make_equity_info(cls):
        cls.set_expected_outcome()
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

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        yield 1, pd.DataFrame(
            {
                'open': [100, 120, 100, 160, 180, 200],
                'high': [100, 120, 100, 160, 180, 200],
                'low': [100, 120, 100, 160, 180, 200],
                'close': [100, 120, 100, 160, 180, 200],
                'volume': 100,
            },
            index=cls.equity_daily_bar_days,
        )

        yield 2, pd.DataFrame(
            {
                'open': [100, 90, 120, 140, 160, 180],
                'high': [100, 90, 120, 140, 160, 180],
                'low': [100, 90, 120, 140, 160, 180],
                'close': [100, 90, 120, 140, 160, 180],
                'volume': 100,
            },
            index=cls.equity_daily_bar_days,
        )

    @classmethod
    def set_expected_outcome(cls):
        cls.expected_daily = {
            'returns': np.array([0., 0., -0.002, 0.00601202, 0.00199203,
                                 0.00198807]),
            'pnl': np.array([0., 0., -20000., 60000., 20000., 20000.]),
            'capital_used': np.array([0., -120000., 0., 0., 0., 0.]),
            'portfolio_value': np.array([10000000., 10000000., 9980000.,
                                         10040000., 10060000., 10080000.])
        }

    def data_portal_mock(self, *args, **kwargs):
        return self.data_portal

    def asset_finder_mock(self, *args, **kwargs):
        return self.data_portal.asset_finder

    @classmethod
    def validate_perf(cls, perf):
        for stat, expected_output in cls.expected_daily.items():
            np.testing.assert_array_almost_equal(
                np.array(perf[stat]),
                expected_output,
                err_msg='daily ' + stat,
            )

    def test_run_no_parameter_and_no_bennchmark_set(self):
        """
        No benchmark parameter is provided and the benchmark is not set
        in the algorithm's initialize function
        Expected outcome : ValueError: Must specify either
        benchmark_sid or benchmark_returns.
        """
        with mock.patch("zipline.utils.run_algo.DataPortal",
                        side_effect=self.data_portal_mock), \
                self.assertRaises(ValueError), \
                make_test_handler(self) as log_catcher:
            _run(
                trading_calendar=self.trading_calendar,
                algotext=self.SCRIPT,
                benchmark_returns=None,
                **self.PARS
            )
        logs = [r.message for r in log_catcher.records]
        self.assertIn("Failed to cache the new benchmark returns", logs)

    def test_run_no_parameter_and_benchmark_set(self):
        """
        No benchmark parameter is provided and the benchmark is
        set to B in the algorithm's initialize function
        Expected outcome : warning and succesful run with B as benchmark
        """

        with mock.patch("zipline.utils.run_algo.DataPortal",
                        side_effect=self.data_portal_mock), \
                warnings.catch_warnings(record=True) as w:
            perf = _run(
                trading_calendar=self.trading_calendar,
                algotext=self.SCRIPT_SET_BENCHMARK,
                benchmark_returns=None,
                **self.PARS
            )
            self.validate_perf(perf)
            self.assertIn(
                'Please specify manually a benchmark symbol using one '
                'of the following options: '
                '\n--benchmark-file, --benchmark-symbol, --no-benchmark'
                '\nYou can still retrieve market data from IEX by setting '
                'the IEX_API_KEY environment variable.\n'
                'Please note that this feature is expected to '
                'be deprecated in the future',
                str(w[-1].message))

    def test_run_no_benchmark_par(self):
        """
        --no-benchmark parameter is set to True.
        Null metrics for alpha, beta, benchmark are expected

        """
        with mock.patch("zipline.utils.run_algo.DataPortal",
                        side_effect=self.data_portal_mock), \
                make_test_handler(self) as log_catcher:
            perf = _run(
                trading_calendar=self.trading_calendar,
                algotext=self.SCRIPT,
                benchmark_returns=None,
                no_benchmark=True,
                **self.PARS
            )
            np.testing.assert_equal(
                np.array(perf.alpha),
                np.array([None, None, None, None, None, None]),

            )
            self.validate_perf(perf)
            logs = [r.message for r in log_catcher.records]
            self.assertIn(
                "Warning: Using zero returns as a benchmark. "
                "Alpha, beta and benchmark data"
                " will not be calculated.",
                logs)

    def test_run_benchmark_symbol_par(self):
        """
        --benchmark-symbol parameter is provided

        """
        with mock.patch("zipline.utils.run_algo.DataPortal",
                        side_effect=self.data_portal_mock), \
                mock.patch("zipline.utils.run_algo.bundles.core.AssetFinder",
                           side_effect=self.asset_finder_mock):
            perf = _run(
                trading_calendar=self.trading_calendar,
                algotext=self.SCRIPT,
                benchmark_returns=None,
                benchmark_symbol='b',
                **self.PARS
            )
            self.validate_perf(perf)

    def test_run_benchmark_file_par(self):
        """
        --benchmark-file parameter provided in the below format

        """
        with mock.patch("zipline.utils.run_algo.DataPortal",
                        side_effect=self.data_portal_mock):
            csv_file_path = os.path.join(self.tmpdir.path, 'b.csv')
            with open(csv_file_path, 'w') as csv_file:
                csv_file.write("date,return\n"
                               "2020-01-03 00:00:00+00:00,-0.1\n"
                               "2020-01-06 00:00:00+00:00,0.3333333333\n"
                               "2020-01-07 00:00:00+00:00,0.1666666667\n"
                               "2020-01-08 00:00:00+00:00,0.1428571429\n"
                               "2020-01-09 00:00:00+00:00,6.375\n"
                               )

            perf = _run(
                trading_calendar=self.trading_calendar,
                algotext=self.SCRIPT,
                benchmark_returns=None,
                benchmark_file=csv_file_path,
                **self.PARS
            )
            self.validate_perf(perf)
