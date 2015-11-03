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
from unittest import TestCase
from datetime import timedelta
import numpy as np
import pandas as pd
from testfixtures import TempDirectory
from zipline.data.us_equity_pricing import SQLiteAdjustmentWriter, \
    SQLiteAdjustmentReader
from zipline.errors import (
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
    InvalidBenchmarkAsset)

from zipline.finance.trading import TradingEnvironment
from zipline.sources.benchmark_source import BenchmarkSource
from zipline.utils import factory
from zipline.utils.test_utils import create_data_portal, write_minute_data
from .test_perf_tracking import MockDailyBarSpotReader


class TestBenchmark(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.tempdir = TempDirectory()

        cls.sim_params = factory.create_simulation_parameters()

        cls.env.write_data(equities_data={
            1: {
                "start_date": cls.sim_params.trading_days[0],
                "end_date": cls.sim_params.trading_days[-1] + timedelta(days=1)
            },
            2: {
                "start_date": cls.sim_params.trading_days[0],
                "end_date": cls.sim_params.trading_days[-1] + timedelta(days=1)
            },
            3: {
                "start_date": cls.sim_params.trading_days[100],
                "end_date": cls.sim_params.trading_days[-100]
            },
            4: {
                "start_date": cls.sim_params.trading_days[0],
                "end_date": cls.sim_params.trading_days[-1] + timedelta(days=1)
            }

        })

        dbpath = os.path.join(cls.tempdir.path, "adjustments.db")

        writer = SQLiteAdjustmentWriter(dbpath, cls.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([], dtype=np.uint32),
            'amount': np.array([], dtype=np.float64),
            'declared_date': np.array([], dtype='datetime64[ns]'),
            'ex_date': np.array([], dtype='datetime64[ns]'),
            'pay_date': np.array([], dtype='datetime64[ns]'),
            'record_date': np.array([], dtype='datetime64[ns]'),
        })
        declared_date = cls.sim_params.trading_days[45]
        ex_date = cls.sim_params.trading_days[50]
        record_date = pay_date = cls.sim_params.trading_days[55]

        stock_dividends = pd.DataFrame({
            'sid': np.array([4], dtype=np.uint32),
            'payment_sid': np.array([5], dtype=np.uint32),
            'ratio': np.array([2], dtype=np.float64),
            'declared_date': np.array([declared_date], dtype='datetime64[ns]'),
            'ex_date': np.array([ex_date], dtype='datetime64[ns]'),
            'record_date': np.array([record_date], dtype='datetime64[ns]'),
            'pay_date': np.array([pay_date], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends,
                     stock_dividends=stock_dividends)

        cls.data_portal = create_data_portal(
            cls.env,
            cls.tempdir,
            cls.sim_params,
            [1, 2, 3, 4],
            adjustment_reader=SQLiteAdjustmentReader(dbpath)
        )

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()

    def test_normal(self):
        days_to_use = self.sim_params.trading_days[1:]

        source = BenchmarkSource(
            1, self.env, days_to_use, self.data_portal
        )

        # should be the equivalent of getting the price history, then doing
        # a pct_change on it
        manually_calculated = self.data_portal.get_history_window(
            [1], days_to_use[-1], len(days_to_use), "1d", "close_price"
        )[1].pct_change()

        # compare all the fields except the first one, for which we don't have
        # data in manually_calculated
        for idx, day in enumerate(days_to_use[1:]):
            self.assertEqual(
                source.get_value(day),
                manually_calculated[idx + 1]
            )

    def test_asset_not_trading(self):
        with self.assertRaises(BenchmarkAssetNotAvailableTooEarly) as exc:
            BenchmarkSource(
                3,
                self.env,
                self.sim_params.trading_days[1:],
                self.data_portal
            )

        self.assertEqual(
            '3 does not exist on 2006-01-04 00:00:00+00:00. '
            'It started trading on 2006-05-26 00:00:00+00:00.',
            exc.exception.message
        )

        with self.assertRaises(BenchmarkAssetNotAvailableTooLate) as exc2:
            BenchmarkSource(
                3,
                self.env,
                self.sim_params.trading_days[120:],
                self.data_portal
            )

        self.assertEqual(
            '3 does not exist on 2006-06-26 00:00:00+00:00. '
            'It stopped trading on 2006-08-09 00:00:00+00:00.',
            exc2.exception.message
        )

    def test_asset_IPOed_same_day(self):
        # gotta get some minute data up in here.
        # add sid 4 for a couple of days
        minutes = self.env.minutes_for_days_in_range(
            self.sim_params.trading_days[0],
            self.sim_params.trading_days[5]
        )

        path = write_minute_data(
            self.tempdir,
            minutes,
            [2]
        )

        self.data_portal.minutes_equities_path = path

        source = BenchmarkSource(
            2,
            self.env,
            self.sim_params.trading_days,
            self.data_portal
        )

        days_to_use = self.sim_params.trading_days

        # first value should be 0.10, coming from daily data
        self.assertAlmostEquals(0.10, source.get_value(days_to_use[0]))

        manually_calculated = self.data_portal.get_history_window(
            [2], days_to_use[-1], len(days_to_use), "1d", "close_price"
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
            source = BenchmarkSource(
                4, self.env, self.sim_params.trading_days, self.data_portal
            )

        self.assertEqual("4 cannot be used as the benchmark because it has a "
                         "stock dividend on 2006-03-16 00:00:00.  Choose "
                         "another asset to use as the benchmark.",
                         exc.exception.message)
