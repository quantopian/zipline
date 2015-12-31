from os.path import dirname, join, realpath
from textwrap import dedent
from unittest import TestCase
import bcolz
import os
from datetime import timedelta
from nose_parameterized import parameterized
from pandas.tslib import normalize_date
from testfixtures import TempDirectory
import numpy as np
from numpy import array
import pandas as pd
from pandas import (
    read_csv,
    Timestamp,
    DataFrame, DatetimeIndex)

from six import iteritems
from zipline import TradingAlgorithm

from zipline.data.data_portal import DataPortal
from zipline.data.us_equity_pricing import (
    DailyBarWriterFromCSVs,
    SQLiteAdjustmentWriter,
    SQLiteAdjustmentReader,
)
from zipline.errors import HistoryInInitialize
from zipline.utils.test_utils import (
    make_simple_asset_info,
    str_to_seconds,
    MockDailyBarReader
)
from zipline.data.future_pricing import FutureMinuteReader
from zipline.data.us_equity_pricing import BcolzDailyBarReader
from zipline.data.us_equity_minutes import (
    MinuteBarWriterFromCSVs,
    BcolzMinuteBarReader
)
from zipline.utils.tradingcalendar import trading_days
from zipline.finance.trading import (
    TradingEnvironment,
    SimulationParameters
)

TEST_MINUTE_RESOURCE_PATH = join(
    dirname(dirname(realpath(__file__))),  # zipline_repo/tests
    'tests',
    'resources',
    'history_inputs',
)

TEST_DAILY_RESOURCE_PATH = join(
    dirname(dirname(realpath(__file__))),  # zipline_repo/tests
    'tests',
    'resources',
    'pipeline_inputs',
)


class HistoryTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AAPL = 1
        cls.MSFT = 2
        cls.DELL = 3
        cls.TSLA = 4
        cls.BRKA = 5
        cls.IBM = 6
        cls.GS = 7
        cls.C = 8
        cls.DIVIDEND_SID = 9
        cls.FUTURE_ASSET = 10
        cls.FUTURE_ASSET2 = 11
        cls.FUTURE_ASSET3 = 12
        cls.FOO = 13
        cls.assets = [cls.AAPL, cls.MSFT, cls.DELL, cls.TSLA, cls.BRKA,
                      cls.IBM, cls.GS, cls.C, cls.DIVIDEND_SID, cls.FOO]

        asset_info = make_simple_asset_info(
            cls.assets,
            Timestamp('2014-03-03'),
            Timestamp('2014-08-30'),
            ['AAPL', 'MSFT', 'DELL', 'TSLA', 'BRKA', 'IBM', 'GS', 'C',
             'DIVIDEND_SID', 'FOO']
        )
        cls.env = TradingEnvironment()

        cls.env.write_data(
            equities_df=asset_info,
            futures_data={
                cls.FUTURE_ASSET: {
                    "start_date": pd.Timestamp('2015-11-23', tz='UTC'),
                    "end_date": pd.Timestamp('2014-12-01', tz='UTC'),
                    'symbol': 'TEST_FUTURE',
                    'asset_type': 'future',
                },
                cls.FUTURE_ASSET2: {
                    "start_date": pd.Timestamp('2014-03-19', tz='UTC'),
                    "end_date": pd.Timestamp('2014-03-22', tz='UTC'),
                    'symbol': 'TEST_FUTURE2',
                    'asset_type': 'future',
                },
                cls.FUTURE_ASSET3: {
                    "start_date": pd.Timestamp('2014-03-19', tz='UTC'),
                    "end_date": pd.Timestamp('2014-03-22', tz='UTC'),
                    'symbol': 'TEST_FUTURE3',
                    'asset_type': 'future',
                }
            }
        )

        cls.tempdir = TempDirectory()
        cls.tempdir.create()

        try:
            cls.create_fake_minute_data(cls.tempdir)

            cls.futures_start_dates = {
                cls.FUTURE_ASSET: pd.Timestamp("2015-11-23 20:11", tz='UTC'),
                cls.FUTURE_ASSET2: pd.Timestamp("2014-03-19 13:31", tz='UTC'),
                cls.FUTURE_ASSET3: pd.Timestamp("2014-03-19 13:31", tz='UTC')
            }

            futures_tempdir = os.path.join(cls.tempdir.path,
                                           'futures', 'minutes')
            os.makedirs(futures_tempdir)
            cls.create_fake_futures_minute_data(
                futures_tempdir,
                cls.env.asset_finder.retrieve_asset(cls.FUTURE_ASSET),
                cls.futures_start_dates[cls.FUTURE_ASSET],
                cls.futures_start_dates[cls.FUTURE_ASSET] +
                timedelta(minutes=10000)
            )

            # build data for FUTURE_ASSET2 from 2014-03-19 13:31 to
            # 2014-03-21 20:00
            cls.create_fake_futures_minute_data(
                futures_tempdir,
                cls.env.asset_finder.retrieve_asset(cls.FUTURE_ASSET2),
                cls.futures_start_dates[cls.FUTURE_ASSET2],
                cls.futures_start_dates[cls.FUTURE_ASSET2] +
                timedelta(minutes=3270)
            )

            # build data for FUTURE_ASSET3 from 2014-03-19 13:31 to
            # 2014-03-21 20:00.
            # Pause trading between 2014-03-20 14:00 and 2014-03-20 15:00
            gap_start = pd.Timestamp('2014-03-20 14:00', tz='UTC')
            gap_end = pd.Timestamp('2014-03-20 15:00', tz='UTC')
            cls.create_fake_futures_minute_data(
                futures_tempdir,
                cls.env.asset_finder.retrieve_asset(cls.FUTURE_ASSET3),
                cls.futures_start_dates[cls.FUTURE_ASSET3],
                cls.futures_start_dates[cls.FUTURE_ASSET3] +
                timedelta(minutes=3270),
                gap_start_dt=gap_start,
                gap_end_dt=gap_end,
            )

            cls.create_fake_daily_data(cls.tempdir)

            splits = DataFrame([
                {'effective_date': str_to_seconds("2002-01-03"),
                 'ratio': 0.5,
                 'sid': cls.AAPL},
                {'effective_date': str_to_seconds("2014-03-20"),
                 'ratio': 0.5,
                 'sid': cls.AAPL},
                {'effective_date': str_to_seconds("2014-03-21"),
                 'ratio': 0.5,
                 'sid': cls.AAPL},
                {'effective_date': str_to_seconds("2014-04-01"),
                 'ratio': 0.5,
                 'sid': cls.IBM},
                {'effective_date': str_to_seconds("2014-07-01"),
                 'ratio': 0.5,
                 'sid': cls.IBM},
                {'effective_date': str_to_seconds("2014-07-07"),
                 'ratio': 0.5,
                 'sid': cls.IBM},
                {'effective_date': str_to_seconds("2002-03-21"),
                 'ratio': 0.5,
                 'sid': cls.FOO},
            ],
                columns=['effective_date', 'ratio', 'sid'],
            )

            mergers = DataFrame([
                {'effective_date': str_to_seconds("2014-07-16"),
                 'ratio': 0.5,
                 'sid': cls.C}
            ],
                columns=['effective_date', 'ratio', 'sid'])

            dividends = DataFrame([
                {'ex_date':
                 Timestamp("2014-03-18", tz='UTC').to_datetime64(),
                 'record_date':
                 Timestamp("2014-03-19", tz='UTC').to_datetime64(),
                 'declared_date':
                 Timestamp("2014-03-18", tz='UTC').to_datetime64(),
                 'pay_date':
                 Timestamp("2014-03-20", tz='UTC').to_datetime64(),
                 'amount': 2.0,
                 'sid': cls.DIVIDEND_SID},
                {'ex_date':
                 Timestamp("2014-03-20", tz='UTC').to_datetime64(),
                 'record_date':
                 Timestamp("2014-03-21", tz='UTC').to_datetime64(),
                 'declared_date':
                 Timestamp("2014-03-18", tz='UTC').to_datetime64(),
                 'pay_date':
                 Timestamp("2014-03-23", tz='UTC').to_datetime64(),
                 'amount': 4.0,
                 'sid': cls.DIVIDEND_SID}],
                columns=['ex_date',
                         'record_date',
                         'declared_date',
                         'pay_date',
                         'amount',
                         'sid'])

            cls.create_fake_adjustments(cls.tempdir,
                                        "adjustments.sqlite",
                                        splits=splits,
                                        mergers=mergers,
                                        dividends=dividends)

            cls.data_portal = cls.get_portal(
                daily_equities_filename="test_daily_data.bcolz",
                adjustments_filename="adjustments.sqlite"
            )
        except:
            cls.tempdir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def create_fake_futures_minute_data(cls, tempdir, asset, start_dt, end_dt,
                                        gap_start_dt=None, gap_end_dt=None):
        num_minutes = int((end_dt - start_dt).total_seconds() / 60)

        # need to prepend one 0 per minute between normalize_date(start_dt)
        # and start_dt
        zeroes_buffer = \
            [0] * int((start_dt -
                       normalize_date(start_dt)).total_seconds() / 60)

        future_df = pd.DataFrame({
            "open": np.array(zeroes_buffer +
                             list(range(0, num_minutes))) * 1000,
            "high": np.array(zeroes_buffer +
                             list(range(10000, 10000 + num_minutes))) * 1000,
            "low": np.array(zeroes_buffer +
                            list(range(20000, 20000 + num_minutes))) * 1000,
            "close": np.array(zeroes_buffer +
                              list(range(30000, 30000 + num_minutes))) * 1000,
            "volume": np.array(zeroes_buffer +
                               list(range(40000, 40000 + num_minutes)))
        })

        if gap_start_dt and gap_end_dt:
            minutes = pd.date_range(normalize_date(start_dt), end_dt, freq='T')
            gap_start_ix = minutes.get_loc(gap_start_dt)
            gap_end_ix = minutes.get_loc(gap_end_dt)
            future_df.iloc[gap_start_ix:gap_end_ix, :] = 0

        path = join(tempdir, "{0}.bcolz".format(asset.sid))
        ctable = bcolz.ctable.fromdataframe(future_df, rootdir=path)

        ctable.attrs["start_dt"] = start_dt.value / 1e9
        ctable.attrs["last_dt"] = end_dt.value / 1e9

    @classmethod
    def create_fake_minute_data(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_MINUTE_RESOURCE_PATH, 'AAPL_minute.csv.gz'),
            cls.MSFT: join(TEST_MINUTE_RESOURCE_PATH, 'MSFT_minute.csv.gz'),
            cls.DELL: join(TEST_MINUTE_RESOURCE_PATH, 'DELL_minute.csv.gz'),
            cls.TSLA: join(TEST_MINUTE_RESOURCE_PATH, "TSLA_minute.csv.gz"),
            cls.BRKA: join(TEST_MINUTE_RESOURCE_PATH, "BRKA_minute.csv.gz"),
            cls.IBM: join(TEST_MINUTE_RESOURCE_PATH, "IBM_minute.csv.gz"),
            cls.GS:
            join(TEST_MINUTE_RESOURCE_PATH, "IBM_minute.csv.gz"),  # unused
            cls.C: join(TEST_MINUTE_RESOURCE_PATH, "C_minute.csv.gz"),
            cls.DIVIDEND_SID: join(TEST_MINUTE_RESOURCE_PATH,
                                   "DIVIDEND_minute.csv.gz"),
            cls.FOO: join(TEST_MINUTE_RESOURCE_PATH,
                          "FOO_minute.csv.gz"),
        }

        equities_tempdir = os.path.join(tempdir.path, 'equity', 'minutes')
        os.makedirs(equities_tempdir)

        MinuteBarWriterFromCSVs(resources,
                                pd.Timestamp('2002-01-02', tz='UTC')).write(
                                    equities_tempdir, cls.assets)

    @classmethod
    def create_fake_daily_data(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_DAILY_RESOURCE_PATH, 'AAPL.csv'),
            cls.MSFT: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),
            cls.DELL: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),  # unused
            cls.TSLA: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),  # unused
            cls.BRKA: join(TEST_DAILY_RESOURCE_PATH, 'BRK-A.csv'),
            cls.IBM: join(TEST_MINUTE_RESOURCE_PATH, 'IBM_daily.csv.gz'),
            cls.GS: join(TEST_MINUTE_RESOURCE_PATH, 'GS_daily.csv.gz'),
            cls.C: join(TEST_MINUTE_RESOURCE_PATH, 'C_daily.csv.gz'),
            cls.DIVIDEND_SID: join(TEST_MINUTE_RESOURCE_PATH,
                                   'DIVIDEND_daily.csv.gz'),
            cls.FOO: join(TEST_MINUTE_RESOURCE_PATH, 'FOO_daily.csv.gz'),
        }
        raw_data = {
            asset: read_csv(path, parse_dates=['day']).set_index('day')
            for asset, path in iteritems(resources)
        }
        for frame in raw_data.values():
            frame['price'] = frame['close']

        writer = DailyBarWriterFromCSVs(resources)
        data_path = tempdir.getpath('test_daily_data.bcolz')
        writer.write(data_path, trading_days, cls.assets)

    @classmethod
    def create_fake_adjustments(cls, tempdir, filename,
                                splits=None, mergers=None, dividends=None):
        writer = SQLiteAdjustmentWriter(tempdir.getpath(filename),
                                        cls.env.trading_days,
                                        MockDailyBarReader())

        if dividends is None:
            dividends = DataFrame(
                {
                    # Hackery to make the dtypes correct on an empty frame.
                    'ex_date': array([], dtype='datetime64[ns]'),
                    'pay_date': array([], dtype='datetime64[ns]'),
                    'record_date': array([], dtype='datetime64[ns]'),
                    'declared_date': array([], dtype='datetime64[ns]'),
                    'amount': array([], dtype=float),
                    'sid': array([], dtype=int),
                },
                index=DatetimeIndex([], tz='UTC'),
                columns=['ex_date',
                         'pay_date',
                         'record_date',
                         'declared_date',
                         'amount',
                         'sid']
                )

        if splits is None:
            splits = DataFrame(
                {
                    # Hackery to make the dtypes correct on an empty frame.
                    'effective_date': array([], dtype=int),
                    'ratio': array([], dtype=float),
                    'sid': array([], dtype=int),
                },
                index=DatetimeIndex([], tz='UTC'))

        if mergers is None:
            mergers = DataFrame(
                {
                    # Hackery to make the dtypes correct on an empty frame.
                    'effective_date': array([], dtype=int),
                    'ratio': array([], dtype=float),
                    'sid': array([], dtype=int),
                },
                index=DatetimeIndex([], tz='UTC'))

        writer.write(splits, mergers, dividends)

    @classmethod
    def get_portal(cls,
                   daily_equities_filename="test_daily_data.bcolz",
                   adjustments_filename="adjustments.sqlite",
                   env=None):

        if env is None:
            env = cls.env

        temp_path = cls.tempdir.path

        minutes_path = os.path.join(temp_path, 'equity', 'minutes')
        futures_path = os.path.join(temp_path, 'futures', 'minutes')

        adjustment_reader = SQLiteAdjustmentReader(
            join(temp_path, adjustments_filename))

        equity_minute_reader = BcolzMinuteBarReader(minutes_path)

        equity_daily_reader = BcolzDailyBarReader(
            join(temp_path, daily_equities_filename))

        future_minute_reader = FutureMinuteReader(futures_path)

        return DataPortal(
            env,
            equity_minute_reader=equity_minute_reader,
            future_minute_reader=future_minute_reader,
            equity_daily_reader=equity_daily_reader,
            adjustment_reader=adjustment_reader
        )

    def test_history_in_initialize(self):
        algo_text = dedent(
            """\
            from zipline.api import history

            def initialize(context):
                history([24], 10, '1d', 'price')

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

    def test_minute_basic_functionality(self):
        # get a 5-bar minute history from the very end of the available data
        window = self.data_portal.get_history_window(
            [1],
            pd.Timestamp("2014-03-21 18:23:00+00:00", tz='UTC'),
            5,
            "1m",
            "open_price"
        )

        self.assertEqual(len(window), 5)
        reference = [534.469, 534.471, 534.475, 534.477, 534.477]
        for i in range(0, 4):
            self.assertEqual(window.iloc[-5 + i].loc[1], reference[i])

    def test_minute_splits(self):
        portal = self.data_portal

        window = portal.get_history_window(
            [1],
            pd.Timestamp("2014-03-21 18:30:00+00:00", tz='UTC'),
            1000,
            "1m",
            "open_price"
        )

        self.assertEqual(len(window), 1000)

        # there are two splits for AAPL (on 2014-03-20 and 2014-03-21),
        # each with ratio 0.5).

        day1_end = pd.Timestamp("2014-03-19 20:00", tz='UTC')
        day2_start = pd.Timestamp("2014-03-20 13:31", tz='UTC')
        day2_end = pd.Timestamp("2014-03-20 20:00", tz='UTC')
        day3_start = pd.Timestamp("2014-03-21 13:31", tz='UTC')

        self.assertEquals(window.loc[day1_end, 1], 533.086)
        self.assertEquals(window.loc[day2_start, 1], 533.087)
        self.assertEquals(window.loc[day2_end, 1], 533.853)
        self.assertEquals(window.loc[day3_start, 1], 533.854)

    def test_ffill_minute_equity_window_starts_with_nan(self):
        """
        Test that forward filling does not leave leading nan if there is data
        available before the start of the window.
        """

        window = self.data_portal.get_history_window(
            [self.FOO],
            pd.Timestamp("2014-03-21 13:41:00+00:00", tz='UTC'),
            20,
            "1m",
            "price"
        )

        # The previous value is on 2014-03-20, and there is a split between
        # the two dates, the spot price of the latest value is 1066.92, with
        # the expected result being 533.46 after the 2:1 split is applied.
        expected = np.append(np.full(19, 533.460),
                             np.array(529.601))

        np.testing.assert_allclose(window.loc[:, self.FOO], expected)

    def test_ffill_minute_equity_window_no_previous(self):
        """
        Test that forward filling handles the case where the window starts
        with a nan, and there are no previous values.
        """

        window = self.data_portal.get_history_window(
            [self.FOO],
            pd.Timestamp("2014-03-19 13:41:00+00:00", tz='UTC'),
            20,
            "1m",
            "price"
        )

        # There should be no values, since there is no data before 2014-03-20
        expected = np.full(20, np.nan)

        np.testing.assert_allclose(window.loc[:, self.FOO], expected)

    def test_ffill_minute_future_window_starts_with_nan(self):
        """
        Test that forward filling does not leave leading nan if there is data
        available before the start of the window.
        """

        window = self.data_portal.get_history_window(
            [self.FUTURE_ASSET3],
            pd.Timestamp("2014-03-20 15:00:00+00:00", tz='UTC'),
            20,
            "1m",
            "price"
        )

        # 31468 is the value at 2014-03-20 13:59, and should be the forward
        # filled value until 2015-03-20 15:00
        expected = np.append(np.full(19, 31468),
                             np.array(31529))

        np.testing.assert_allclose(window.loc[:, self.FUTURE_ASSET3],
                                   expected)

    def test_ffill_daily_equity_window_starts_with_nan(self):
        """
        Test that forward filling does not leave leading nan if there is data
        available before the start of the window.
        """
        window = self.data_portal.get_history_window(
            [self.FOO],
            pd.Timestamp("2014-03-21 00:00:00+00:00", tz='UTC'),
            2,
            "1d",
            "price"
        )

        # The previous value is on 2014-03-20, and there is a split between
        # the two dates, the spot price of the latest value is 106.692, with
        # the expected result being 533.46 after the 2:1 split is applied.
        expected = np.array([
            53.346,
            52.95,
        ])

        np.testing.assert_allclose(window.loc[:, self.FOO], expected)

    def test_minute_window_starts_before_trading_start(self):
        portal = self.data_portal

        # get a 50-bar minute history for MSFT starting 5 minutes into 3/20,
        # its first trading day
        window = portal.get_history_window(
            [2],
            pd.Timestamp("2014-03-20 13:35:00", tz='UTC'),
            50,
            "1m",
            "high",
        )

        self.assertEqual(len(window), 50)
        reference = [107.081, 109.476, 102.316, 107.861, 106.040]
        for i in range(0, 4):
            self.assertEqual(window.iloc[-5 + i].loc[2], reference[i])

        # get history for two securities at the same time, where one starts
        # trading a day later than the other
        window2 = portal.get_history_window(
            [1, 2],
            pd.Timestamp("2014-03-20 13:35:00", tz='UTC'),
            50,
            "1m",
            "low",
        )

        self.assertEqual(len(window2), 50)
        reference2 = {
            1: [1059.318, 1055.914, 1061.136, 1063.698, 1055.964],
            2: [98.902, 99.841, 90.984, 99.891, 98.027]
        }

        for i in range(0, 45):
            self.assertFalse(np.isnan(window2.iloc[i].loc[1]))

            # there should be 45 NaNs for MSFT until it starts trading
            self.assertTrue(np.isnan(window2.iloc[i].loc[2]))

        for i in range(0, 4):
            self.assertEquals(window2.iloc[-5 + i].loc[1],
                              reference2[1][i])
            self.assertEquals(window2.iloc[-5 + i].loc[2],
                              reference2[2][i])

    def test_minute_window_ends_before_trading_start(self):
        # entire window is before the trading start
        window = self.data_portal.get_history_window(
            [2],
            pd.Timestamp("2014-02-05 14:35:00", tz='UTC'),
            100,
            "1m",
            "high"
        )

        self.assertEqual(len(window), 100)
        for i in range(0, 100):
            self.assertTrue(np.isnan(window.iloc[i].loc[2]))

    def test_minute_window_ends_after_trading_end(self):
        portal = self.data_portal

        window = portal.get_history_window(
            [2],
            pd.Timestamp("2014-03-24 13:35:00", tz='UTC'),
            50,
            "1m",
            "high",
        )

        # should be 45 non-NaNs then 5 NaNs as MSFT has stopped trading at
        # the end of the day 2014-03-21 (and the 22nd and 23rd is weekend)
        self.assertEqual(len(window), 50)

        for i in range(0, 45):
            self.assertFalse(np.isnan(window.iloc[i].loc[2]))

        for i in range(46, 50):
            self.assertTrue(np.isnan(window.iloc[i].loc[2]))

    def test_minute_window_starts_after_trading_end(self):
        # entire window is after the trading end
        window = self.data_portal.get_history_window(
            [2],
            pd.Timestamp("2014-04-02 14:35:00", tz='UTC'),
            100,
            "1m",
            "high"
        )

        self.assertEqual(len(window), 100)
        for i in range(0, 100):
            self.assertTrue(np.isnan(window.iloc[i].loc[2]))

    def test_minute_window_starts_before_1_2_2002(self):
        window = self.data_portal.get_history_window(
            [3],
            pd.Timestamp("2002-01-02 14:35:00", tz='UTC'),
            50,
            "1m",
            "close_price"
        )

        self.assertEqual(len(window), 50)
        for i in range(0, 45):
            self.assertTrue(np.isnan(window.iloc[i].loc[3]))

        for i in range(46, 50):
            self.assertFalse(np.isnan(window.iloc[i].loc[3]))

    def test_minute_early_close(self):
        # market was closed early on 7/3, and that's reflected in our
        # fake IBM minute data.  also, IBM had a split that takes effect
        # right after the early close.

        # five minutes into the day after an early close, get 20 1m bars
        window = self.data_portal.get_history_window(
            [self.IBM],
            pd.Timestamp("2014-07-07 13:35:00", tz='UTC'),
            20,
            "1m",
            "high"
        )

        self.assertEqual(len(window), 20)

        reference = [27134.486, 27134.802, 27134.660, 27132.813, 27130.964,
                     27133.767, 27133.268, 27131.510, 27134.946, 27132.400,
                     27134.350, 27130.588, 27132.528, 27130.418, 27131.040,
                     27132.664, 27131.307, 27133.978, 27132.779, 27134.476]

        for i in range(0, 20):
            self.assertAlmostEquals(window.iloc[i].loc[self.IBM], reference[i])

    def test_minute_merger(self):
        def check(field, ref):
            window = self.data_portal.get_history_window(
                [self.C],
                pd.Timestamp("2014-07-16 13:35", tz='UTC'),
                10,
                "1m",
                field
            )

            self.assertEqual(len(window), len(ref))

            for i in range(0, len(ref) - 1):
                self.assertEquals(window.iloc[i].loc[self.C], ref[i])

        open_ref = [71.99, 71.991, 71.992, 71.996, 71.996,
                    72.000, 72.001, 72.002, 72.004, 72.005]
        high_ref = [77.334, 80.196, 80.387, 72.331, 79.184,
                    75.439, 81.176, 78.564, 80.498, 82.000]
        low_ref = [62.621, 70.427, 65.572, 68.357, 63.623,
                   69.805, 67.245, 64.238, 64.487, 71.864]
        close_ref = [69.977, 75.311, 72.979, 70.344, 71.403,
                     72.622, 74.210, 71.401, 72.492, 73.669]
        vol_ref = [12663, 12662, 12661, 12661, 12660, 12661,
                   12663, 12662, 12663, 12662]

        check("open_price", open_ref)
        check("high", high_ref)
        check("low", low_ref)
        check("close_price", close_ref)
        check("price", close_ref)
        check("volume", vol_ref)

    def test_minute_forward_fill(self):
        # only forward fill if ffill=True AND we are asking for "price"

        # our fake TSLA data (sid 4) is missing a bunch of minute bars
        # right after the open on 2002-01-02

        for field in ["open_price", "high", "low", "volume", "close_price"]:
            no_ffill = self.data_portal.get_history_window(
                [4],
                pd.Timestamp("2002-01-02 21:00:00", tz='UTC'),
                390,
                "1m",
                field
            )

            missing_bar_indices = [1, 3, 5, 7, 9, 11, 13]
            if field == 'volume':
                for bar_idx in missing_bar_indices:
                    self.assertEqual(no_ffill.iloc[bar_idx].loc[4], 0)
            else:
                for bar_idx in missing_bar_indices:
                    self.assertTrue(np.isnan(no_ffill.iloc[bar_idx].loc[4]))

        ffill_window = self.data_portal.get_history_window(
            [4],
            pd.Timestamp("2002-01-02 21:00:00", tz='UTC'),
            390,
            "1m",
            "price"
        )

        for i in range(0, 390):
            self.assertFalse(np.isnan(ffill_window.iloc[i].loc[4]))

        # 2002-01-02 14:31:00+00:00  126.183
        # 2002-01-02 14:32:00+00:00  126.183
        # 2002-01-02 14:33:00+00:00  125.648
        # 2002-01-02 14:34:00+00:00  125.648
        # 2002-01-02 14:35:00+00:00  126.016
        # 2002-01-02 14:36:00+00:00  126.016
        # 2002-01-02 14:37:00+00:00  127.918
        # 2002-01-02 14:38:00+00:00  127.918
        # 2002-01-02 14:39:00+00:00  126.423
        # 2002-01-02 14:40:00+00:00  126.423
        # 2002-01-02 14:41:00+00:00  129.825
        # 2002-01-02 14:42:00+00:00  129.825
        # 2002-01-02 14:43:00+00:00  125.392
        # 2002-01-02 14:44:00+00:00  125.392

        vals = [126.183, 125.648, 126.016, 127.918, 126.423, 129.825, 125.392]
        for idx, val in enumerate(vals):
            self.assertEqual(ffill_window.iloc[2 * idx].loc[4], val)
            self.assertEqual(ffill_window.iloc[(2 * idx) + 1].loc[4], val)

        # make sure that if we pass ffill=False with field="price", we do
        # not ffill
        really_no_ffill_window = self.data_portal.get_history_window(
            [4],
            pd.Timestamp("2002-01-02 21:00:00", tz='UTC'),
            390,
            "1m",
            "price",
            ffill=False
        )

        for idx, val in enumerate(vals):
            idx1 = 2 * idx
            idx2 = idx1 + 1
            self.assertEqual(really_no_ffill_window.iloc[idx1].loc[4], val)
            self.assertTrue(np.isnan(really_no_ffill_window.iloc[idx2].loc[4]))

    def test_daily_functionality(self):
        # 9 daily bars
        # 2014-03-10,183999.0,186400.0,183601.0,186400.0,400
        # 2014-03-11,186925.0,187490.0,185910.0,187101.0,600
        # 2014-03-12,186498.0,187832.0,186005.0,187750.0,300
        # 2014-03-13,188150.0,188852.0,185254.0,185750.0,700
        # 2014-03-14,185825.0,186507.0,183418.0,183860.0,600
        # 2014-03-17,184350.0,185790.0,184350.0,185050.0,400
        # 2014-03-18,185400.0,185400.0,183860.0,184860.0,200
        # 2014-03-19,184860.0,185489.0,182764.0,183860.0,200
        # 2014-03-20,183999.0,186742.0,183630.0,186540.0,300

        # 5 one-minute bars that will be aggregated
        # 2014-03-21 13:31:00+00:00,185422401,185426332,185413974,185420153,304
        # 2014-03-21 13:32:00+00:00,185422402,185424165,185417717,185420941,300
        # 2014-03-21 13:33:00+00:00,185422403,185430663,185419420,185425041,303
        # 2014-03-21 13:34:00+00:00,185422403,185431290,185417079,185424184,302
        # 2014-03-21 13:35:00+00:00,185422405,185430210,185416293,185423251,302

        def run_query(field, values):
            window = self.data_portal.get_history_window(
                [self.BRKA],
                pd.Timestamp("2014-03-21 13:35", tz='UTC'),
                10,
                "1d",
                field
            )

            self.assertEqual(len(window), 10)

            for i in range(0, 10):
                self.assertEquals(window.iloc[i].loc[self.BRKA],
                                  values[i])

        # last value is the first minute's open
        opens = [183999, 186925, 186498, 188150, 185825, 184350,
                 185400, 184860, 183999, 185422.401]

        # last value is the last minute's close
        closes = [186400, 187101, 187750, 185750, 183860, 185050,
                  184860, 183860, 186540, 185423.251]

        # last value is the highest high value
        highs = [186400, 187490, 187832, 188852, 186507, 185790,
                 185400, 185489, 186742, 185431.290]

        # last value is the lowest low value
        lows = [183601, 185910, 186005, 185254, 183418, 184350, 183860,
                182764, 183630, 185413.974]

        # last value is the sum of all the minute volumes
        volumes = [400, 600, 300, 700, 600, 400, 200, 200, 300, 1511]

        run_query("open_price", opens)
        run_query("close_price", closes)
        run_query("price", closes)
        run_query("high", highs)
        run_query("low", lows)
        run_query("volume", volumes)

    def test_daily_splits_with_no_minute_data(self):
        # scenario is that we have daily data for AAPL through 6/11,
        # but we have no minute data for AAPL on 6/11. there's also a split
        # for AAPL on 6/9.
        splits = DataFrame(
            [
                {
                    'effective_date': str_to_seconds('2014-06-09'),
                    'ratio': (1 / 7.0),
                    'sid': self.AAPL,
                }
            ],
            columns=['effective_date', 'ratio', 'sid'])

        self.create_fake_adjustments(self.tempdir,
                                     "adjustments2.sqlite",
                                     splits=splits)

        portal = self.get_portal(adjustments_filename="adjustments2.sqlite")

        def test_window(field, reference, ffill=True):
            window = portal.get_history_window(
                [self.AAPL],
                pd.Timestamp("2014-06-11 15:30", tz='UTC'),
                6,
                "1d",
                field,
                ffill
            )

            self.assertEqual(len(window), 6)

            for i in range(0, 5):
                self.assertEquals(window.iloc[i].loc[self.AAPL],
                                  reference[i])

            if ffill and field == "price":
                last_val = window.iloc[5].loc[self.AAPL]
                second_to_last_val = window.iloc[4].loc[self.AAPL]

                self.assertEqual(last_val, second_to_last_val)
            else:
                if field == "volume":
                    self.assertEqual(window.iloc[5].loc[self.AAPL], 0)
                else:
                    self.assertTrue(np.isnan(window.iloc[5].loc[self.AAPL]))

        # 2014-06-04,637.4400099999999,647.8899690000001,636.110046,644.819992,p
        # 2014-06-05,646.20005,649.370003,642.610008,647.349983,75951400
        # 2014-06-06,649.900002,651.259979,644.469971,645.570023,87484600
        # 2014-06-09,92.699997,93.879997,91.75,93.699997,75415000
        # 2014-06-10,94.730003,95.050003,93.57,94.25,62777000
        open_data = [91.063, 92.314, 92.843, 92.699, 94.730]
        test_window("open_price", open_data, ffill=False)
        test_window("open_price", open_data)

        high_data = [92.556, 92.767, 93.037, 93.879, 95.050]
        test_window("high", high_data, ffill=False)
        test_window("high", high_data)

        low_data = [90.873, 91.801, 92.067, 91.750, 93.570]
        test_window("low", low_data, ffill=False)
        test_window("low", low_data)

        close_data = [92.117, 92.478, 92.224, 93.699, 94.250]
        test_window("close_price", close_data, ffill=False)
        test_window("close_price", close_data)
        test_window("price", close_data, ffill=False)
        test_window("price", close_data)

        vol_data = [587093500, 531659800, 612392200, 75415000, 62777000]
        test_window("volume", vol_data)
        test_window("volume", vol_data, ffill=False)

    def test_daily_window_starts_before_trading_start(self):
        portal = self.data_portal

        # MSFT started on 3/3/2014, so try to go before that
        window = portal.get_history_window(
            [self.MSFT],
            pd.Timestamp("2014-03-05 13:35:00", tz='UTC'),
            5,
            "1d",
            "high"
        )

        self.assertEqual(len(window), 5)

        # should be two empty days, then 3/3 and 3/4, then
        # an empty day because we don't have minute data for 3/5
        self.assertTrue(np.isnan(window.iloc[0].loc[self.MSFT]))
        self.assertTrue(np.isnan(window.iloc[1].loc[self.MSFT]))
        self.assertEquals(window.iloc[2].loc[self.MSFT], 38.130)
        self.assertEquals(window.iloc[3].loc[self.MSFT], 38.48)
        self.assertTrue(np.isnan(window.iloc[4].loc[self.MSFT]))

    def test_daily_window_ends_before_trading_start(self):
        portal = self.data_portal

        # MSFT started on 3/3/2014, so try to go before that
        window = portal.get_history_window(
            [self.MSFT],
            pd.Timestamp("2014-02-28 13:35:00", tz='UTC'),
            5,
            "1d",
            "high"
        )

        self.assertEqual(len(window), 5)
        for i in range(0, 5):
            self.assertTrue(np.isnan(window.iloc[i].loc[self.MSFT]))

    def test_daily_window_starts_after_trading_end(self):
        # MSFT stopped trading EOD Friday 8/29/2014
        window = self.data_portal.get_history_window(
            [self.MSFT],
            pd.Timestamp("2014-09-12 13:35:00", tz='UTC'),
            8,
            "1d",
            "high",
        )

        self.assertEqual(len(window), 8)
        for i in range(0, 8):
            self.assertTrue(np.isnan(window.iloc[i].loc[self.MSFT]))

    def test_daily_window_ends_after_trading_end(self):
        # MSFT stopped trading EOD Friday 8/29/2014
        window = self.data_portal.get_history_window(
            [self.MSFT],
            pd.Timestamp("2014-09-04 13:35:00", tz='UTC'),
            10,
            "1d",
            "high",
        )

        # should be 7 non-NaNs (8/21-8/22, 8/25-8/29) and 3 NaNs (9/2 - 9/4)
        # (9/1/2014 is labor day)
        self.assertEqual(len(window), 10)

        for i in range(0, 7):
            self.assertFalse(np.isnan(window.iloc[i].loc[self.MSFT]))

        for i in range(7, 10):
            self.assertTrue(np.isnan(window.iloc[i].loc[self.MSFT]))

    def test_empty_sid_list(self):
        portal = self.data_portal

        fields = ["open_price",
                  "close_price",
                  "high",
                  "low",
                  "volume",
                  "price"]
        freqs = ["1m", "1d"]

        for field in fields:
            for freq in freqs:
                window = portal.get_history_window(
                    [],
                    pd.Timestamp("2014-06-11 15:30", tz='UTC'),
                    6,
                    freq,
                    field
                )

                self.assertEqual(len(window), 6)

                for i in range(0, 6):
                    self.assertEqual(len(window.iloc[i]), 0)

    def test_daily_window_starts_before_minute_data(self):

        env = TradingEnvironment()
        asset_info = make_simple_asset_info(
            [self.GS],
            Timestamp('1999-04-05'),
            Timestamp('2004-08-30'),
            ['GS']
        )
        env.write_data(equities_df=asset_info)
        portal = self.get_portal(env=env)

        window = portal.get_history_window(
            [self.GS],
            # 3rd day of daily data for GS, minute data starts in 2002.
            pd.Timestamp("1999-04-07 14:35:00", tz='UTC'),
            10,
            "1d",
            "low"
        )

        # 12/20, 12/21, 12/24, 12/26, 12/27, 12/28, 12/31 should be NaNs
        # 1/2 and 1/3 should be non-NaN
        # 1/4 should be NaN (since we don't have minute data for it)

        self.assertEqual(len(window), 10)

        for i in range(0, 7):
            self.assertTrue(np.isnan(window.iloc[i].loc[self.GS]))

        for i in range(8, 9):
            self.assertFalse(np.isnan(window.iloc[i].loc[self.GS]))

        self.assertTrue(np.isnan(window.iloc[9].loc[self.GS]))

    def test_minute_window_ends_before_1_2_2002(self):
        with self.assertRaises(ValueError):
            self.data_portal.get_history_window(
                [self.GS],
                pd.Timestamp("2001-12-31 14:35:00", tz='UTC'),
                50,
                "1m",
                "close_price"
            )

    def test_bad_history_inputs(self):
        portal = self.data_portal

        # bad fieldname
        for field in ["foo", "bar", "", "5"]:
            with self.assertRaises(ValueError):
                portal.get_history_window(
                    [self.AAPL],
                    pd.Timestamp("2014-06-11 15:30", tz='UTC'),
                    6,
                    "1d",
                    field
                )

        # bad frequency
        for freq in ["2m", "30m", "3d", "300d", "", "5"]:
            with self.assertRaises(ValueError):
                portal.get_history_window(
                    [self.AAPL],
                    pd.Timestamp("2014-06-11 15:30", tz='UTC'),
                    6,
                    freq,
                    "volume"
                )

    def test_daily_merger(self):
        def check(field, ref):
            window = self.data_portal.get_history_window(
                [self.C],
                pd.Timestamp("2014-07-17 13:35", tz='UTC'),
                4,
                "1d",
                field
            )

            self.assertEqual(len(window), len(ref),)

            for i in range(0, len(ref) - 1):
                self.assertEquals(window.iloc[i].loc[self.C], ref[i], i)

        # 2014-07-14 00:00:00+00:00,139.18,139.14,139.2,139.17,12351
        # 2014-07-15 00:00:00+00:00,139.2,139.2,139.18,139.19,12354
        # 2014-07-16 00:00:00+00:00,69.58,69.56,69.57,69.565,12352
        # 2014-07-17 13:31:00+00:00,72767,80146,63406,71776,12876
        # 2014-07-17 13:32:00+00:00,72769,76943,68907,72925,12875
        # 2014-07-17 13:33:00+00:00,72771,76127,63194,69660,12875
        # 2014-07-17 13:34:00+00:00,72774,79349,69771,74560,12877
        # 2014-07-17 13:35:00+00:00,72776,75340,68970,72155,12879

        open_ref = [69.59, 69.6, 69.58, 72.767]
        high_ref = [69.57, 69.6, 69.56, 80.146]
        low_ref = [69.6, 69.59, 69.57, 63.194]
        close_ref = [69.585, 69.595, 69.565, 72.155]
        vol_ref = [12351, 12354, 12352, 64382]

        check("open_price", open_ref)
        check("high", high_ref)
        check("low", low_ref)
        check("close_price", close_ref)
        check("price", close_ref)
        check("volume", vol_ref)

    def test_minute_adjustments_as_of_lookback_date(self):
        # AAPL has splits on 2014-03-20 and 2014-03-21
        window_0320 = self.data_portal.get_history_window(
            [self.AAPL],
            pd.Timestamp("2014-03-20 13:35", tz='UTC'),
            395,
            "1m",
            "open_price"
        )

        window_0321 = self.data_portal.get_history_window(
            [self.AAPL],
            pd.Timestamp("2014-03-21 13:35", tz='UTC'),
            785,
            "1m",
            "open_price"
        )

        for i in range(0, 395):
            # history on 3/20, since the 3/21 0.5 split hasn't
            # happened yet, should return values 2x larger than history on
            # 3/21
            self.assertEqual(window_0320.iloc[i].loc[self.AAPL],
                             window_0321.iloc[i].loc[self.AAPL] * 2)

    def test_daily_adjustments_as_of_lookback_date(self):
        window_0402 = self.data_portal.get_history_window(
            [self.IBM],
            pd.Timestamp("2014-04-02 13:35", tz='UTC'),
            23,
            "1d",
            "open_price"
        )

        window_0702 = self.data_portal.get_history_window(
            [self.IBM],
            pd.Timestamp("2014-07-02 13:35", tz='UTC'),
            86,
            "1d",
            "open_price"
        )

        for i in range(0, 22):
            self.assertEqual(window_0402.iloc[i].loc[self.IBM],
                             window_0702.iloc[i].loc[self.IBM] * 2)

    def test_minute_dividends(self):
        def check(field, ref):
            window = self.data_portal.get_history_window(
                [self.DIVIDEND_SID],
                pd.Timestamp("2014-03-18 13:35", tz='UTC'),
                10,
                "1m",
                field
            )

            self.assertEqual(len(window), len(ref))

            np.testing.assert_allclose(window.loc[:, self.DIVIDEND_SID], ref)

        # the DIVIDEND stock has dividends on 2014-03-18 (0.98)
        # 2014-03-17 19:56:00+00:00,118923,123229,112445,117837,2273
        # 2014-03-17 19:57:00+00:00,118927,122997,117911,120454,2274
        # 2014-03-17 19:58:00+00:00,118930,129112,111136,120124,2274
        # 2014-03-17 19:59:00+00:00,118932,126147,112112,119129,2276
        # 2014-03-17 20:00:00+00:00,118932,124541,108717,116628,2275
        # 2014-03-18 13:31:00+00:00,116457,120731,114148,117439,2274
        # 2014-03-18 13:32:00+00:00,116461,116520,106572,111546,2275
        # 2014-03-18 13:33:00+00:00,116461,117115,108506,112810,2274
        # 2014-03-18 13:34:00+00:00,116461,119787,108861,114323,2273
        # 2014-03-18 13:35:00+00:00,116464,117221,112698,114960,2272

        open_ref = [116.545,  # 2014-03-17 19:56:00+00:00
                    116.548,  # 2014-03-17 19:57:00+00:00
                    116.551,  # 2014-03-17 19:58:00+00:00
                    116.553,  # 2014-03-17 19:59:00+00:00
                    116.553,  # 2014-03-17 20:00:00+00:00
                    116.457,  # 2014-03-18 13:31:00+00:00
                    116.461,  # 2014-03-18 13:32:00+00:00
                    116.461,  # 2014-03-18 13:33:00+00:00
                    116.461,  # 2014-03-18 13:34:00+00:00
                    116.464]  # 2014-03-18 13:35:00+00:00

        high_ref = [120.764,  # 2014-03-17 19:56:00+00:00
                    120.537,  # 2014-03-17 19:57:00+00:00
                    126.530,  # 2014-03-17 19:58:00+00:00
                    123.624,  # 2014-03-17 19:59:00+00:00
                    122.050,  # 2014-03-17 20:00:00+00:00
                    120.731,  # 2014-03-18 13:31:00+00:00
                    116.520,  # 2014-03-18 13:32:00+00:00
                    117.115,  # 2014-03-18 13:33:00+00:00
                    119.787,  # 2014-03-18 13:34:00+00:00
                    117.221]  # 2014-03-18 13:35:00+00:00

        low_ref = [110.196,  # 2014-03-17 19:56:00+00:00
                   115.553,  # 2014-03-17 19:57:00+00:00
                   108.913,  # 2014-03-17 19:58:00+00:00
                   109.870,  # 2014-03-17 19:59:00+00:00
                   106.543,  # 2014-03-17 20:00:00+00:00
                   114.148,  # 2014-03-18 13:31:00+00:00
                   106.572,  # 2014-03-18 13:32:00+00:00
                   108.506,  # 2014-03-18 13:33:00+00:00
                   108.861,  # 2014-03-18 13:34:00+00:00
                   112.698]  # 2014-03-18 13:35:00+00:00

        close_ref = [115.480,  # 2014-03-17 19:56:00+00:00
                     118.045,  # 2014-03-17 19:57:00+00:00
                     117.722,  # 2014-03-17 19:58:00+00:00
                     116.746,  # 2014-03-17 19:59:00+00:00
                     114.295,  # 2014-03-17 20:00:00+00:00
                     117.439,  # 2014-03-18 13:31:00+00:00
                     111.546,  # 2014-03-18 13:32:00+00:00
                     112.810,  # 2014-03-18 13:33:00+00:00
                     114.323,  # 2014-03-18 13:34:00+00:00
                     114.960]  # 2014-03-18 13:35:00+00:00

        volume_ref = [2273,  # 2014-03-17 19:56:00+00:00
                      2274,  # 2014-03-17 19:57:00+00:00
                      2274,  # 2014-03-17 19:58:00+00:00
                      2276,  # 2014-03-17 19:59:00+00:00
                      2275,  # 2014-03-17 20:00:00+00:00
                      2274,  # 2014-03-18 13:31:00+00:00
                      2275,  # 2014-03-18 13:32:00+00:00
                      2274,  # 2014-03-18 13:33:00+00:00
                      2273,  # 2014-03-18 13:34:00+00:00
                      2272]  # 2014-03-18 13:35:00+00:00

        check("open_price", open_ref)
        check("high", high_ref)
        check("low", low_ref)
        check("close_price", close_ref)
        check("price", close_ref)
        check("volume", volume_ref)

    def test_daily_dividends(self):
        def check(field, ref):
            window = self.data_portal.get_history_window(
                [self.DIVIDEND_SID],
                pd.Timestamp("2014-03-21 13:35", tz='UTC'),
                6,
                "1d",
                field
            )

            self.assertEqual(len(window), len(ref))

            np.testing.assert_allclose(window.loc[:, self.DIVIDEND_SID], ref)

        # 2014-03-14 00:00:00+00:00,106408,106527,103498,105012,950
        # 2014-03-17 00:00:00+00:00,106411,110252,99877,105064,950
        # 2014-03-18 00:00:00+00:00,104194,110891,95342,103116,972
        # 2014-03-19 00:00:00+00:00,104198,107086,102615,104851,973
        # 2014-03-20 00:00:00+00:00,100032,102989,92179,97584,1016
        # 2014-03-21 13:31:00+00:00,114098,120818,110333,115575,2866
        # 2014-03-21 13:32:00+00:00,114099,120157,105353,112755,2866
        # 2014-03-21 13:33:00+00:00,114099,122263,108838,115550,2867
        # 2014-03-21 13:34:00+00:00,114101,116620,106654,111637,2867
        # 2014-03-21 13:35:00+00:00,114104,123773,107769,115771,2867

        open_ref = [100.108,  # 2014-03-14 00:00:00+00:00
                    100.111,  # 2014-03-17 00:00:00+00:00
                    100.026,  # 2014-03-18 00:00:00+00:00
                    100.030,  # 2014-03-19 00:00:00+00:00
                    100.032,  # 2014-03-20 00:00:00+00:00
                    114.098]  # 2014-03-21 00:00:00+00:00

        high_ref = [100.221,  # 2014-03-14 00:00:00+00:00
                    103.725,  # 2014-03-17 00:00:00+00:00
                    106.455,  # 2014-03-18 00:00:00+00:00
                    102.803,  # 2014-03-19 00:00:00+00:00
                    102.988,  # 2014-03-20 00:00:00+00:00
                    123.773]  # 2014-03-21 00:00:00+00:00

        low_ref = [97.370,  # 2014-03-14 00:00:00+00:00
                   93.964,  # 2014-03-17 00:00:00+00:00
                   91.528,  # 2014-03-18 00:00:00+00:00
                   98.510,  # 2014-03-19 00:00:00+00:00
                   92.179,  # 2014-03-20 00:00:00+00:00
                   105.353]  # 2014-03-21 00:00:00+00:00

        close_ref = [98.795,  # 2014-03-14 00:00:00+00:00
                     98.844,  # 2014-03-17 00:00:00+00:00
                     98.991,  # 2014-03-18 00:00:00+00:00
                     100.657,  # 2014-03-19 00:00:00+00:00
                     97.584,  # 2014-03-20 00:00:00+00:00
                     115.771]  # 2014-03-21 00:00:00+00:00

        volume_ref = [950,  # 2014-03-14 00:00:00+00:00
                      950,  # 2014-03-17 00:00:00+00:00
                      972,  # 2014-03-18 00:00:00+00:00
                      973,  # 2014-03-19 00:00:00+00:00
                      1016,  # 2014-03-20 00:00:00+00:00
                      14333]  # 2014-03-21 00:00:00+00:00

        check("open_price", open_ref)
        check("high", high_ref)
        check("low", low_ref)
        check("close_price", close_ref)
        check("price", close_ref)
        check("volume", volume_ref)

    @parameterized.expand([('open', 0),
                           ('high', 10000),
                           ('low', 20000),
                           ('close', 30000),
                           ('price', 30000),
                           ('volume', 40000)])
    def test_futures_history_minutes(self, field, offset):
        # our history data, for self.FUTURE_ASSET, is 10,000 bars starting at
        # self.futures_start_dt.  Those 10k bars are 24/7.

        # = 2015-11-30 18:50 UTC, 13:50 Eastern = during market hours
        futures_end_dt = \
            self.futures_start_dates[self.FUTURE_ASSET] + \
            timedelta(minutes=9999)

        window = self.data_portal.get_history_window(
            [self.FUTURE_ASSET],
            futures_end_dt,
            1000,
            "1m",
            field
        )

        # check the minutes are right
        reference_minutes = self.env.market_minute_window(
            futures_end_dt, 1000, step=-1
        )[::-1]

        np.testing.assert_array_equal(window.index, reference_minutes)

        # check the values

        # 2015-11-24 18:41
        # ...
        # 2015-11-24 21:00
        # 2015-11-25 14:31
        # ...
        # 2015-11-25 21:00
        # 2015-11-27 14:31
        # ...
        # 2015-11-27 18:00  # early close
        # 2015-11-30 14:31
        # ...
        # 2015-11-30 18:50

        reference_values = pd.date_range(
            start=self.futures_start_dates[self.FUTURE_ASSET],
            end=futures_end_dt,
            freq="T"
        )

        for idx, dt in enumerate(window.index):
            date_val = reference_values.searchsorted(dt)
            self.assertEqual(offset + date_val,
                             window.iloc[idx][self.FUTURE_ASSET])

    def test_history_minute_blended(self):
        window = self.data_portal.get_history_window(
            [self.FUTURE_ASSET2, self.AAPL],
            pd.Timestamp("2014-03-21 20:00", tz='UTC'),
            200,
            "1m",
            "price"
        )

        # just a sanity check
        self.assertEqual(200, len(window[self.AAPL]))
        self.assertEqual(200, len(window[self.FUTURE_ASSET2]))

    def test_futures_history_daily(self):
        # get 3 days ending 11/30 10:00 am Eastern
        # = 11/25, 11/27 (half day), 11/30 (partial)

        window = self.data_portal.get_history_window(
            [self.env.asset_finder.retrieve_asset(self.FUTURE_ASSET)],
            pd.Timestamp("2015-11-30 15:00", tz='UTC'),
            3,
            "1d",
            "high"
        )

        self.assertEqual(3, len(window[self.FUTURE_ASSET]))

        np.testing.assert_array_equal([12929.0, 15629.0, 19769.0],
                                      window.values.T[0])
