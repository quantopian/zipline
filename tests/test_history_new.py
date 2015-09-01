from os.path import dirname, join, realpath
from unittest import TestCase

from testfixtures import TempDirectory
import numpy as np
from numpy import array
import pandas as pd
from pandas import (
    read_csv,
    Timestamp,
    DataFrame, DatetimeIndex)

from six import iteritems

from zipline.assets import AssetFinder
from zipline.data.data_portal import DataPortal
from zipline.data.ffc.loaders.us_equity_pricing import DailyBarWriterFromCSVs, \
    SQLiteAdjustmentWriter
from zipline.utils.test_utils import make_simple_asset_info, str_to_seconds
from zipline.data.minute_writer import MinuteBarWriterFromCSVs
from zipline.utils.tradingcalendar import trading_days

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
    'modelling_inputs',
)


class HistoryTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AAPL = 1
        cls.MSFT = 2
        cls.DELL = 3
        cls.TSLA = 4
        cls.assets = [cls.AAPL, cls.MSFT, cls.DELL, cls.TSLA]

        asset_info = make_simple_asset_info(
            cls.assets,
            Timestamp('2014'),
            Timestamp('2015'),
            ['AAPL', 'MSFT', 'DELL', 'TSLA']
        )

        cls.asset_finder = AssetFinder(asset_info)

        cls.tempdir = TempDirectory()
        cls.tempdir.create()

        try:
            cls.create_fake_minute_data(cls.tempdir)
            cls.create_fake_daily_data(cls.tempdir)
            cls.create_fake_adjustments(cls.tempdir)
        except:
            cls.tempdir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def create_fake_minute_data(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_MINUTE_RESOURCE_PATH, 'AAPL_minute.csv'),
            cls.MSFT: join(TEST_MINUTE_RESOURCE_PATH, 'MSFT_minute.csv'),
            cls.DELL: join(TEST_MINUTE_RESOURCE_PATH, 'DELL_minute.csv'),
            cls.TSLA: join(TEST_MINUTE_RESOURCE_PATH, "TSLA_minute.csv")
        }

        MinuteBarWriterFromCSVs(resources).write(tempdir.path, cls.assets)

    @classmethod
    def create_fake_daily_data(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_DAILY_RESOURCE_PATH, 'AAPL.csv'),
            cls.MSFT: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),
            cls.DELL: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),
            cls.TSLA: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv')
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
    def create_fake_adjustments(cls, tempdir):
        writer = SQLiteAdjustmentWriter(tempdir.getpath('adjustments.sqlite'))

        mergers = dividends = DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': array([], dtype=int),
                'ratio': array([], dtype=float),
                'sid': array([], dtype=int),
            },
            index=DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )

        splits = DataFrame([
                {'effective_date': str_to_seconds("2014-03-20"),
                 'ratio': 0.5,
                 'sid': 1},
                {'effective_date': str_to_seconds("2014-03-21"),
                 'ratio': 0.5,
                 'sid': 1},
            ],
            columns=['effective_date', 'ratio', 'sid'],
        )

        writer.write(splits, mergers, dividends)

    def get_portal(self):
        temp_path = self.tempdir.path

        return DataPortal(
            None,
            findata_dir=temp_path,
            asset_finder=self.asset_finder,
            daily_equities_path=join(temp_path, "test_daily_data.bcolz"),
            adjustments_path=join(temp_path, "adjustments.sqlite")
        )

    def test_basic_minute_functionality(self):
        portal = self.get_portal()

        # get a 5-bar minute history from the very end of the available data
        window = portal.get_history_window(
            [1],
            pd.Timestamp("2014-03-21 18:23:00+00:00", tz='UTC'),
            5,
            "minute",
            "open"
        )

        self.assertEqual(len(window), 5)
        reference = [534.469, 534.471, 534.475, 534.477, 534.477]
        for i in range(0, 4):
            self.assertEqual(window.iloc[-5 + i].loc[1], reference[i])

    def test_minute_splits(self):
        portal = self.get_portal()

        window = portal.get_history_window(
            [1],
            pd.Timestamp("2014-03-21 18:30:00+00:00", tz='UTC'),
            1000,
            "minute",
            "open"
        )

        self.assertEqual(len(window), 1000)

        # there are two splits for AAPL (on 2014-03-20 and 2014-03-21),
        # each with ratio 0.5).

        day1_end = pd.Timestamp("2014-03-19 20:00", tz='UTC')
        day2_start = pd.Timestamp("2014-03-20 13:31", tz='UTC')
        day2_end = pd.Timestamp("2014-03-20 20:00", tz='UTC')
        day3_start = pd.Timestamp("2014-03-21 13:31", tz='UTC')

        self.assertEqual(window.loc[day1_end, 1], 533.086)
        self.assertEqual(window.loc[day2_start, 1], 533.087)
        self.assertEqual(window.loc[day2_end, 1], 533.853)
        self.assertEqual(window.loc[day3_start, 1], 533.854)

    def test_minute_window_starts_before_trading_start(self):
        portal = self.get_portal()

        # get a 50-bar minute history for MSFT starting 5 minutes into 3/20,
        # its first trading day
        window = portal.get_history_window(
            [2],
            pd.Timestamp("2014-03-20 13:35:00", tz='UTC'),
            50,
            "minute",
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
            "minute",
            "low",
        )

        self.assertEqual(len(window2), 50)
        reference2 = {
            1: [529.659, 527.957, 530.568, 531.849, 527.982],
            2: [98.902, 99.841, 90.984, 99.891, 98.027]
        }

        for i in range(0, 45):
            self.assertFalse(np.isnan(window2.iloc[i].loc[1]))

            # there should be 45 NaNs for MSFT until it starts trading
            self.assertTrue(np.isnan(window2.iloc[i].loc[2]))

        for i in range(0, 4):
            self.assertEqual(window2.iloc[-5 + i].loc[1], reference2[1][i])
            self.assertEqual(window2.iloc[-5 + i].loc[2], reference2[2][i])

    def test_minute_window_ends_before_trading_start(self):
        # entire window is before the trading start
        window = self.get_portal().get_history_window(
            [2],
            pd.Timestamp("2014-02-05 14:35:00", tz='UTC'),
            100,
            "minute",
            "high"
        )

        self.assertEqual(len(window), 100)
        for i in range(0, 100):
            self.assertTrue(np.isnan(window.iloc[i].loc[2]))

    def test_minute_window_ends_after_trading_end(self):
        portal = self.get_portal()

        # get a 50-bar minute history for MSFT starting 5 minutes into 3/20,
        # its first trading day
        window = portal.get_history_window(
            [2],
            pd.Timestamp("2014-03-24 13:35:00", tz='UTC'),
            50,
            "minute",
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
        window = self.get_portal().get_history_window(
            [2],
            pd.Timestamp("2014-04-02 14:35:00", tz='UTC'),
            100,
            "minute",
            "high"
        )

        self.assertEqual(len(window), 100)
        for i in range(0, 100):
            self.assertTrue(np.isnan(window.iloc[i].loc[2]))

    def test_minute_window_starts_before_1_2_2002(self):
        window = self.get_portal().get_history_window(
            [3],
            pd.Timestamp("2002-01-02 14:35:00", tz='UTC'),
            50,
            "minute",
            "close"
        )

        self.assertEqual(len(window), 50)
        for i in range(0, 45):
            self.assertTrue(np.isnan(window.iloc[i].loc[3]))

        for i in range(46, 50):
            self.assertFalse(np.isnan(window.iloc[i].loc[3]))

    def test_minute_window_ends_before_1_2_2002(self):
        with self.assertRaises(ValueError):
            self.get_portal().get_history_window(
                [3],
                pd.Timestamp("2001-12-31 14:35:00", tz='UTC'),
                50,
                "minute",
                "close"
            )

    def test_minute_forward_fill(self):
        # only forward fill if ffill=True AND we are asking for "price"

        # our fake TSLA data (sid 4) is missing a bunch of minute bars
        # right after the open on 2002-01-02

        for field in ["open", "high", "low", "volume", "close_price"]:
            no_ffill = self.get_portal().get_history_window(
                [4],
                pd.Timestamp("2002-01-02 21:00:00", tz='UTC'),
                390,
                "minute",
                field
            )

            missing_bar_indices = [1, 3, 5, 7, 9, 11, 13]
            if field == 'volume':
                for bar_idx in missing_bar_indices:
                    self.assertEqual(no_ffill.iloc[bar_idx].loc[4], 0)
            else:
                for bar_idx in missing_bar_indices:
                    self.assertTrue(np.isnan(no_ffill.iloc[bar_idx].loc[4]))

        ffill_window = self.get_portal().get_history_window(
            [4],
            pd.Timestamp("2002-01-02 21:00:00", tz='UTC'),
            390,
            "minute",
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
        really_no_ffill_window = self.get_portal().get_history_window(
            [4],
            pd.Timestamp("2002-01-02 21:00:00", tz='UTC'),
            390,
            "minute",
            "price",
            ffill=False
        )

        for idx, val in enumerate(vals):
            idx1 = 2 * idx
            idx2 = idx1 + 1
            self.assertEqual(really_no_ffill_window.iloc[idx1].loc[4], val)
            self.assertTrue(np.isnan(really_no_ffill_window.iloc[idx2].loc[4]))
