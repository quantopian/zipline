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
        cls.BRKA = 5
        cls.IBM = 6
        cls.assets = [cls.AAPL, cls.MSFT, cls.DELL, cls.TSLA, cls.BRKA,
                      cls.IBM]

        asset_info = make_simple_asset_info(
            cls.assets,
            Timestamp('2014-03-03'),
            Timestamp('2014-08-29'),
            ['AAPL', 'MSFT', 'DELL', 'TSLA', 'BRKA', 'IBM']
        )

        cls.asset_finder = AssetFinder(asset_info)

        cls.tempdir = TempDirectory()
        cls.tempdir.create()

        try:
            cls.create_fake_minute_data(cls.tempdir)
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
                ],
                columns=['effective_date', 'ratio', 'sid'],
            )

            cls.create_fake_adjustments(cls.tempdir,
                                        "adjustments.sqlite",
                                        splits)
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
            cls.TSLA: join(TEST_MINUTE_RESOURCE_PATH, "TSLA_minute.csv"),
            cls.BRKA: join(TEST_MINUTE_RESOURCE_PATH, "BRKA_minute.csv"),
            cls.IBM: join(TEST_MINUTE_RESOURCE_PATH, "IBM_minute.csv")
        }

        MinuteBarWriterFromCSVs(resources).write(tempdir.path, cls.assets)

    @classmethod
    def create_fake_daily_data(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_DAILY_RESOURCE_PATH, 'AAPL.csv'),
            cls.MSFT: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),
            cls.DELL: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),
            cls.TSLA: join(TEST_DAILY_RESOURCE_PATH, 'MSFT.csv'),
            cls.BRKA: join(TEST_DAILY_RESOURCE_PATH, 'BRK-A.csv'),
            cls.IBM: join(TEST_MINUTE_RESOURCE_PATH, 'IBM_daily.csv')
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
    def create_fake_adjustments(cls, tempdir, filename, splits):
        writer = SQLiteAdjustmentWriter(tempdir.getpath(filename))

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

        writer.write(splits, mergers, dividends)

    def get_portal(self,
                   daily_equities_filename="test_daily_data.bcolz",
                   adjustments_filename="adjustments.sqlite"):
        temp_path = self.tempdir.path

        return DataPortal(
            None,
            findata_dir=temp_path,
            asset_finder=self.asset_finder,
            daily_equities_path=join(temp_path, daily_equities_filename),
            adjustments_path=join(temp_path, adjustments_filename)
        )

    def test_minute_basic_functionality(self):
        # get a 5-bar minute history from the very end of the available data
        window = self.get_portal().get_history_window(
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

    def test_minute_early_close(self):
        # market was closed early on 7/3, and that's reflected in our
        # fake IBM minute data.  also, IBM had a split that takes effect
        # right after the early close.

        # five minutes into the day after an early close, get 20 1m bars
        window = self.get_portal().get_history_window(
            [self.IBM],
            pd.Timestamp("2014-07-07 13:35:00", tz='UTC'),
            20,
            "minute",
            "high"
        )

        self.assertEqual(len(window), 20)

        reference = [27134.486, 27134.802, 27134.660, 27132.813, 27130.964,
                     27133.767, 27133.268, 27131.510, 27134.946, 27132.400,
                     27134.350, 27130.588, 27132.528, 27130.418, 27131.040,
                     27132.664, 27131.307, 27133.978, 27132.779, 27134.476]

        for i in range(0, 20):
            self.assertEqual(window.iloc[i].loc[self.IBM], reference[i])

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
            window = self.get_portal().get_history_window(
                [self.BRKA],
                pd.Timestamp("2014-03-21 13:35", tz='UTC'),
                10,
                "daily",
                field
            )

            self.assertEqual(len(window), 10)

            for i in range(0, 10):
                self.assertEqual(window.iloc[i].loc[self.BRKA], values[i])

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

        run_query("open", opens)
        run_query("close", closes)
        run_query("close_price", closes)
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
                                    splits)

        portal = self.get_portal(adjustments_filename="adjustments2.sqlite")

        def test_window(field, reference, ffill=True):
            window = portal.get_history_window(
                [self.AAPL],
                pd.Timestamp("2014-06-11 15:30", tz='UTC'),
                6,
                "daily",
                field,
                ffill
            )

            self.assertEqual(len(window), 6)

            for i in range(0, 5):
                self.assertEqual(window.iloc[i].loc[self.AAPL], reference[i])

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
        test_window("open", open_data, ffill=False)
        test_window("open", open_data)

        high_data = [92.556, 92.767, 93.037, 93.879, 95.050]
        test_window("high", high_data, ffill=False)
        test_window("high", high_data)

        low_data = [90.873, 91.801, 92.067, 91.750, 93.570]
        test_window("low", low_data, ffill=False)
        test_window("low", low_data)

        close_data = [92.117, 92.478, 92.224, 93.699, 94.250]
        test_window("close", close_data, ffill=False)
        test_window("close", close_data)
        test_window("price", close_data, ffill=False)
        test_window("price", close_data)

        vol_data = [587093500, 531659800, 612392200, 75415000, 62777000]
        test_window("volume", vol_data)
        test_window("volume", vol_data, ffill=False)










