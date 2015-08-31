from os.path import dirname, join, realpath
from unittest import TestCase

from testfixtures import TempDirectory
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
from zipline.utils.test_utils import make_simple_asset_info
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
        cls.assets = [cls.AAPL]

        asset_info = make_simple_asset_info(
            cls.assets,
            Timestamp('2014'),
            Timestamp('2015'),
            ['AAPL']
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
            cls.AAPL: join(TEST_MINUTE_RESOURCE_PATH, 'AAPL_minute.csv')
        }
        raw_data = {
            asset: read_csv(path, parse_dates=['minute']).set_index('minute')
            for asset, path in iteritems(resources)
        }
        # Add 'price' column as an alias because all kinds of stuff in zipline
        # depends on it being present. :/
        for frame in raw_data.values():
            frame['price'] = frame['close']

        MinuteBarWriterFromCSVs(resources).write(tempdir.path, cls.assets)

    @classmethod
    def create_fake_daily_data(cls, tempdir):
        resources = {
            cls.AAPL: join(TEST_DAILY_RESOURCE_PATH, 'AAPL.csv')
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
        dbpath = tempdir.getpath('adjustments.sqlite')
        writer = SQLiteAdjustmentWriter(dbpath)

        splits = mergers = dividends = DataFrame(
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

    def test_basic_functionality(self):
        temp_path = self.tempdir.path

        portal = DataPortal(
            None,
            findata_dir=temp_path,
            asset_finder=self.asset_finder,
            daily_equities_path=join(temp_path, "test_daily_data.bcolz"),
            adjustments_path=join(temp_path, "adjustments.sqlite")
        )

        # get a 5-bar minute history
        window = portal.get_history_window(
            [1],
            pd.Timestamp("2014-03-21 18:30:00+00:00", tz='UTC'),
            5,
            "minute",
            "open"
        )

        self.assertEqual(len(window), 5)
        reference = [534.362, 534.363, 534.367, 534.367, 534.371]
        for i in range(0, 4):
            self.assertEqual(window.iloc[-5 + i].loc[1], reference[i])

    

