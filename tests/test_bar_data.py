from unittest import TestCase

from testfixtures import TempDirectory
import pandas as pd
import numpy as np

from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import BcolzMinuteBarWriter, \
    US_EQUITIES_MINUTES_PER_DAY, BcolzMinuteBarReader
from zipline.data.us_equity_pricing import BcolzDailyBarReader
from zipline.finance.trading import TradingEnvironment
from zipline.protocol import BarData
from zipline.utils.test_utils import write_minute_data_for_asset, \
    create_daily_df_for_asset, DailyBarWriterFromDataFrames

OHLC = ["open", "high", "low", "close"]
OHLCP = OHLC + ["price"]
ALL_FIELDS = OHLCP + ["volume", "last_traded"]

# offsets used in test data
field_info = {
    "open": 1,
    "high": 2,
    "low": -1,
    "close": 0
}


class TestBarDataBase(TestCase):
    def assert_same(self, val1, val2):
        try:
            self.assertEqual(val1, val2)
        except AssertionError:
            if val1 is pd.NaT:
                self.assertTrue(val2 is pd.NaT)
            elif np.isnan(val1):
                self.assertTrue(np.isnan(val2))
            else:
                raise

    def check_internal_consistency(self, bar_data):
        df = bar_data.spot_value([self.ASSET1, self.ASSET2], ALL_FIELDS)

        asset1_multi_field = bar_data.spot_value(self.ASSET1, ALL_FIELDS)
        asset2_multi_field = bar_data.spot_value(self.ASSET2, ALL_FIELDS)

        for field in ALL_FIELDS:
            asset1_value = bar_data.spot_value(self.ASSET1, field)
            asset2_value = bar_data.spot_value(self.ASSET2, field)

            multi_asset_series = bar_data.spot_value(
                [self.ASSET1, self.ASSET2], field
            )

            # make sure all the different query forms are internally
            # consistent
            self.assert_same(multi_asset_series[self.ASSET1], asset1_value)
            self.assert_same(multi_asset_series[self.ASSET2], asset2_value)

            self.assert_same(df.loc[self.ASSET1][field], asset1_value)
            self.assert_same(df.loc[self.ASSET2][field], asset2_value)

            self.assert_same(asset1_multi_field[field], asset1_value)
            self.assert_same(asset2_multi_field[field], asset2_value)


class TestMinuteBarData(TestBarDataBase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()

        # asset1 has trades every minute
        # asset2 has trades every 10 minutes

        cls.env = TradingEnvironment()

        cls.days = cls.env.days_in_range(
            start=pd.Timestamp("2016-01-05", tz='UTC'),
            end=pd.Timestamp("2016-01-07", tz='UTC')
        )

        cls.env.write_data(equities_data={
            sid: {
                'start_date': cls.days[0],
                'end_date': cls.days[-1],
                'symbol': "ASSET{0}".format(sid)
            } for sid in [1, 2]
        })

        cls.data_portal = DataPortal(
            cls.env,
            equity_minute_reader=cls.build_minute_data()
        )

        cls.ASSET1 = cls.env.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.env.asset_finder.retrieve_asset(2)

        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def build_minute_data(cls):
        market_opens = cls.env.open_and_closes.market_open.loc[cls.days]

        writer = BcolzMinuteBarWriter(
            cls.days[0],
            cls.tempdir.path,
            market_opens,
            US_EQUITIES_MINUTES_PER_DAY
        )

        write_minute_data_for_asset(
            cls.env,
            writer,
            cls.days[0],
            cls.days[-2],
            1
        )

        write_minute_data_for_asset(
            cls.env,
            writer,
            cls.days[0],
            cls.days[-2],
            2,
            10
        )

        return BcolzMinuteBarReader(cls.tempdir.path)

    def test_minute_before_assets_trading(self):
        # grab minutes that include the day before the asset start
        minutes = self.env.market_minutes_for_day(
            self.env.previous_trading_day(self.days[0])
        )

        # this entire day is before either asset has started trading
        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            self.check_internal_consistency(bar_data)

            self.assertFalse(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.can_trade(self.ASSET2))

            self.assertFalse(bar_data.is_stale(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET2))

            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.spot_value(asset, field)

                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == "volume":
                        self.assertEqual(0, asset_value)
                    elif field == "last_traded":
                        self.assertTrue(asset_value is pd.NaT)

    def test_regular_minute(self):
        minutes = self.env.market_minutes_for_day(self.days[0])

        for idx, minute in enumerate(minutes):
            # day2 has prices
            # (every minute for asset1, every 10 minutes for asset2)

            # asset1:
            # opens: 2-391
            # high: 3-392
            # low: 0-389
            # close: 1-390
            # volume: 100-3900 (by 100)

            # asset2 is the same thing, but with only every 10th minute
            # populated.

            # this test covers the "IPO morning" case, because asset2 only
            # has data starting on the 10th minute.

            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            self.check_internal_consistency(bar_data)
            asset2_has_data = (((idx + 1) % 10) == 0)

            self.assertTrue(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET1))

            if idx < 9:
                self.assertFalse(bar_data.can_trade(self.ASSET2))
                self.assertFalse(bar_data.is_stale(self.ASSET2))
            else:
                self.assertTrue(bar_data.can_trade(self.ASSET2))

                if asset2_has_data:
                    self.assertFalse(bar_data.is_stale(self.ASSET2))
                else:
                    self.assertTrue(bar_data.is_stale(self.ASSET2))

            for field in ALL_FIELDS:
                asset1_value = bar_data.spot_value(self.ASSET1, field)
                asset2_value = bar_data.spot_value(self.ASSET2, field)

                # now check the actual values
                if idx == 0 and field == "low":
                    # first low value is 0, which is interpreted as NaN
                    self.assertTrue(np.isnan(asset1_value))
                else:
                    if field in OHLC:
                        self.assertEqual(
                            idx + 1 + field_info[field],
                            asset1_value
                        )

                        if asset2_has_data:
                            self.assertEqual(
                                idx + 1 + field_info[field],
                                asset2_value
                            )
                        else:
                            self.assertTrue(np.isnan(asset2_value))
                    elif field == "volume":
                        self.assertEqual((idx + 1) * 100, asset1_value)

                        if asset2_has_data:
                            self.assertEqual((idx + 1) * 100, asset2_value)
                        else:
                            self.assertEqual(0, asset2_value)
                    elif field == "price":
                        self.assertEqual(idx + 1, asset1_value)

                        if asset2_has_data:
                            self.assertEqual(idx + 1, asset2_value)
                        elif idx < 9:
                            # no price to forward fill from
                            self.assertTrue(np.isnan(asset2_value))
                        else:
                            # forward-filled price
                            self.assertEqual((idx // 10) * 10, asset2_value)
                    elif field == "last_traded":
                        self.assertEqual(minute, asset1_value)

                        if idx < 9:
                            self.assertTrue(asset2_value is pd.NaT)
                        elif asset2_has_data:
                            self.assertEqual(minute, asset2_value)
                        else:
                            last_traded_minute = minutes[(idx // 10) * 10]
                            self.assertEqual(last_traded_minute - 1,
                                             asset2_value)

    def test_minute_after_assets_stopped(self):
        minutes = self.env.market_minutes_for_day(self.days[-1])

        last_trading_minute = \
            self.env.market_minutes_for_day(self.days[-2])[-1]

        # this entire day is after both assets have stopped trading
        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")

            self.assertFalse(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.can_trade(self.ASSET2))

            self.assertFalse(bar_data.is_stale(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET2))

            self.check_internal_consistency(bar_data)

            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.spot_value(asset, field)

                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == "volume":
                        self.assertEqual(0, asset_value)
                    elif field == "last_traded":
                        self.assertEqual(last_trading_minute, asset_value)


class TestDailyBarData(TestBarDataBase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()

        # asset1 has a daily data for each day (1/5, 1/6, 1/7)
        # asset2 only has daily data for day2 (1/6)

        cls.env = TradingEnvironment()

        cls.days = cls.env.days_in_range(
            start=pd.Timestamp("2016-01-05", tz='UTC'),
            end=pd.Timestamp("2016-01-08", tz='UTC')
        )

        cls.env.write_data(equities_data={
            sid: {
                'start_date': cls.days[0],
                'end_date': cls.days[-1],
                'symbol': "ASSET{0}".format(sid)
            } for sid in [1, 2]
        })

        cls.data_portal = DataPortal(
            cls.env,
            equity_daily_reader=cls.build_daily_data()
        )

        cls.ASSET1 = cls.env.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.env.asset_finder.retrieve_asset(2)

        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def build_daily_data(cls):
        path = cls.tempdir.getpath("testdaily.bcolz")

        dfs = {
            1: create_daily_df_for_asset(cls.env, cls.days[0], cls.days[-2]),
            2: create_daily_df_for_asset(
                cls.env, cls.days[0], cls.days[-2], interval=2
            )
        }

        daily_writer = DailyBarWriterFromDataFrames(dfs)
        daily_writer.write(path, cls.days[:-1], dfs)

        return BcolzDailyBarReader(path)

    def test_day_before_assets_trading(self):
        # use the day before self.days[0]
        day = self.env.previous_trading_day(self.days[0])

        bar_data = BarData(self.data_portal, lambda: day, "daily")
        self.check_internal_consistency(bar_data)

        self.assertFalse(bar_data.can_trade(self.ASSET1))
        self.assertFalse(bar_data.can_trade(self.ASSET2))

        self.assertFalse(bar_data.is_stale(self.ASSET1))
        self.assertFalse(bar_data.is_stale(self.ASSET2))

        for field in ALL_FIELDS:
            for asset in self.ASSETS:
                asset_value = bar_data.spot_value(asset, field)

                if field in OHLCP:
                    self.assertTrue(np.isnan(asset_value))
                elif field == "volume":
                    self.assertEqual(0, asset_value)
                elif field == "last_traded":
                    self.assertTrue(asset_value is pd.NaT)

    def test_semi_active_day(self):
        # on self.days[0], only asset1 has data
        bar_data = BarData(self.data_portal, lambda: self.days[0], "daily")
        self.check_internal_consistency(bar_data)

        self.assertTrue(bar_data.can_trade(self.ASSET1))
        self.assertFalse(bar_data.can_trade(self.ASSET2))

        # because there is real data
        self.assertFalse(bar_data.is_stale(self.ASSET1))

        # because there has never been a trade bar yet
        self.assertFalse(bar_data.is_stale(self.ASSET2))

        self.assertEqual(3, bar_data.spot_value(self.ASSET1, "open"))
        self.assertEqual(4, bar_data.spot_value(self.ASSET1, "high"))
        self.assertEqual(1, bar_data.spot_value(self.ASSET1, "low"))
        self.assertEqual(2, bar_data.spot_value(self.ASSET1, "close"))
        self.assertEqual(200, bar_data.spot_value(self.ASSET1, "volume"))
        self.assertEqual(2, bar_data.spot_value(self.ASSET1, "price"))
        self.assertEqual(self.days[0],
                         bar_data.spot_value(self.ASSET1, "last_traded"))

        for field in OHLCP:
            self.assertTrue(np.isnan(bar_data.spot_value(self.ASSET2, field)))

        self.assertEqual(0, bar_data.spot_value(self.ASSET2, "volume"))
        self.assertTrue(
            bar_data.spot_value(self.ASSET2, "last_traded") is pd.NaT
        )

    def test_fully_active_day(self):
        bar_data = BarData(self.data_portal, lambda: self.days[1], "daily")
        self.check_internal_consistency(bar_data)

        # on self.days[1], both assets have data
        for asset in self.ASSETS:
            self.assertTrue(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))

            self.assertEqual(4, bar_data.spot_value(asset, "open"))
            self.assertEqual(5, bar_data.spot_value(asset, "high"))
            self.assertEqual(2, bar_data.spot_value(asset, "low"))
            self.assertEqual(3, bar_data.spot_value(asset, "close"))
            self.assertEqual(300, bar_data.spot_value(asset, "volume"))
            self.assertEqual(3, bar_data.spot_value(asset, "price"))
            self.assertEqual(
                self.days[1],
                bar_data.spot_value(asset, "last_traded")
            )

    def test_after_assets_dead(self):
        # both assets are dead by self.days[-1]
        bar_data = BarData(self.data_portal, lambda: self.days[-1], "daily")
        self.check_internal_consistency(bar_data)

        for asset in self.ASSETS:
            self.assertFalse(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))

            for field in OHLCP:
                self.assertTrue(np.isnan(bar_data.spot_value(asset, field)))

            self.assertEqual(0, bar_data.spot_value(asset, "volume"))

            last_traded_dt = bar_data.spot_value(asset, "last_traded")

            if asset == self.ASSET1:
                self.assertEqual(self.days[-2], last_traded_dt)
            else:
                self.assertEqual(self.days[1], last_traded_dt)
