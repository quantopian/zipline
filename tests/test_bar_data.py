#
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
from unittest import TestCase

from testfixtures import TempDirectory
import pandas as pd
import numpy as np
from nose_parameterized import parameterized

from zipline._protocol import handle_non_market_minutes
from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import BcolzMinuteBarWriter, \
    US_EQUITIES_MINUTES_PER_DAY, BcolzMinuteBarReader
from zipline.data.us_equity_pricing import BcolzDailyBarReader, \
    SQLiteAdjustmentReader, SQLiteAdjustmentWriter
from zipline.finance.trading import TradingEnvironment
from zipline.protocol import BarData
from zipline.testing.core import write_minute_data_for_asset, \
    create_daily_df_for_asset, DailyBarWriterFromDataFrames, \
    create_mock_adjustments, str_to_seconds, MockDailyBarReader

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
        df = bar_data.current([self.ASSET1, self.ASSET2], ALL_FIELDS)

        asset1_multi_field = bar_data.current(self.ASSET1, ALL_FIELDS)
        asset2_multi_field = bar_data.current(self.ASSET2, ALL_FIELDS)

        for field in ALL_FIELDS:
            asset1_value = bar_data.current(self.ASSET1, field)
            asset2_value = bar_data.current(self.ASSET2, field)

            multi_asset_series = bar_data.current(
                [self.ASSET1, self.ASSET2], field
            )

            # make sure all the different query forms are internally
            # consistent
            self.assert_same(multi_asset_series.loc[self.ASSET1], asset1_value)
            self.assert_same(multi_asset_series.loc[self.ASSET2], asset2_value)

            self.assert_same(df.loc[self.ASSET1][field], asset1_value)
            self.assert_same(df.loc[self.ASSET2][field], asset2_value)

            self.assert_same(asset1_multi_field[field], asset1_value)
            self.assert_same(asset2_multi_field[field], asset2_value)

        # also verify that bar_data doesn't expose anything bad
        for field in ["data_portal", "simulation_dt_func", "data_frequency",
                      "_views", "_universe_func", "_last_calculated_universe",
                      "_universe_last_updatedat"]:
            with self.assertRaises(AttributeError):
                getattr(bar_data, field)


class TestMinuteBarData(TestBarDataBase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()

        # asset1 has trades every minute
        # asset2 has trades every 10 minutes
        # split_asset trades every minute
        # illiquid_split_asset trades every 10 minutes

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
            } for sid in [1, 2, 3, 4, 5]
        })

        cls.ASSET1 = cls.env.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.env.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.env.asset_finder.retrieve_asset(3)
        cls.ILLIQUID_SPLIT_ASSET = cls.env.asset_finder.retrieve_asset(4)
        cls.HILARIOUSLY_ILLIQUID_ASSET = cls.env.asset_finder.retrieve_asset(5)

        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

        cls.adjustments_reader = cls.create_adjustments_reader()
        cls.data_portal = DataPortal(
            cls.env,
            equity_minute_reader=cls.build_minute_data(),
            adjustment_reader=cls.adjustments_reader
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def create_adjustments_reader(cls):
        path = create_mock_adjustments(
            cls.tempdir,
            cls.days,
            splits=[{
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET.sid
            }, {
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.ILLIQUID_SPLIT_ASSET.sid
            }]
        )

        return SQLiteAdjustmentReader(path)

    @classmethod
    def build_minute_data(cls):
        market_opens = cls.env.open_and_closes.market_open.loc[cls.days]
        market_closes = cls.env.open_and_closes.market_close.loc[cls.days]

        writer = BcolzMinuteBarWriter(
            cls.days[0],
            cls.tempdir.path,
            market_opens,
            market_closes,
            US_EQUITIES_MINUTES_PER_DAY
        )

        for sid in [cls.ASSET1.sid, cls.SPLIT_ASSET.sid]:
            write_minute_data_for_asset(
                cls.env,
                writer,
                cls.days[0],
                cls.days[-1],
                sid
            )

        for sid in [cls.ASSET2.sid, cls.ILLIQUID_SPLIT_ASSET.sid]:
            write_minute_data_for_asset(
                cls.env,
                writer,
                cls.days[0],
                cls.days[-1],
                sid,
                10
            )

        write_minute_data_for_asset(
            cls.env,
            writer,
            cls.days[0],
            cls.days[-1],
            cls.HILARIOUSLY_ILLIQUID_ASSET.sid,
            50
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
                    asset_value = bar_data.current(asset, field)

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
                asset1_value = bar_data.current(self.ASSET1, field)
                asset2_value = bar_data.current(self.ASSET2, field)

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

    def test_minute_of_last_day(self):
        minutes = self.env.market_minutes_for_day(self.days[-1])

        # this is the last day the assets exist
        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")

            self.assertTrue(bar_data.can_trade(self.ASSET1))
            self.assertTrue(bar_data.can_trade(self.ASSET2))

    def test_minute_after_assets_stopped(self):
        minutes = self.env.market_minutes_for_day(
            self.env.next_trading_day(self.days[-1])
        )

        last_trading_minute = \
            self.env.market_minutes_for_day(self.days[-1])[-1]

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
                    asset_value = bar_data.current(asset, field)

                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == "volume":
                        self.assertEqual(0, asset_value)
                    elif field == "last_traded":
                        self.assertEqual(last_trading_minute, asset_value)

    def test_spot_price_is_unadjusted(self):
        # verify there is a split for SPLIT_ASSET
        splits = self.adjustments_reader.get_adjustments_for_sid(
            "splits",
            self.SPLIT_ASSET.sid
        )

        self.assertEqual(1, len(splits))
        split = splits[0]
        self.assertEqual(
            split[0],
            pd.Timestamp("2016-01-06", tz='UTC')
        )

        # ... but that's it's not applied when using spot value
        minutes = self.env.minutes_for_days_in_range(
            start=self.days[0], end=self.days[1]
        )

        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            self.assertEqual(
                idx + 1,
                bar_data.current(self.SPLIT_ASSET, "price")
            )

    def test_spot_price_is_adjusted_if_needed(self):
        # on cls.days[1], the first 9 minutes of ILLIQUID_SPLIT_ASSET are
        # missing. let's get them.
        day0_minutes = self.env.market_minutes_for_day(self.days[0])
        day1_minutes = self.env.market_minutes_for_day(self.days[1])

        for idx, minute in enumerate(day0_minutes[-10:-1]):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            self.assertEqual(
                380,
                bar_data.current(self.ILLIQUID_SPLIT_ASSET, "price")
            )

        bar_data = BarData(
            self.data_portal, lambda: day0_minutes[-1], "minute"
        )

        self.assertEqual(
            390,
            bar_data.current(self.ILLIQUID_SPLIT_ASSET, "price")
        )

        for idx, minute in enumerate(day1_minutes[0:9]):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")

            # should be half of 390, due to the split
            self.assertEqual(
                195,
                bar_data.current(self.ILLIQUID_SPLIT_ASSET, "price")
            )

    def test_spot_price_at_midnight(self):
        # make sure that if we try to get a minute price at a non-market
        # minute, we use the previous market close's timestamp
        day = self.days[1]

        eight_fortyfive_am_eastern = \
            pd.Timestamp("{0}-{1}-{2} 8:45".format(
                day.year, day.month, day.day),
                tz='US/Eastern'
            )

        bar_data = BarData(self.data_portal, lambda: day, "minute")
        bar_data2 = BarData(self.data_portal,
                            lambda: eight_fortyfive_am_eastern,
                            "minute")

        with handle_non_market_minutes(bar_data), \
                handle_non_market_minutes(bar_data2):
            for bd in [bar_data, bar_data2]:
                for field in ["close", "price"]:
                    self.assertEqual(
                        390,
                        bd.current(self.ASSET1, field)
                    )

                # make sure that if the asset didn't trade at the previous
                # close, we properly ffill (or not ffill)
                self.assertEqual(
                    350,
                    bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, "price")
                )

                self.assertTrue(
                    np.isnan(bd.current(self.HILARIOUSLY_ILLIQUID_ASSET,
                                        "high"))
                )

                self.assertEqual(
                    0,
                    bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, "volume")
                )

    def test_can_trade_at_midnight(self):
        # make sure that if we use `can_trade` at midnight, we don't pretend
        # we're in the previous day's last minute
        the_day_after = self.env.next_trading_day(self.days[-1])

        bar_data = BarData(self.data_portal, lambda: the_day_after, "minute")

        for asset in [self.ASSET1, self.HILARIOUSLY_ILLIQUID_ASSET]:
            self.assertFalse(bar_data.can_trade(asset))

            with handle_non_market_minutes(bar_data):
                self.assertFalse(bar_data.can_trade(asset))

        # but make sure it works when the assets are alive
        bar_data2 = BarData(self.data_portal, lambda: self.days[1], "minute")
        for asset in [self.ASSET1, self.HILARIOUSLY_ILLIQUID_ASSET]:
            self.assertTrue(bar_data2.can_trade(asset))

            with handle_non_market_minutes(bar_data2):
                self.assertTrue(bar_data2.can_trade(asset))

    def test_is_stale_at_midnight(self):
        bar_data = BarData(self.data_portal, lambda: self.days[1], "minute")

        with handle_non_market_minutes(bar_data):
            self.assertTrue(bar_data.is_stale(self.HILARIOUSLY_ILLIQUID_ASSET))

    def test_overnight_adjustments(self):
        # verify there is a split for SPLIT_ASSET
        splits = self.adjustments_reader.get_adjustments_for_sid(
            "splits",
            self.SPLIT_ASSET.sid
        )

        self.assertEqual(1, len(splits))
        split = splits[0]
        self.assertEqual(
            split[0],
            pd.Timestamp("2016-01-06", tz='UTC')
        )

        # Current day is 1/06/16
        day = self.days[1]
        eight_fortyfive_am_eastern = \
            pd.Timestamp("{0}-{1}-{2} 8:45".format(
                day.year, day.month, day.day),
                tz='US/Eastern'
            )

        bar_data = BarData(self.data_portal,
                           lambda: eight_fortyfive_am_eastern,
                           "minute")

        expected = {
            'open': 391 / 2.0,
            'high': 392 / 2.0,
            'low': 389 / 2.0,
            'close': 390 / 2.0,
            'volume': 39000 * 2.0,
            'price': 390 / 2.0,
        }

        with handle_non_market_minutes(bar_data):
            for field in OHLCP + ['volume']:
                value = bar_data.current(self.SPLIT_ASSET, field)

                # Assert the price is adjusted for the overnight split
                self.assertEqual(value, expected[field])


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
            } for sid in [1, 2, 3, 4, 5, 6, 7, 8]
        })

        cls.ASSET1 = cls.env.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.env.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.env.asset_finder.retrieve_asset(3)
        cls.ILLIQUID_SPLIT_ASSET = cls.env.asset_finder.retrieve_asset(4)
        cls.MERGER_ASSET = cls.env.asset_finder.retrieve_asset(5)
        cls.ILLIQUID_MERGER_ASSET = cls.env.asset_finder.retrieve_asset(6)
        cls.DIVIDEND_ASSET = cls.env.asset_finder.retrieve_asset(7)
        cls.ILLIQUID_DIVIDEND_ASSET = cls.env.asset_finder.retrieve_asset(8)
        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

        cls.adjustments_reader = cls.create_adjustments_reader()
        cls.data_portal = DataPortal(
            cls.env,
            equity_daily_reader=cls.build_daily_data(),
            adjustment_reader=cls.adjustments_reader
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def create_adjustments_reader(cls):
        path = cls.tempdir.getpath("test_adjustments.db")

        adj_writer = SQLiteAdjustmentWriter(
            path,
            cls.env.trading_days,
            MockDailyBarReader()
        )

        splits = pd.DataFrame([
            {
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET.sid
            },
            {
                'effective_date': str_to_seconds("2016-01-07"),
                'ratio': 0.5,
                'sid': cls.ILLIQUID_SPLIT_ASSET.sid
            }
        ])

        mergers = pd.DataFrame([
            {
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.MERGER_ASSET.sid
            },
            {
                'effective_date': str_to_seconds("2016-01-07"),
                'ratio': 0.6,
                'sid': cls.ILLIQUID_MERGER_ASSET.sid
            }
        ])

        # we're using a fake daily reader in the adjustments writer which
        # returns every daily price as 100, so dividend amounts of 2.0 and 4.0
        # correspond to 2% and 4% dividends, respectively.
        dividends = pd.DataFrame([
            {
                # only care about ex date, the other dates don't matter here
                'ex_date':
                    pd.Timestamp("2016-01-06", tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp("2016-01-06", tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp("2016-01-06", tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp("2016-01-06", tz='UTC').to_datetime64(),
                'amount': 2.0,
                'sid': cls.DIVIDEND_ASSET.sid
            },
            {
                'ex_date':
                    pd.Timestamp("2016-01-07", tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp("2016-01-07", tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp("2016-01-07", tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp("2016-01-07", tz='UTC').to_datetime64(),
                'amount': 4.0,
                'sid': cls.ILLIQUID_DIVIDEND_ASSET.sid
            }],
            columns=['ex_date',
                     'record_date',
                     'declared_date',
                     'pay_date',
                     'amount',
                     'sid']
        )

        adj_writer.write(splits, mergers, dividends)

        return SQLiteAdjustmentReader(path)

    @classmethod
    def build_daily_data(cls):
        path = cls.tempdir.getpath("testdaily.bcolz")

        dfs = {
            1: create_daily_df_for_asset(cls.env, cls.days[0], cls.days[-1]),
            2: create_daily_df_for_asset(
                cls.env, cls.days[0], cls.days[-1], interval=2
            ),
            3: create_daily_df_for_asset(cls.env, cls.days[0], cls.days[-1]),
            4: create_daily_df_for_asset(
                cls.env, cls.days[0], cls.days[-1], interval=2
            ),
            5: create_daily_df_for_asset(cls.env, cls.days[0], cls.days[-1]),
            6: create_daily_df_for_asset(
                cls.env, cls.days[0], cls.days[-1], interval=2
            ),
            7: create_daily_df_for_asset(cls.env, cls.days[0], cls.days[-1]),
            8: create_daily_df_for_asset(
                cls.env, cls.days[0], cls.days[-1], interval=2
            ),
        }

        daily_writer = DailyBarWriterFromDataFrames(dfs)
        daily_writer.write(path, cls.days, dfs)

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
                asset_value = bar_data.current(asset, field)

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

        self.assertEqual(3, bar_data.current(self.ASSET1, "open"))
        self.assertEqual(4, bar_data.current(self.ASSET1, "high"))
        self.assertEqual(1, bar_data.current(self.ASSET1, "low"))
        self.assertEqual(2, bar_data.current(self.ASSET1, "close"))
        self.assertEqual(200, bar_data.current(self.ASSET1, "volume"))
        self.assertEqual(2, bar_data.current(self.ASSET1, "price"))
        self.assertEqual(self.days[0],
                         bar_data.current(self.ASSET1, "last_traded"))

        for field in OHLCP:
            self.assertTrue(np.isnan(bar_data.current(self.ASSET2, field)),
                            field)

        self.assertEqual(0, bar_data.current(self.ASSET2, "volume"))
        self.assertTrue(
            bar_data.current(self.ASSET2, "last_traded") is pd.NaT
        )

    def test_fully_active_day(self):
        bar_data = BarData(self.data_portal, lambda: self.days[1], "daily")
        self.check_internal_consistency(bar_data)

        # on self.days[1], both assets have data
        for asset in self.ASSETS:
            self.assertTrue(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))

            self.assertEqual(4, bar_data.current(asset, "open"))
            self.assertEqual(5, bar_data.current(asset, "high"))
            self.assertEqual(2, bar_data.current(asset, "low"))
            self.assertEqual(3, bar_data.current(asset, "close"))
            self.assertEqual(300, bar_data.current(asset, "volume"))
            self.assertEqual(3, bar_data.current(asset, "price"))
            self.assertEqual(
                self.days[1],
                bar_data.current(asset, "last_traded")
            )

    def test_last_active_day(self):
        bar_data = BarData(self.data_portal, lambda: self.days[-1], "daily")
        self.check_internal_consistency(bar_data)

        for asset in self.ASSETS:
            self.assertTrue(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))

            self.assertEqual(6, bar_data.current(asset, "open"))
            self.assertEqual(7, bar_data.current(asset, "high"))
            self.assertEqual(4, bar_data.current(asset, "low"))
            self.assertEqual(5, bar_data.current(asset, "close"))
            self.assertEqual(500, bar_data.current(asset, "volume"))
            self.assertEqual(5, bar_data.current(asset, "price"))

    def test_after_assets_dead(self):
        # both assets end on self.day[-1], so let's try the next day
        next_day = self.env.next_trading_day(self.days[-1])

        bar_data = BarData(self.data_portal, lambda: next_day, "daily")
        self.check_internal_consistency(bar_data)

        for asset in self.ASSETS:
            self.assertFalse(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))

            for field in OHLCP:
                self.assertTrue(np.isnan(bar_data.current(asset, field)))

            self.assertEqual(0, bar_data.current(asset, "volume"))

            last_traded_dt = bar_data.current(asset, "last_traded")

            if asset == self.ASSET1:
                self.assertEqual(self.days[-2], last_traded_dt)
            else:
                self.assertEqual(self.days[1], last_traded_dt)

    @parameterized.expand([
        ("split", 2, 3, 3, 1.5),
        ("merger", 2, 3, 3, 1.8),
        ("dividend", 2, 3, 3, 2.88)
    ])
    def test_spot_price_adjustments(self,
                                    adjustment_type,
                                    liquid_day_0_price,
                                    liquid_day_1_price,
                                    illiquid_day_0_price,
                                    illiquid_day_1_price_adjusted):
        """Test the behaviour of spot prices during adjustments."""
        table_name = adjustment_type + 's'
        liquid_asset = getattr(self, (adjustment_type.upper() + "_ASSET"))
        illiquid_asset = getattr(
            self,
            ("ILLIQUID_" + adjustment_type.upper() + "_ASSET")
        )
        # verify there is an adjustment for liquid_asset
        adjustments = self.adjustments_reader.get_adjustments_for_sid(
            table_name,
            liquid_asset.sid
        )

        self.assertEqual(1, len(adjustments))
        adjustment = adjustments[0]
        self.assertEqual(
            adjustment[0],
            pd.Timestamp("2016-01-06", tz='UTC')
        )

        # ... but that's it's not applied when using spot value
        bar_data = BarData(self.data_portal, lambda: self.days[0], "daily")
        self.assertEqual(
            liquid_day_0_price,
            bar_data.current(liquid_asset, "price")
        )
        bar_data = BarData(self.data_portal, lambda: self.days[1], "daily")
        self.assertEqual(
            liquid_day_1_price,
            bar_data.current(liquid_asset, "price")
        )

        # ... except when we have to forward fill across a day boundary
        # ILLIQUID_ASSET has no data on days 0 and 2, and a split on day 2
        bar_data = BarData(self.data_portal, lambda: self.days[1], "daily")
        self.assertEqual(
            illiquid_day_0_price, bar_data.current(illiquid_asset, "price")
        )

        bar_data = BarData(self.data_portal, lambda: self.days[2], "daily")

        # 3 (price from previous day) * 0.5 (split ratio)
        self.assertAlmostEqual(
            illiquid_day_1_price_adjusted,
            bar_data.current(illiquid_asset, "price")
        )
