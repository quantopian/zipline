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
from nose_parameterized import parameterized
import numpy as np
import pandas as pd
from toolz import merge

from zipline._protocol import handle_non_market_minutes
from zipline.protocol import BarData
from zipline.testing import (
    MockDailyBarReader,
    create_daily_df_for_asset,
    create_minute_df_for_asset,
    str_to_seconds,
)
from zipline.testing.fixtures import (
    WithDataPortal,
    ZiplineTestCase,
)

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


class WithBarDataChecks(object):
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


class TestMinuteBarData(WithBarDataChecks,
                        WithDataPortal,
                        ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp(
        '2016-01-07',
        tz='UTC',
    )

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3, 4, 5

    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    HILARIOUSLY_ILLIQUID_ASSET_SID = 5

    @classmethod
    def make_minute_bar_data(cls):
        # asset1 has trades every minute
        # asset2 has trades every 10 minutes
        # split_asset trades every minute
        # illiquid_split_asset trades every 10 minutes
        return merge(
            {
                sid: create_minute_df_for_asset(
                    cls.env,
                    cls.bcolz_minute_bar_days[0],
                    cls.bcolz_minute_bar_days[-1],
                )
                for sid in (1, cls.SPLIT_ASSET_SID)
            },
            {
                sid: create_minute_df_for_asset(
                    cls.env,
                    cls.bcolz_minute_bar_days[0],
                    cls.bcolz_minute_bar_days[-1],
                    10,
                )
                for sid in (2, cls.ILLIQUID_SPLIT_ASSET_SID)
            },
            {
                cls.HILARIOUSLY_ILLIQUID_ASSET_SID: create_minute_df_for_asset(
                    cls.env,
                    cls.bcolz_minute_bar_days[0],
                    cls.bcolz_minute_bar_days[-1],
                    50,
                )
            },
        )

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame([
            {
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET_SID,
            },
            {
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.ILLIQUID_SPLIT_ASSET_SID,
            },
        ])

    @classmethod
    def init_class_fixtures(cls):
        super(TestMinuteBarData, cls).init_class_fixtures()

        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(
            cls.SPLIT_ASSET_SID,
        )
        cls.ILLIQUID_SPLIT_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_SPLIT_ASSET_SID,
        )
        cls.HILARIOUSLY_ILLIQUID_ASSET = cls.asset_finder.retrieve_asset(
            cls.HILARIOUSLY_ILLIQUID_ASSET_SID,
        )

        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    def test_minute_before_assets_trading(self):
        # grab minutes that include the day before the asset start
        minutes = self.env.market_minutes_for_day(
            self.env.previous_trading_day(self.bcolz_minute_bar_days[0])
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
        minutes = self.env.market_minutes_for_day(
            self.bcolz_minute_bar_days[0],
        )

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
        minutes = self.env.market_minutes_for_day(
            self.bcolz_daily_bar_days[-1],
        )

        # this is the last day the assets exist
        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")

            self.assertTrue(bar_data.can_trade(self.ASSET1))
            self.assertTrue(bar_data.can_trade(self.ASSET2))

    def test_minute_after_assets_stopped(self):
        minutes = self.env.market_minutes_for_day(
            self.env.next_trading_day(self.bcolz_minute_bar_days[-1])
        )

        last_trading_minute = \
            self.env.market_minutes_for_day(self.bcolz_minute_bar_days[-1])[-1]

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
        splits = self.adjustment_reader.get_adjustments_for_sid(
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
            start=self.bcolz_minute_bar_days[0],
            end=self.bcolz_minute_bar_days[1],
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
        day0_minutes = self.env.market_minutes_for_day(
            self.bcolz_minute_bar_days[0],
        )
        day1_minutes = self.env.market_minutes_for_day(
            self.bcolz_minute_bar_days[1],
        )

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
        day = self.bcolz_minute_bar_days[1]

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
        the_day_after = self.env.next_trading_day(
            self.bcolz_minute_bar_days[-1],
        )

        bar_data = BarData(self.data_portal, lambda: the_day_after, "minute")

        for asset in [self.ASSET1, self.HILARIOUSLY_ILLIQUID_ASSET]:
            self.assertFalse(bar_data.can_trade(asset))

            with handle_non_market_minutes(bar_data):
                self.assertFalse(bar_data.can_trade(asset))

        # but make sure it works when the assets are alive
        bar_data2 = BarData(
            self.data_portal,
            lambda: self.bcolz_minute_bar_days[1],
            "minute",
        )
        for asset in [self.ASSET1, self.HILARIOUSLY_ILLIQUID_ASSET]:
            self.assertTrue(bar_data2.can_trade(asset))

            with handle_non_market_minutes(bar_data2):
                self.assertTrue(bar_data2.can_trade(asset))

    def test_is_stale_at_midnight(self):
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_minute_bar_days[1],
            "minute",
        )

        with handle_non_market_minutes(bar_data):
            self.assertTrue(bar_data.is_stale(self.HILARIOUSLY_ILLIQUID_ASSET))

    def test_overnight_adjustments(self):
        # verify there is a split for SPLIT_ASSET
        splits = self.adjustment_reader.get_adjustments_for_sid(
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
        day = self.bcolz_daily_bar_days[1]
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


class TestDailyBarData(WithBarDataChecks,
                       WithDataPortal,
                       ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp(
        '2016-01-08',
        tz='UTC',
    )

    sids = ASSET_FINDER_EQUITY_SIDS = set(range(1, 9))

    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    MERGER_ASSET_SID = 5
    ILLIQUID_MERGER_ASSET_SID = 6
    DIVIDEND_ASSET_SID = 7
    ILLIQUID_DIVIDEND_ASSET_SID = 8

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame.from_records([
            {
                'effective_date': str_to_seconds("2016-01-06"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET_SID,
            },
            {
                'effective_date': str_to_seconds("2016-01-07"),
                'ratio': 0.5,
                'sid': cls.ILLIQUID_SPLIT_ASSET_SID,
            },
        ])

    @classmethod
    def make_mergers_data(cls):
        return pd.DataFrame.from_records([
            {
                'effective_date': str_to_seconds('2016-01-06'),
                'ratio': 0.5,
                'sid': cls.MERGER_ASSET_SID,
            },
            {
                'effective_date': str_to_seconds('2016-01-07'),
                'ratio': 0.6,
                'sid': cls.ILLIQUID_MERGER_ASSET_SID,
            }
        ])

    @classmethod
    def make_dividends_data(cls):
        return pd.DataFrame.from_records([
            {
                # only care about ex date, the other dates don't matter here
                'ex_date':
                    pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(),
                'amount': 2.0,
                'sid': cls.DIVIDEND_ASSET_SID,
            },
            {
                'ex_date':
                    pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(),
                'amount': 4.0,
                'sid': cls.ILLIQUID_DIVIDEND_ASSET_SID,
            }],
            columns=[
                'ex_date',
                'record_date',
                'declared_date',
                'pay_date',
                'amount',
                'sid',
            ]
        )

    @classmethod
    def make_adjustment_writer_daily_bar_reader(cls):
        return MockDailyBarReader()

    @classmethod
    def make_daily_bar_data(cls):
        for sid in cls.sids:
            yield sid, create_daily_df_for_asset(
                cls.env,
                cls.bcolz_daily_bar_days[0],
                cls.bcolz_daily_bar_days[-1],
                interval=2 - sid % 2
            )

    @classmethod
    def init_class_fixtures(cls):
        super(TestDailyBarData, cls).init_class_fixtures()

        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(
            cls.SPLIT_ASSET_SID,
        )
        cls.ILLIQUID_SPLIT_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_SPLIT_ASSET_SID,
        )
        cls.MERGER_ASSET = cls.asset_finder.retrieve_asset(
            cls.MERGER_ASSET_SID,
        )
        cls.ILLIQUID_MERGER_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_MERGER_ASSET_SID,
        )
        cls.DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(
            cls.DIVIDEND_ASSET_SID,
        )
        cls.ILLIQUID_DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(
            cls.ILLIQUID_DIVIDEND_ASSET_SID,
        )
        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    def test_day_before_assets_trading(self):
        # use the day before self.bcolz_daily_bar_days[0]
        day = self.env.previous_trading_day(self.bcolz_daily_bar_days[0])

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
        # on self.bcolz_daily_bar_days[0], only asset1 has data
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[0],
            "daily",
        )
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
        self.assertEqual(self.bcolz_daily_bar_days[0],
                         bar_data.current(self.ASSET1, "last_traded"))

        for field in OHLCP:
            self.assertTrue(np.isnan(bar_data.current(self.ASSET2, field)),
                            field)

        self.assertEqual(0, bar_data.current(self.ASSET2, "volume"))
        self.assertTrue(
            bar_data.current(self.ASSET2, "last_traded") is pd.NaT
        )

    def test_fully_active_day(self):
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[1],
            "daily",
        )
        self.check_internal_consistency(bar_data)

        # on self.bcolz_daily_bar_days[1], both assets have data
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
                self.bcolz_daily_bar_days[1],
                bar_data.current(asset, "last_traded")
            )

    def test_last_active_day(self):
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[-1],
            "daily",
        )
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
        next_day = self.env.next_trading_day(self.bcolz_daily_bar_days[-1])

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
                self.assertEqual(self.bcolz_daily_bar_days[-2], last_traded_dt)
            else:
                self.assertEqual(self.bcolz_daily_bar_days[1], last_traded_dt)

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
        adjustments = self.adjustment_reader.get_adjustments_for_sid(
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
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[0],
            "daily",
        )
        self.assertEqual(
            liquid_day_0_price,
            bar_data.current(liquid_asset, "price")
        )
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[1],
            "daily",
        )
        self.assertEqual(
            liquid_day_1_price,
            bar_data.current(liquid_asset, "price")
        )

        # ... except when we have to forward fill across a day boundary
        # ILLIQUID_ASSET has no data on days 0 and 2, and a split on day 2
        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[1],
            "daily",
        )
        self.assertEqual(
            illiquid_day_0_price, bar_data.current(illiquid_asset, "price")
        )

        bar_data = BarData(
            self.data_portal,
            lambda: self.bcolz_daily_bar_days[2],
            "daily",
        )

        # 3 (price from previous day) * 0.5 (split ratio)
        self.assertAlmostEqual(
            illiquid_day_1_price_adjusted,
            bar_data.current(illiquid_asset, "price")
        )
