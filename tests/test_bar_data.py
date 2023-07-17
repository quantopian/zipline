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
from datetime import timedelta, time
from itertools import chain

import numpy as np
import pandas as pd
import pytest
from numpy import nan
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from toolz import concat
from zipline._protocol import handle_non_market_minutes

from zipline.finance.asset_restrictions import (
    Restriction,
    HistoricalRestrictions,
    RESTRICTION_STATES,
)
from zipline.testing import (
    MockDailyBarReader,
    create_daily_df_for_asset,
    create_minute_df_for_asset,
    str_to_seconds,
)
from zipline.testing.fixtures import (
    WithCreateBarData,
    WithDataPortal,
    ZiplineTestCase,
)
from zipline.utils.calendar_utils import get_calendar, days_at_time

OHLC = ["open", "high", "low", "close"]
OHLCP = OHLC + ["price"]
ALL_FIELDS = OHLCP + ["volume", "last_traded"]

# offsets used in test data
field_info = {"open": 1, "high": 2, "low": -1, "close": 0}


def str_to_ts(dt_str):
    return pd.Timestamp(dt_str, tz="UTC")


def handle_get_calendar_exception(f):
    """exchange_calendars raises a ValueError when we call get_calendar
    for an already registered calendar with the 'side' argument"""

    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except ValueError as e:
            if (
                str(e)
                == "Receieved calendar arguments although TEST is registered as a specific instance "
                "of class <class 'exchange_calendars.exchange_calendar_xnys.XNYSExchangeCalendar'>, "
                "not as a calendar factory."
            ):
                msg = "Ignore get_calendar errors for now: " + str(e)
                print(msg)
                pytest.skip(msg)
            else:
                raise e

    return wrapper


class WithBarDataChecks:
    def assert_same(self, val1, val2):
        try:
            assert val1 == val2
        except AssertionError:
            if val1 is pd.NaT:
                assert val2 is pd.NaT
            elif np.isnan(val1):
                assert np.isnan(val2)
            else:
                raise

    def check_internal_consistency(self, bar_data):
        df = bar_data.current([self.ASSET1, self.ASSET2], ALL_FIELDS)

        asset1_multi_field = bar_data.current(self.ASSET1, ALL_FIELDS)
        asset2_multi_field = bar_data.current(self.ASSET2, ALL_FIELDS)

        for field in ALL_FIELDS:
            asset1_value = bar_data.current(self.ASSET1, field)
            asset2_value = bar_data.current(self.ASSET2, field)

            multi_asset_series = bar_data.current([self.ASSET1, self.ASSET2], field)

            # make sure all the different query forms are internally
            # consistent
            self.assert_same(multi_asset_series.loc[self.ASSET1], asset1_value)
            self.assert_same(multi_asset_series.loc[self.ASSET2], asset2_value)

            self.assert_same(df.loc[self.ASSET1][field], asset1_value)
            self.assert_same(df.loc[self.ASSET2][field], asset2_value)

            self.assert_same(asset1_multi_field[field], asset1_value)
            self.assert_same(asset2_multi_field[field], asset2_value)

        # also verify that bar_data doesn't expose anything bad
        for field in [
            "data_portal",
            "simulation_dt_func",
            "data_frequency",
            "_views",
            "_universe_func",
            "_last_calculated_universe",
            "_universe_last_updatedat",
        ]:
            with pytest.raises(AttributeError):
                getattr(bar_data, field)


class TestMinuteBarData(
    WithCreateBarData, WithBarDataChecks, WithDataPortal, ZiplineTestCase
):
    START_DATE = pd.Timestamp("2016-01-05")
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp("2016-01-07")

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3, 4, 5

    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    HILARIOUSLY_ILLIQUID_ASSET_SID = 5

    @classmethod
    def make_equity_minute_bar_data(cls):
        # asset1 has trades every minute
        # asset2 has trades every 10 minutes
        # split_asset trades every minute
        # illiquid_split_asset trades every 10 minutes
        for sid in (1, cls.SPLIT_ASSET_SID):
            yield sid, create_minute_df_for_asset(
                cls.trading_calendar,
                cls.equity_minute_bar_days[0],
                cls.equity_minute_bar_days[-1],
            )

        for sid in (2, cls.ILLIQUID_SPLIT_ASSET_SID):
            yield sid, create_minute_df_for_asset(
                cls.trading_calendar,
                cls.equity_minute_bar_days[0],
                cls.equity_minute_bar_days[-1],
                10,
            )

        yield cls.HILARIOUSLY_ILLIQUID_ASSET_SID, create_minute_df_for_asset(
            cls.trading_calendar,
            cls.equity_minute_bar_days[0],
            cls.equity_minute_bar_days[-1],
            50,
        )

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                6: {
                    "symbol": "CLG06",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2005-12-01"),
                    "notice_date": pd.Timestamp("2005-12-20"),
                    "expiration_date": pd.Timestamp("2006-01-20"),
                    "exchange": "ICEUS",
                },
                7: {
                    "symbol": "CLK06",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2005-12-01"),
                    "notice_date": pd.Timestamp("2006-03-20"),
                    "expiration_date": pd.Timestamp("2006-04-20"),
                    "exchange": "ICEUS",
                },
            },
            orient="index",
        )

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.SPLIT_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.ILLIQUID_SPLIT_ASSET_SID,
                },
            ]
        )

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

    def test_current_session(self):
        regular_minutes = self.trading_calendar.sessions_minutes(
            self.equity_minute_bar_days[0], self.equity_minute_bar_days[-1]
        )

        bts_minutes = days_at_time(
            self.equity_minute_bar_days,
            time(8, 45),
            "US/Eastern",
            day_offset=0,
        )

        # some other non-market-minute
        three_oh_six_am_minutes = days_at_time(
            self.equity_minute_bar_days,
            time(3, 6),
            "US/Eastern",
            day_offset=0,
        )

        all_minutes = [regular_minutes, bts_minutes, three_oh_six_am_minutes]
        for minute in list(concat(all_minutes)):
            bar_data = self.create_bardata(lambda: minute)

            assert (
                self.trading_calendar.minute_to_session(minute)
                == bar_data.current_session
            )

    def test_current_session_minutes(self):
        first_day_minutes = self.trading_calendar.session_minutes(
            self.equity_minute_bar_days[0]
        )

        for minute in first_day_minutes:
            bar_data = self.create_bardata(lambda: minute)
            np.testing.assert_array_equal(
                first_day_minutes, bar_data.current_session_minutes
            )

    def test_minute_before_assets_trading(self):
        # grab minutes that include the day before the asset start
        minutes = self.trading_calendar.session_minutes(
            self.trading_calendar.previous_session(self.equity_minute_bar_days[0])
        )

        # this entire day is before either asset has started trading
        for _, minute in enumerate(minutes):
            bar_data = self.create_bardata(
                lambda: minute,
            )
            self.check_internal_consistency(bar_data)

            assert not bar_data.can_trade(self.ASSET1)
            assert not bar_data.can_trade(self.ASSET2)

            assert not bar_data.is_stale(self.ASSET1)
            assert not bar_data.is_stale(self.ASSET2)

            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.current(asset, field)

                    if field in OHLCP:
                        assert np.isnan(asset_value)
                    elif field == "volume":
                        assert 0 == asset_value
                    elif field == "last_traded":
                        assert asset_value is pd.NaT

    @handle_get_calendar_exception
    def test_regular_minute(self):
        minutes = self.trading_calendar.session_minutes(self.equity_minute_bar_days[0])

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

            bar_data = self.create_bardata(
                lambda: minute,
            )
            self.check_internal_consistency(bar_data)
            asset2_has_data = ((idx + 1) % 10) == 0

            assert bar_data.can_trade(self.ASSET1)
            assert not bar_data.is_stale(self.ASSET1)

            if idx < 9:
                assert not bar_data.can_trade(self.ASSET2)
                assert not bar_data.is_stale(self.ASSET2)
            else:
                assert bar_data.can_trade(self.ASSET2)

                if asset2_has_data:
                    assert not bar_data.is_stale(self.ASSET2)
                else:
                    assert bar_data.is_stale(self.ASSET2)

            for field in ALL_FIELDS:
                asset1_value = bar_data.current(self.ASSET1, field)
                asset2_value = bar_data.current(self.ASSET2, field)

                # now check the actual values
                if idx == 0 and field == "low":
                    # first low value is 0, which is interpreted as NaN
                    assert np.isnan(asset1_value)
                else:
                    if field in OHLC:
                        assert idx + 1 + field_info[field] == asset1_value

                        if asset2_has_data:
                            assert idx + 1 + field_info[field] == asset2_value
                        else:
                            assert np.isnan(asset2_value)
                    elif field == "volume":
                        assert (idx + 1) * 100 == asset1_value

                        if asset2_has_data:
                            assert (idx + 1) * 100 == asset2_value
                        else:
                            assert 0 == asset2_value
                    elif field == "price":
                        assert idx + 1 == asset1_value

                        if asset2_has_data:
                            assert idx + 1 == asset2_value
                        elif idx < 9:
                            # no price to forward fill from
                            assert np.isnan(asset2_value)
                        else:
                            # forward-filled price
                            assert (idx // 10) * 10 == asset2_value
                    elif field == "last_traded":
                        assert minute == asset1_value

                        if idx < 9:
                            assert asset2_value is pd.NaT
                        elif asset2_has_data:
                            assert minute == asset2_value
                        else:
                            last_traded_minute = minutes[(idx // 10) * 10]
                            assert (
                                last_traded_minute - timedelta(minutes=1)
                                == asset2_value
                            )

    @handle_get_calendar_exception
    def test_minute_of_last_day(self):
        minutes = self.trading_calendar.session_minutes(
            self.equity_daily_bar_days[-1],
        )

        # this is the last day the assets exist
        for _, minute in enumerate(minutes):
            bar_data = self.create_bardata(
                lambda: minute,
            )

            assert bar_data.can_trade(self.ASSET1)
            assert bar_data.can_trade(self.ASSET2)

    def test_minute_after_assets_stopped(self):
        minutes = self.trading_calendar.session_minutes(
            self.trading_calendar.next_session(self.equity_minute_bar_days[-1])
        )

        last_trading_minute = self.trading_calendar.session_minutes(
            self.equity_minute_bar_days[-1]
        )[-1]

        # this entire day is after both assets have stopped trading
        for _, minute in enumerate(minutes):
            bar_data = self.create_bardata(
                lambda: minute,
            )

            assert not bar_data.can_trade(self.ASSET1)
            assert not bar_data.can_trade(self.ASSET2)

            assert not bar_data.is_stale(self.ASSET1)
            assert not bar_data.is_stale(self.ASSET2)

            self.check_internal_consistency(bar_data)

            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.current(asset, field)

                    if field in OHLCP:
                        assert np.isnan(asset_value)
                    elif field == "volume":
                        assert 0 == asset_value
                    elif field == "last_traded":
                        assert last_trading_minute == asset_value

    def test_get_value_is_unadjusted(self):
        # verify there is a split for SPLIT_ASSET
        splits = self.adjustment_reader.get_adjustments_for_sid(
            "splits", self.SPLIT_ASSET.sid
        )

        assert 1 == len(splits)
        split = splits[0]
        assert split[0] == pd.Timestamp("2016-01-06")

        # ... but that's it's not applied when using spot value
        minutes = self.trading_calendar.sessions_minutes(
            self.equity_minute_bar_days[0], self.equity_minute_bar_days[1]
        )

        for idx, minute in enumerate(minutes):
            bar_data = self.create_bardata(
                lambda: minute,
            )
            assert idx + 1 == bar_data.current(self.SPLIT_ASSET, "price")

    def test_get_value_is_adjusted_if_needed(self):
        # on cls.days[1], the first 9 minutes of ILLIQUID_SPLIT_ASSET are
        # missing. let's get them.
        day0_minutes = self.trading_calendar.session_minutes(
            self.equity_minute_bar_days[0]
        )
        day1_minutes = self.trading_calendar.session_minutes(
            self.equity_minute_bar_days[1]
        )

        for _, minute in enumerate(day0_minutes[-10:-1]):
            bar_data = self.create_bardata(
                lambda: minute,
            )
            assert 380 == bar_data.current(self.ILLIQUID_SPLIT_ASSET, "price")

        bar_data = self.create_bardata(
            lambda: day0_minutes[-1],
        )

        assert 390 == bar_data.current(self.ILLIQUID_SPLIT_ASSET, "price")

        for _, minute in enumerate(day1_minutes[0:9]):
            bar_data = self.create_bardata(
                lambda: minute,
            )

            # should be half of 390, due to the split
            assert 195 == bar_data.current(self.ILLIQUID_SPLIT_ASSET, "price")

    def test_get_value_at_midnight(self):
        # make sure that if we try to get a minute price at a non-market
        # minute, we use the previous market close's timestamp
        day = self.equity_minute_bar_days[1]

        eight_fortyfive_am_eastern = pd.Timestamp(
            "{0}-{1}-{2} 8:45".format(day.year, day.month, day.day), tz="US/Eastern"
        )

        bar_data = self.create_bardata(
            lambda: day,
        )
        bar_data2 = self.create_bardata(
            lambda: eight_fortyfive_am_eastern,
        )

        with handle_non_market_minutes(bar_data), handle_non_market_minutes(bar_data2):
            for bd in [bar_data, bar_data2]:
                for field in ["close", "price"]:
                    assert 390 == bd.current(self.ASSET1, field)

                # make sure that if the asset didn't trade at the previous
                # close, we properly ffill (or not ffill)
                assert 350 == bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, "price")
                assert np.isnan(bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, "high"))
                assert 0 == bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, "volume")

    def test_get_value_during_non_market_hours(self):
        # make sure that if we try to get the OHLCV values of ASSET1 during
        # non-market hours, we don't get the previous market minute's values

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: pd.Timestamp("2016-01-06 4:15", tz="US/Eastern"),
        )

        assert np.isnan(bar_data.current(self.ASSET1, "open"))
        assert np.isnan(bar_data.current(self.ASSET1, "high"))
        assert np.isnan(bar_data.current(self.ASSET1, "low"))
        assert np.isnan(bar_data.current(self.ASSET1, "close"))
        assert 0 == bar_data.current(self.ASSET1, "volume")

        # price should still forward fill
        assert 390 == bar_data.current(self.ASSET1, "price")

    def test_can_trade_equity_same_cal_outside_lifetime(self):

        # verify that can_trade returns False for the session before the
        # asset's first session
        session_before_asset1_start = self.trading_calendar.previous_session(
            self.ASSET1.start_date
        )
        minutes_for_session = self.trading_calendar.session_minutes(
            session_before_asset1_start
        )

        # for good measure, check the minute before the session too
        minutes_to_check = chain(
            [minutes_for_session[0] - pd.Timedelta(minutes=1)], minutes_for_session
        )

        for minute in minutes_to_check:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: minute,
            )

            assert not bar_data.can_trade(self.ASSET1)

        # after asset lifetime
        session_after_asset1_end = self.trading_calendar.next_session(
            self.ASSET1.end_date
        )
        bts_after_asset1_end = session_after_asset1_end.replace(
            hour=8, minute=45
        ).tz_localize("US/Eastern")

        minutes_to_check = chain(
            self.trading_calendar.session_minutes(session_after_asset1_end),
            [bts_after_asset1_end],
        )

        for minute in minutes_to_check:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: minute,
            )

            assert not bar_data.can_trade(self.ASSET1)

    @handle_get_calendar_exception
    def test_can_trade_equity_same_cal_exchange_closed(self):
        # verify that can_trade returns true for minutes that are
        # outside the asset's calendar (assuming the asset is alive and
        # there is a last price), because the asset is alive on the
        # next market minute.
        minutes = self.trading_calendar.sessions_minutes(
            self.ASSET1.start_date, self.ASSET1.end_date
        )

        for minute in minutes:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: minute,
            )

            assert bar_data.can_trade(self.ASSET1)

    @handle_get_calendar_exception
    def test_can_trade_equity_same_cal_no_last_price(self):
        # self.HILARIOUSLY_ILLIQUID_ASSET's first trade is at
        # 2016-01-05 15:20:00+00:00.  Make sure that can_trade returns false
        # for all minutes in that session before the first trade, and true
        # for all minutes afterwards.

        minutes_in_session = self.trading_calendar.session_minutes(
            self.ASSET1.start_date
        )

        for minute in minutes_in_session[0:49]:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: minute,
            )

            assert not bar_data.can_trade(self.HILARIOUSLY_ILLIQUID_ASSET)

        for minute in minutes_in_session[50:]:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: minute,
            )

            assert bar_data.can_trade(self.HILARIOUSLY_ILLIQUID_ASSET)

    def test_is_stale_during_non_market_hours(self):
        bar_data = self.create_bardata(
            lambda: self.equity_minute_bar_days[1],
        )

        with handle_non_market_minutes(bar_data):
            assert bar_data.is_stale(self.HILARIOUSLY_ILLIQUID_ASSET)

    def test_overnight_adjustments(self):
        # verify there is a split for SPLIT_ASSET
        splits = self.adjustment_reader.get_adjustments_for_sid(
            "splits", self.SPLIT_ASSET.sid
        )

        assert 1 == len(splits)
        split = splits[0]
        assert split[0] == pd.Timestamp("2016-01-06")

        # Current day is 1/06/16
        day = self.equity_daily_bar_days[1]
        eight_fortyfive_am_eastern = pd.Timestamp(
            "{0}-{1}-{2} 8:45".format(day.year, day.month, day.day), tz="US/Eastern"
        )

        bar_data = self.create_bardata(
            lambda: eight_fortyfive_am_eastern,
        )

        expected = {
            "open": 391 / 2.0,
            "high": 392 / 2.0,
            "low": 389 / 2.0,
            "close": 390 / 2.0,
            "volume": 39000 * 2.0,
            "price": 390 / 2.0,
        }

        with handle_non_market_minutes(bar_data):
            for field in OHLCP + ["volume"]:
                value = bar_data.current(self.SPLIT_ASSET, field)

                # Assert the price is adjusted for the overnight split
                assert value == expected[field]

    @handle_get_calendar_exception
    def test_can_trade_restricted(self):
        """Test that can_trade will return False for a sid if it is restricted
        on that dt
        """

        minutes_to_check = [
            (str_to_ts("2016-01-05 14:31"), False),
            (str_to_ts("2016-01-06 14:31"), False),
            (str_to_ts("2016-01-07 14:31"), True),
            (str_to_ts("2016-01-07 15:00"), False),
            (str_to_ts("2016-01-07 15:30"), True),
        ]

        rlm = HistoricalRestrictions(
            [
                Restriction(1, str_to_ts("2016-01-05"), RESTRICTION_STATES.FROZEN),
                Restriction(1, str_to_ts("2016-01-07"), RESTRICTION_STATES.ALLOWED),
                Restriction(
                    1, str_to_ts("2016-01-07 15:00"), RESTRICTION_STATES.FROZEN
                ),
                Restriction(
                    1, str_to_ts("2016-01-07 15:30"), RESTRICTION_STATES.ALLOWED
                ),
            ]
        )

        for info in minutes_to_check:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: info[0],
                restrictions=rlm,
            )
            assert bar_data.can_trade(self.ASSET1) == info[1]


class TestMinuteBarDataFuturesCalendar(
    WithCreateBarData, WithBarDataChecks, ZiplineTestCase
):
    START_DATE = pd.Timestamp("2016-01-05")
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp("2016-01-07")

    ASSET_FINDER_EQUITY_SIDS = [1]

    @classmethod
    def make_equity_minute_bar_data(cls):
        # asset1 has trades every minute
        yield 1, create_minute_df_for_asset(
            cls.trading_calendar,
            cls.equity_minute_bar_days[0],
            cls.equity_minute_bar_days[-1],
        )

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                6: {
                    "symbol": "CLH16",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2016-01-04"),
                    "notice_date": pd.Timestamp("2016-01-19"),
                    "expiration_date": pd.Timestamp("2016-02-19"),
                    "exchange": "ICEUS",
                },
                7: {
                    "symbol": "FVH16",
                    "root_symbol": "FV",
                    "start_date": pd.Timestamp("2016-01-04"),
                    "notice_date": pd.Timestamp("2016-01-22"),
                    "expiration_date": pd.Timestamp("2016-02-22"),
                    "auto_close_date": pd.Timestamp("2016-01-20"),
                    "exchange": "CMES",
                },
            },
            orient="index",
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestMinuteBarDataFuturesCalendar, cls).init_class_fixtures()
        cls.trading_calendar = get_calendar("CMES")

    @handle_get_calendar_exception
    def test_can_trade_multiple_exchange_closed(self):
        nyse_asset = self.asset_finder.retrieve_asset(1)
        ice_asset = self.asset_finder.retrieve_asset(6)

        # minutes we're going to check (to verify that that the same bardata
        # can check multiple exchange calendars, all times Eastern):
        # 2016-01-05:
        # 20:00 (minute before ICE opens)
        # 20:01 (first minute of ICE session)
        # 20:02 (second minute of ICE session)
        # 00:00 (Cinderella's ride becomes a pumpkin)
        # 2016-01-06:
        # 9:30 (minute before NYSE opens)
        # 9:31 (first minute of NYSE session)
        # 9:32 (second minute of NYSE session)
        # 15:59 (second-to-last minute of NYSE session)
        # 16:00 (last minute of NYSE session)
        # 16:01 (minute after NYSE closed)
        # 17:59 (second-to-last minute of ICE session)
        # 18:00 (last minute of ICE session)
        # 18:01 (minute after ICE closed)

        # each row is dt, whether-nyse-is-open, whether-ice-is-open
        minutes_to_check = [
            (pd.Timestamp("2016-01-05 20:00", tz="US/Eastern"), False, False),
            (pd.Timestamp("2016-01-05 20:01", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-05 20:02", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-06 00:00", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-06 9:30", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-06 9:31", tz="US/Eastern"), True, True),
            (pd.Timestamp("2016-01-06 9:32", tz="US/Eastern"), True, True),
            (pd.Timestamp("2016-01-06 15:59", tz="US/Eastern"), True, True),
            (pd.Timestamp("2016-01-06 16:00", tz="US/Eastern"), True, True),
            (pd.Timestamp("2016-01-06 16:01", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-06 17:59", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-06 18:00", tz="US/Eastern"), False, True),
            (pd.Timestamp("2016-01-06 18:01", tz="US/Eastern"), False, False),
        ]

        for info in minutes_to_check:
            # use the CMES calendar, which covers 24 hours
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: info[0],
            )

            series = bar_data.can_trade([nyse_asset, ice_asset])

            assert info[1] == series.loc[nyse_asset]
            assert info[2] == series.loc[ice_asset]

    def test_can_trade_delisted(self):
        """
        Test that can_trade returns False for an asset after its auto close
        date.
        """
        auto_closing_asset = self.asset_finder.retrieve_asset(7)

        # Our asset's auto close date is 2016-01-20, which means that as of the
        # market open for the 2016-01-21 session, `can_trade` should return
        # False.
        minutes_to_check = [
            (pd.Timestamp("2016-01-20 00:00:00", tz="UTC"), True),
            (pd.Timestamp("2016-01-20 23:00:00", tz="UTC"), True),
            (pd.Timestamp("2016-01-20 23:01:00", tz="UTC"), False),
            (pd.Timestamp("2016-01-20 23:59:00", tz="UTC"), False),
            (pd.Timestamp("2016-01-21 00:00:00", tz="UTC"), False),
            (pd.Timestamp("2016-01-21 00:01:00", tz="UTC"), False),
            (pd.Timestamp("2016-01-22 00:00:00", tz="UTC"), False),
        ]

        for info in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda: info[0])
            assert bar_data.can_trade(auto_closing_asset) == info[1]


class TestDailyBarData(
    WithCreateBarData, WithBarDataChecks, WithDataPortal, ZiplineTestCase
):
    START_DATE = pd.Timestamp("2016-01-05")
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp("2016-01-11")
    CREATE_BARDATA_DATA_FREQUENCY = "daily"

    ASSET_FINDER_EQUITY_SIDS = set(range(1, 9))

    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    MERGER_ASSET_SID = 5
    ILLIQUID_MERGER_ASSET_SID = 6
    DIVIDEND_ASSET_SID = 7
    ILLIQUID_DIVIDEND_ASSET_SID = 8

    @classmethod
    def make_equity_info(cls):
        frame = super(TestDailyBarData, cls).make_equity_info()
        frame.loc[[1, 2], "end_date"] = pd.Timestamp("2016-01-08")
        return frame

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.SPLIT_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.5,
                    "sid": cls.ILLIQUID_SPLIT_ASSET_SID,
                },
            ]
        )

    @classmethod
    def make_mergers_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-06"),
                    "ratio": 0.5,
                    "sid": cls.MERGER_ASSET_SID,
                },
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.6,
                    "sid": cls.ILLIQUID_MERGER_ASSET_SID,
                },
            ]
        )

    @classmethod
    def make_dividends_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    # only care about ex date, the other dates don't matter here
                    "ex_date": pd.Timestamp("2016-01-06").to_datetime64(),
                    "record_date": pd.Timestamp("2016-01-06").to_datetime64(),
                    "declared_date": pd.Timestamp("2016-01-06").to_datetime64(),
                    "pay_date": pd.Timestamp("2016-01-06").to_datetime64(),
                    "amount": 2.0,
                    "sid": cls.DIVIDEND_ASSET_SID,
                },
                {
                    "ex_date": pd.Timestamp("2016-01-07").to_datetime64(),
                    "record_date": pd.Timestamp("2016-01-07").to_datetime64(),
                    "declared_date": pd.Timestamp("2016-01-07").to_datetime64(),
                    "pay_date": pd.Timestamp("2016-01-07").to_datetime64(),
                    "amount": 4.0,
                    "sid": cls.ILLIQUID_DIVIDEND_ASSET_SID,
                },
            ],
            columns=[
                "ex_date",
                "record_date",
                "declared_date",
                "pay_date",
                "amount",
                "sid",
            ],
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
    def make_equity_daily_bar_data(cls, country_code, sids):
        for sid in sids:
            asset = cls.asset_finder.retrieve_asset(sid)
            yield sid, create_daily_df_for_asset(
                cls.trading_calendar,
                asset.start_date,
                asset.end_date,
                interval=2 - sid % 2,
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

    def get_last_minute_of_session(self, session_label):
        return self.trading_calendar.session_close(session_label)

    def test_current_session(self):
        for session in self.trading_calendar.sessions_in_range(
            self.equity_daily_bar_days[0], self.equity_daily_bar_days[-1]
        ):
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: self.get_last_minute_of_session(session)
            )

            assert session == bar_data.current_session

    def test_day_before_assets_trading(self):
        # use the day before self.bcolz_daily_bar_days[0]
        minute = self.get_last_minute_of_session(
            self.trading_calendar.previous_session(self.equity_daily_bar_days[0])
        )

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: minute,
        )
        self.check_internal_consistency(bar_data)

        assert not bar_data.can_trade(self.ASSET1)
        assert not bar_data.can_trade(self.ASSET2)

        assert not bar_data.is_stale(self.ASSET1)
        assert not bar_data.is_stale(self.ASSET2)

        for field in ALL_FIELDS:
            for asset in self.ASSETS:
                asset_value = bar_data.current(asset, field)

                if field in OHLCP:
                    assert np.isnan(asset_value)
                elif field == "volume":
                    assert 0 == asset_value
                elif field == "last_traded":
                    assert asset_value is pd.NaT

    def test_semi_active_day(self):
        # on self.equity_daily_bar_days[0], only asset1 has data
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.get_last_minute_of_session(
                self.equity_daily_bar_days[0]
            ).tz_convert(None),
        )
        self.check_internal_consistency(bar_data)

        assert bar_data.can_trade(self.ASSET1)
        assert not bar_data.can_trade(self.ASSET2)

        # because there is real data
        assert not bar_data.is_stale(self.ASSET1)

        # because there has never been a trade bar yet
        assert not bar_data.is_stale(self.ASSET2)

        assert 3 == bar_data.current(self.ASSET1, "open")
        assert 4 == bar_data.current(self.ASSET1, "high")
        assert 1 == bar_data.current(self.ASSET1, "low")
        assert 2 == bar_data.current(self.ASSET1, "close")
        assert 200 == bar_data.current(self.ASSET1, "volume")
        assert 2 == bar_data.current(self.ASSET1, "price")
        assert self.equity_daily_bar_days[0] == bar_data.current(
            self.ASSET1, "last_traded"
        )

        for field in OHLCP:
            assert np.isnan(bar_data.current(self.ASSET2, field)), field

        assert 0 == bar_data.current(self.ASSET2, "volume")
        assert bar_data.current(self.ASSET2, "last_traded") is pd.NaT

    def test_fully_active_day(self):
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.get_last_minute_of_session(
                self.equity_daily_bar_days[1]
            ),
        )
        self.check_internal_consistency(bar_data)

        # on self.equity_daily_bar_days[1], both assets have data
        for asset in self.ASSETS:
            assert bar_data.can_trade(asset)
            assert not bar_data.is_stale(asset)

            assert 4 == bar_data.current(asset, "open")
            assert 5 == bar_data.current(asset, "high")
            assert 2 == bar_data.current(asset, "low")
            assert 3 == bar_data.current(asset, "close")
            assert 300 == bar_data.current(asset, "volume")
            assert 3 == bar_data.current(asset, "price")
            assert self.equity_daily_bar_days[1] == bar_data.current(
                asset, "last_traded"
            )

    def test_last_active_day(self):
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.get_last_minute_of_session(
                self.equity_daily_bar_days[-1]
            ),
        )
        self.check_internal_consistency(bar_data)

        for asset in self.ASSETS:
            if asset in (1, 2):
                assert not bar_data.can_trade(asset)
            else:
                assert bar_data.can_trade(asset)
            assert not bar_data.is_stale(asset)

            if asset in (1, 2):
                assert_almost_equal(nan, bar_data.current(asset, "open"))
                assert_almost_equal(nan, bar_data.current(asset, "high"))
                assert_almost_equal(nan, bar_data.current(asset, "low"))
                assert_almost_equal(nan, bar_data.current(asset, "close"))
                assert_almost_equal(0, bar_data.current(asset, "volume"))
                assert_almost_equal(nan, bar_data.current(asset, "price"))
            else:
                assert 6 == bar_data.current(asset, "open")
                assert 7 == bar_data.current(asset, "high")
                assert 4 == bar_data.current(asset, "low")
                assert 5 == bar_data.current(asset, "close")
                assert 500 == bar_data.current(asset, "volume")
                assert 5 == bar_data.current(asset, "price")

    def test_after_assets_dead(self):
        session = self.END_DATE

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: session,
        )
        self.check_internal_consistency(bar_data)

        for asset in self.ASSETS:
            assert not bar_data.can_trade(asset)
            assert not bar_data.is_stale(asset)

            for field in OHLCP:
                assert np.isnan(bar_data.current(asset, field))

            assert 0 == bar_data.current(asset, "volume")

            last_traded_dt = bar_data.current(asset, "last_traded")

            if asset in (self.ASSET1, self.ASSET2):
                assert self.equity_daily_bar_days[3] == last_traded_dt

    @parameterized.expand(
        [("split", 2, 3, 3, 1.5), ("merger", 2, 3, 3, 1.8), ("dividend", 2, 3, 3, 2.88)]
    )
    def test_get_value_adjustments(
        self,
        adjustment_type,
        liquid_day_0_price,
        liquid_day_1_price,
        illiquid_day_0_price,
        illiquid_day_1_price_adjusted,
    ):
        """Test the behaviour of spot prices during adjustments."""
        table_name = adjustment_type + "s"
        liquid_asset = getattr(self, (adjustment_type.upper() + "_ASSET"))
        illiquid_asset = getattr(
            self, ("ILLIQUID_" + adjustment_type.upper() + "_ASSET")
        )
        # verify there is an adjustment for liquid_asset
        adjustments = self.adjustment_reader.get_adjustments_for_sid(
            table_name, liquid_asset.sid
        )

        assert 1 == len(adjustments)
        adjustment = adjustments[0]
        assert adjustment[0] == pd.Timestamp("2016-01-06")

        # ... but that's it's not applied when using spot value
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.equity_daily_bar_days[0],
        )
        assert liquid_day_0_price == bar_data.current(liquid_asset, "price")
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.equity_daily_bar_days[1],
        )
        assert liquid_day_1_price == bar_data.current(liquid_asset, "price")

        # ... except when we have to forward fill across a day boundary
        # ILLIQUID_ASSET has no data on days 0 and 2, and a split on day 2
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.equity_daily_bar_days[1],
        )
        assert illiquid_day_0_price == bar_data.current(illiquid_asset, "price")

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.equity_daily_bar_days[2],
        )

        # 3 (price from previous day) * 0.5 (split ratio)
        assert (
            round(
                abs(
                    illiquid_day_1_price_adjusted
                    - bar_data.current(illiquid_asset, "price")
                ),
                7,
            )
            == 0
        )

    def test_can_trade_restricted(self):
        """Test that can_trade will return False for a sid if it is restricted
        on that dt
        """

        minutes_to_check = [
            (pd.Timestamp("2016-01-05", tz="UTC"), False),
            (pd.Timestamp("2016-01-06", tz="UTC"), False),
            (pd.Timestamp("2016-01-07", tz="UTC"), True),
        ]

        rlm = HistoricalRestrictions(
            [
                Restriction(1, str_to_ts("2016-01-05"), RESTRICTION_STATES.FROZEN),
                Restriction(1, str_to_ts("2016-01-07"), RESTRICTION_STATES.ALLOWED),
            ]
        )

        for info in minutes_to_check:
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: info[0], restrictions=rlm
            )
            assert bar_data.can_trade(self.ASSET1) == info[1]
