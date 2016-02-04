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
from datetime import timedelta
import bcolz
import numpy as np
import pandas as pd

from unittest import TestCase
from pandas.tslib import normalize_date
from testfixtures import TempDirectory
from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import BcolzMinuteBarReader
from zipline.data.us_equity_pricing import SQLiteAdjustmentWriter, \
    SQLiteAdjustmentReader
from zipline.finance.trading import TradingEnvironment, SimulationParameters
from zipline.data.us_equity_pricing import BcolzDailyBarReader
from zipline.data.future_pricing import FutureMinuteReader
from zipline.utils.test_utils import write_bcolz_minute_data
from .utils.daily_bar_writer import DailyBarWriterFromDataFrames


class TestDataPortal(TestCase):
    def test_forward_fill_minute(self):
        tempdir = TempDirectory()
        try:
            env = TradingEnvironment()
            env.write_data(
                equities_data={
                    0: {
                        'start_date': pd.Timestamp("2015-09-28", tz='UTC'),
                        'end_date': pd.Timestamp("2015-09-29", tz='UTC')
                        + timedelta(days=1)
                    }
                }
            )

            minutes = env.minutes_for_days_in_range(
                start=pd.Timestamp("2015-09-28", tz='UTC'),
                end=pd.Timestamp("2015-09-29", tz='UTC')
            )

            df = pd.DataFrame({
                # one missing bar, then 200 bars of real data,
                # then 1.5 days of missing data
                "open": np.array([0] + list(range(0, 200)) + [0] * 579),
                "high": np.array([0] + list(range(1000, 1200)) + [0] * 579),
                "low": np.array([0] + list(range(2000, 2200)) + [0] * 579),
                "close": np.array([0] + list(range(3000, 3200)) + [0] * 579),
                "volume": [0] + list(range(4000, 4200)) + [0] * 579,
                "dt": minutes
            }).set_index("dt")

            write_bcolz_minute_data(
                env,
                pd.date_range(
                    start=normalize_date(minutes[0]),
                    end=normalize_date(minutes[-1])
                ),
                tempdir.path,
                {0: df}
            )

            sim_params = SimulationParameters(
                period_start=minutes[0],
                period_end=minutes[-1],
                data_frequency="minute",
                env=env,
            )

            equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

            dp = DataPortal(
                env,
                equity_minute_reader=equity_minute_reader,
            )

            for minute_idx, minute in enumerate(minutes):
                for field_idx, field in enumerate(
                        ["open", "high", "low", "close", "volume"]):
                    val = dp.get_spot_value(
                        0, field,
                        dt=minute,
                        data_frequency=sim_params.data_frequency)
                    if minute_idx == 0:
                        self.assertEqual(0, val)
                    elif minute_idx < 200:
                        self.assertEqual((minute_idx - 1) +
                                         (field_idx * 1000), val)
                    else:
                        self.assertEqual(199 + (field_idx * 1000), val)
        finally:
            tempdir.cleanup()

    def test_forward_fill_daily(self):
        tempdir = TempDirectory()
        try:
            # 17 trading days
            start_day = pd.Timestamp("2015-09-07", tz='UTC')
            end_day = pd.Timestamp("2015-09-30", tz='UTC')

            env = TradingEnvironment()
            env.write_data(
                equities_data={
                    0: {
                        'start_date': start_day,
                        'end_date': end_day
                    }
                }
            )

            days = env.days_in_range(start_day, end_day)

            # first bar is missing.  then 8 real bars.  then 8 more missing
            # bars.
            df = pd.DataFrame({
                "open": [0] + list(range(0, 8)) + [0] * 8,
                "high": [0] + list(range(10, 18)) + [0] * 8,
                "low": [0] + list(range(20, 28)) + [0] * 8,
                "close": [0] + list(range(30, 38)) + [0] * 8,
                "volume": [0] + list(range(40, 48)) + [0] * 8,
                "day": [day.value for day in days]
            }, index=days)

            assets = {0: df}
            path = os.path.join(tempdir.path, "testdaily.bcolz")

            DailyBarWriterFromDataFrames(assets).write(
                path,
                days,
                assets
            )

            sim_params = SimulationParameters(
                period_start=days[0],
                period_end=days[-1],
                data_frequency="daily"
            )

            equity_daily_reader = BcolzDailyBarReader(path)

            dp = DataPortal(
                env,
                equity_daily_reader=equity_daily_reader,
            )

            for day_idx, day in enumerate(days):
                for field_idx, field in enumerate(
                        ["open", "high", "low", "close", "volume"]):
                    val = dp.get_spot_value(
                        0, field,
                        dt=day,
                        data_frequency=sim_params.data_frequency)
                    if day_idx == 0:
                        self.assertEqual(0, val)
                    elif day_idx < 9:
                        self.assertEqual((day_idx - 1) + (field_idx * 10), val)
                    else:
                        self.assertEqual(7 + (field_idx * 10), val)
        finally:
            tempdir.cleanup()

    def test_adjust_forward_fill_minute(self):
        tempdir = TempDirectory()
        try:
            start_day = pd.Timestamp("2013-06-21", tz='UTC')
            end_day = pd.Timestamp("2013-06-24", tz='UTC')

            env = TradingEnvironment()
            env.write_data(
                equities_data={
                    0: {
                        'start_date': start_day,
                        'end_date': env.next_trading_day(end_day)
                    }
                }
            )

            minutes = env.minutes_for_days_in_range(
                start=start_day,
                end=end_day
            )

            df = pd.DataFrame({
                # 390 bars of real data, then 100 missing bars, then 290
                # bars of data again
                "open": np.array(list(range(0, 390)) + [0] * 100 +
                                 list(range(390, 680))),
                "high": np.array(list(range(1000, 1390)) + [0] * 100 +
                                 list(range(1390, 1680))),
                "low": np.array(list(range(2000, 2390)) + [0] * 100 +
                                list(range(2390, 2680))),
                "close": np.array(list(range(3000, 3390)) + [0] * 100 +
                                  list(range(3390, 3680))),
                "volume": np.array(list(range(4000, 4390)) + [0] * 100 +
                                   list(range(4390, 4680))),
                "dt": minutes
            }).set_index("dt")

            write_bcolz_minute_data(
                env,
                env.days_in_range(start_day, end_day),
                tempdir.path,
                {0: df}
            )

            sim_params = SimulationParameters(
                period_start=minutes[0],
                period_end=minutes[-1],
                data_frequency="minute",
                env=env
            )

            # create a split for 6/24
            adjustments_path = os.path.join(tempdir.path, "adjustments.db")
            writer = SQLiteAdjustmentWriter(adjustments_path,
                                            pd.date_range(start=start_day,
                                                          end=end_day),
                                            None)

            splits = pd.DataFrame([{
                'effective_date': int(end_day.value / 1e9),
                'ratio': 0.5,
                'sid': 0
            }])

            dividend_data = {
                # Hackery to make the dtypes correct on an empty frame.
                'ex_date': np.array([], dtype='datetime64[ns]'),
                'pay_date': np.array([], dtype='datetime64[ns]'),
                'record_date': np.array([], dtype='datetime64[ns]'),
                'declared_date': np.array([], dtype='datetime64[ns]'),
                'amount': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            }
            dividends = pd.DataFrame(
                dividend_data,
                index=pd.DatetimeIndex([], tz='UTC'),
                columns=['ex_date',
                         'pay_date',
                         'record_date',
                         'declared_date',
                         'amount',
                         'sid']
            )

            merger_data = {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            }
            mergers = pd.DataFrame(
                merger_data,
                index=pd.DatetimeIndex([], tz='UTC')
            )

            writer.write(splits, mergers, dividends)

            equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

            dp = DataPortal(
                env,
                equity_minute_reader=equity_minute_reader,
                adjustment_reader=SQLiteAdjustmentReader(adjustments_path)
            )

            # phew, finally ready to start testing.
            for idx, minute in enumerate(minutes[:390]):
                for field_idx, field in enumerate(["open", "high", "low",
                                                   "close", "volume"]):
                    self.assertEqual(
                        dp.get_spot_value(
                            0, field,
                            dt=minute,
                            data_frequency=sim_params.data_frequency),
                        idx + (1000 * field_idx)
                    )

            for idx, minute in enumerate(minutes[390:490]):
                # no actual data for this part, so we'll forward-fill.
                # make sure the forward-filled values are adjusted.
                for field_idx, field in enumerate(["open", "high", "low",
                                                   "close"]):
                    self.assertEqual(
                        dp.get_spot_value(
                            0, field,
                            dt=minute,
                            data_frequency=sim_params.data_frequency),
                        (389 + (1000 * field_idx)) / 2.0
                    )

                self.assertEqual(
                    dp.get_spot_value(
                        0, "volume",
                        dt=minute,
                        data_frequency=sim_params.data_frequency),
                    8778  # 4389 * 2
                )

            for idx, minute in enumerate(minutes[490:]):
                # back to real data
                for field_idx, field in enumerate(["open", "high", "low",
                                                   "close", "volume"]):
                    self.assertEqual(
                        dp.get_spot_value(
                            0, field,
                            dt=minute,
                            data_frequency=sim_params.data_frequency
                        ),
                        (390 + idx + (1000 * field_idx))
                    )
        finally:
            tempdir.cleanup()

    def test_last_traded_dt(self):
        tempdir = TempDirectory()
        try:
            start_day = pd.Timestamp("2013-06-21", tz='UTC')
            end_day = pd.Timestamp("2013-06-24", tz='UTC')

            env = TradingEnvironment()
            env.write_data(
                equities_data={
                    0: {
                        'start_date': start_day,
                        'end_date': env.next_trading_day(end_day)
                    }
                }
            )

            minutes = env.minutes_for_days_in_range(
                start=start_day,
                end=end_day
            )

            df = pd.DataFrame({
                # 390 bars of real data, then 100 missing bars, then 290
                # bars of data again
                "open": np.array(list(range(0, 390)) + [0] * 100 +
                                 list(range(390, 680))),
                "high": np.array(list(range(1000, 1390)) + [0] * 100 +
                                 list(range(1390, 1680))),
                "low": np.array(list(range(2000, 2390)) + [0] * 100 +
                                list(range(2390, 2680))),
                "close": np.array(list(range(3000, 3390)) + [0] * 100 +
                                  list(range(3390, 3680))),
                "volume": np.array(list(range(4000, 4390)) + [0] * 100 +
                                   list(range(4390, 4680))),
                "dt": minutes
            }).set_index("dt")

            write_bcolz_minute_data(
                env,
                env.days_in_range(start_day, end_day),
                tempdir.path,
                {0: df}
            )

            equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

            dp = DataPortal(
                env,
                equity_minute_reader=equity_minute_reader,
            )

            asset = env.asset_finder.retrieve_asset(0)

            minute_with_trade = minutes[389]

            minute_without_trade = minutes[390]

            last_traded = dp.get_last_traded_dt(asset, minute_with_trade,
                                                'minute')

            self.assertEqual(last_traded, minute_with_trade)

            last_traded = dp.get_last_traded_dt(asset, minute_without_trade,
                                                'minute')

            minute_without_trade = minutes[489]

            last_traded = dp.get_last_traded_dt(asset, minute_without_trade,
                                                'minute')

            self.assertEqual(last_traded, minute_with_trade)
        finally:
            tempdir.cleanup()

    def test_last_traded_dt_daily(self):
        tempdir = TempDirectory()
        try:
            # 17 trading days
            start_day = pd.Timestamp("2015-09-07", tz='UTC')
            end_day = pd.Timestamp("2015-09-30", tz='UTC')

            env = TradingEnvironment()
            env.write_data(
                equities_data={
                    0: {
                        'start_date': start_day,
                        'end_date': end_day
                    },
                    1: {
                        'start_date': env.next_trading_day(start_day),
                        'end_date': end_day
                    }
                }
            )

            days = env.days_in_range(start_day, end_day)

            # first bar is missing.  then 8 real bars.  then 8 more missing
            # bars.
            df = pd.DataFrame({
                "open": [0] + list(range(0, 8)) + [0] * 8,
                "high": [0] + list(range(10, 18)) + [0] * 8,
                "low": [0] + list(range(20, 28)) + [0] * 8,
                "close": [0] + list(range(30, 38)) + [0] * 8,
                "volume": [0] + list(range(40, 48)) + [0] * 8,
                "day": [day.value for day in days]
            }, index=days)
            # Test a second sid, so that edge condition with very first sid
            # in calendar, as well as a sid with a start date after the
            # calendar start are tested for the 'no leading data case'
            df_sid_1 = pd.DataFrame({
                "open": [0] + list(range(0, 8)) + [0] * 7,
                "high": [0] + list(range(10, 18)) + [0] * 7,
                "low": [0] + list(range(20, 28)) + [0] * 7,
                "close": [0] + list(range(30, 38)) + [0] * 7,
                "volume": [0] + list(range(40, 48)) + [0] * 7,
                "day": [day.value for day in days[1:]]
            }, index=days[1:])

            assets = {0: df, 1: df_sid_1}
            path = os.path.join(tempdir.path, "testdaily.bcolz")

            DailyBarWriterFromDataFrames(assets).write(
                path,
                days,
                assets
            )

            equity_daily_reader = BcolzDailyBarReader(path)

            dp = DataPortal(
                env,
                equity_daily_reader=equity_daily_reader,
            )

            asset = env.asset_finder.retrieve_asset(0)

            # Day with trades.
            day_with_trade = df.index[8]
            last_traded = dp.get_last_traded_dt(asset, day_with_trade,
                                                'daily')

            self.assertEqual(last_traded, day_with_trade)

            # Day with no trades, should return most recent with trade.
            day_without_trade = df.index[11]
            last_traded = dp.get_last_traded_dt(asset, day_without_trade,
                                                'daily')

            self.assertEqual(last_traded, day_with_trade)

            first_day_also_no_trade = df.index[0]

            # Beginning bar, should return None.
            last_traded = dp.get_last_traded_dt(asset, first_day_also_no_trade,
                                                'daily')

            self.assertEqual(last_traded, None)

            asset = env.asset_finder.retrieve_asset(1)

            # Day with trades.
            day_with_trade = df_sid_1.index[8]
            last_traded = dp.get_last_traded_dt(asset, day_with_trade,
                                                'daily')

            self.assertEqual(last_traded, day_with_trade)

            # Day with no trades, should return most recent with trade.
            day_without_trade = df_sid_1.index[10]
            last_traded = dp.get_last_traded_dt(asset, day_without_trade,
                                                'daily')

            self.assertEqual(last_traded, day_with_trade)

            first_day_also_no_trade = df_sid_1.index[0]

            # Beginning bar, should return None.
            last_traded = dp.get_last_traded_dt(asset, first_day_also_no_trade,
                                                'daily')

            self.assertEqual(last_traded, None)

        finally:
            tempdir.cleanup()

    def test_spot_value_futures(self):
        tempdir = TempDirectory()
        try:
            start_dt = pd.Timestamp("2015-11-20 20:11", tz='UTC')
            end_dt = pd.Timestamp(start_dt + timedelta(minutes=10000))

            zeroes_buffer = \
                [0] * int((start_dt -
                           normalize_date(start_dt)).total_seconds() / 60)

            df = pd.DataFrame({
                "open": np.array(zeroes_buffer + list(range(0, 10000))) * 1000,
                "high": np.array(
                    zeroes_buffer + list(range(10000, 20000))) * 1000,
                "low": np.array(
                    zeroes_buffer + list(range(20000, 30000))) * 1000,
                "close": np.array(
                    zeroes_buffer + list(range(30000, 40000))) * 1000,
                "volume": np.array(zeroes_buffer + list(range(40000, 50000)))
            })

            path = os.path.join(tempdir.path, "123.bcolz")
            ctable = bcolz.ctable.fromdataframe(df, rootdir=path)
            ctable.attrs["start_dt"] = start_dt.value / 1e9
            ctable.attrs["last_dt"] = end_dt.value / 1e9

            env = TradingEnvironment()
            env.write_data(futures_data={
                123: {
                    "start_date": normalize_date(start_dt),
                    "end_date": env.next_trading_day(normalize_date(end_dt)),
                    'symbol': 'TEST_FUTURE',
                    'asset_type': 'future',
                }
            })

            future_minute_reader = FutureMinuteReader(tempdir.path)

            dp = DataPortal(
                env,
                future_minute_reader=future_minute_reader
            )

            future123 = env.asset_finder.retrieve_asset(123)

            data_frequency = 'minute'

            for i in range(0, 10000):
                dt = pd.Timestamp(start_dt + timedelta(minutes=i))

                self.assertEqual(i,
                                 dp.get_spot_value(
                                     future123, "open", dt, data_frequency))
                self.assertEqual(i + 10000,
                                 dp.get_spot_value(
                                     future123, "high", dt, data_frequency))
                self.assertEqual(i + 20000,
                                 dp.get_spot_value(
                                     future123, "low", dt, data_frequency))
                self.assertEqual(i + 30000,
                                 dp.get_spot_value(
                                     future123, "close", dt, data_frequency))
                self.assertEqual(i + 40000,
                                 dp.get_spot_value(
                                     future123, "volume", dt, data_frequency))

        finally:
            tempdir.cleanup()
