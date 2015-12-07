import os
from datetime import timedelta
import bcolz
import numpy as np
import pandas as pd

from unittest import TestCase
from pandas.tslib import normalize_date
from testfixtures import TempDirectory
from zipline.data.data_portal import DataPortal
from zipline.data.us_equity_pricing import SQLiteAdjustmentWriter, \
    SQLiteAdjustmentReader
from zipline.finance.trading import TradingEnvironment, SimulationParameters
from zipline.data.us_equity_minutes import (
    MinuteBarWriterFromDataFrames,
    BcolzMinuteBarReader
)
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
                "open": np.array([0] + list(range(0, 200)) + [0] * 579)
                * 1000,
                "high": np.array([0] + list(range(1000, 1200)) + [0] * 579)
                * 1000,
                "low": np.array([0] + list(range(2000, 2200)) + [0] * 579)
                * 1000,
                "close": np.array([0] + list(range(3000, 3200)) + [0] * 579)
                * 1000,
                "volume": [0] + list(range(4000, 4200)) + [0] * 579,
                "minute": minutes
            })

            MinuteBarWriterFromDataFrames(
                pd.Timestamp('2002-01-02', tz='UTC')).write(
                    tempdir.path, {0: df})

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
                sim_params=sim_params
            )

            for minute_idx, minute in enumerate(minutes):
                for field_idx, field in enumerate(
                        ["open", "high", "low", "close", "volume"]):
                    val = dp.get_spot_value(0, field, dt=minute)
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

            dp = DataPortal(
                env,
                daily_equities_path=path,
                sim_params=sim_params
            )

            for day_idx, day in enumerate(days):
                for field_idx, field in enumerate(
                        ["open", "high", "low", "close", "volume"]):
                    val = dp.get_spot_value(0, field, dt=day)
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
                                 list(range(390, 680))) * 1000,
                "high": np.array(list(range(1000, 1390)) + [0] * 100 +
                                 list(range(1390, 1680))) * 1000,
                "low": np.array(list(range(2000, 2390)) + [0] * 100 +
                                list(range(2390, 2680))) * 1000,
                "close": np.array(list(range(3000, 3390)) + [0] * 100 +
                                  list(range(3390, 3680))) * 1000,
                "volume": np.array(list(range(4000, 4390)) + [0] * 100 +
                                   list(range(4390, 4680))),
                "minute": minutes
            })

            MinuteBarWriterFromDataFrames(
                pd.Timestamp('2002-01-02', tz='UTC')).write(
                    tempdir.path, {0: df})

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
                sim_params=sim_params,
                adjustment_reader=SQLiteAdjustmentReader(adjustments_path)
            )

            # phew, finally ready to start testing.
            for idx, minute in enumerate(minutes[:390]):
                for field_idx, field in enumerate(["open", "high", "low",
                                                   "close", "volume"]):
                    self.assertEqual(
                        dp.get_spot_value(0, field, dt=minute),
                        idx + (1000 * field_idx)
                    )

            for idx, minute in enumerate(minutes[390:490]):
                # no actual data for this part, so we'll forward-fill.
                # make sure the forward-filled values are adjusted.
                for field_idx, field in enumerate(["open", "high", "low",
                                                   "close"]):
                    self.assertEqual(
                        dp.get_spot_value(0, field, dt=minute),
                        (389 + (1000 * field_idx)) / 2.0
                    )

                self.assertEqual(
                    dp.get_spot_value(0, "volume", dt=minute),
                    8778  # 4389 * 2
                )

            for idx, minute in enumerate(minutes[490:]):
                # back to real data
                for field_idx, field in enumerate(["open", "high", "low",
                                                   "close", "volume"]):
                    self.assertEqual(
                        dp.get_spot_value(0, field, dt=minute),
                        (390 + idx + (1000 * field_idx))
                    )
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

            dp = DataPortal(
                env,
                minutes_futures_path=tempdir.path
            )

            future123 = env.asset_finder.retrieve_asset(123)

            for i in range(0, 10000):
                dt = pd.Timestamp(start_dt + timedelta(minutes=i))

                self.assertEqual(i,
                                 dp.get_spot_value(future123, "open", dt))
                self.assertEqual(i + 10000,
                                 dp.get_spot_value(future123, "high", dt))
                self.assertEqual(i + 20000,
                                 dp.get_spot_value(future123, "low", dt))
                self.assertEqual(i + 30000,
                                 dp.get_spot_value(future123, "close", dt))
                self.assertEqual(i + 40000,
                                 dp.get_spot_value(future123, "volume", dt))

        finally:
            tempdir.cleanup()
