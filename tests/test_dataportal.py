import os
import numpy as np
import pandas as pd

from unittest import TestCase
from testfixtures import TempDirectory
from zipline.data.data_portal import DataPortal
from zipline.finance.trading import TradingEnvironment, SimulationParameters
from zipline.data.minute_writer import MinuteBarWriterFromDataFrames
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
                "open": np.array([0] + range(0, 200) + [0] * 579) * 1000,
                "high": np.array([0] + range(1000, 1200) + [0] * 579) * 1000,
                "low": np.array([0] + range(2000, 2200) + [0] * 579) * 1000,
                "close": np.array([0] + range(3000, 3200) + [0] * 579) * 1000,
                "volume": [0] + range(4000, 4200) + [0] * 579,
                "minute": minutes
            })

            MinuteBarWriterFromDataFrames().write(tempdir.path, {0: df})

            sim_params = SimulationParameters(
                period_start=minutes[0],
                period_end=minutes[-1],
                data_frequency="minute"
            )

            dp = DataPortal(
                env,
                minutes_equities_path=tempdir.path,
                sim_params=sim_params,
                asset_finder=env.asset_finder
            )

            for minute_idx, minute in enumerate(minutes):
                for field_idx, field in enumerate(
                        ["open", "high", "low", "close", "volume"]):
                    val = dp.get_current_price_data(0, field, dt=minute)
                    if minute_idx < 200:
                        self.assertEqual(minute_idx + (field_idx * 1000), val)
                    else:
                        self.assertEqual(199 + (field_idx * 1000), val)
        finally:
            tempdir.cleanup()

    def forward_fill_daily(self):
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
                "open": [0] + range(0, 8) + [0] * 8,
                "high": [0] + range(10, 18) + [0] * 8,
                "low": [0] + range(20, 28) + [0] * 8,
                "close": [0] + range(30, 38) + [0] * 8,
                "volume": [0] + range(40, 48) + [0] * 8,
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
                sim_params=sim_params,
                asset_finder=env.asset_finder
            )

            for day_idx, day in enumerate(days):
                for field_idx, field in enumerate(
                        ["open", "high", "low", "close", "volume"]):
                    val = dp.get_current_price_data(0, field, dt=day)
                    if day_idx == 0:
                        self.assertEqual(0, val)
                    elif day_idx < 9:
                        self.assertEqual((day_idx - 1) + (field_idx * 10), val)
                    else:
                        self.assertEqual(7 + (field_idx * 10), val)
        finally:
            tempdir.cleanup()