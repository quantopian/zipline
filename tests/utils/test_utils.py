import os
import pandas as pd
import numpy as np

from zipline.data.data_portal import DataPortal
from zipline.data.minute_writer import MinuteBarWriterFromDataFrames
from .daily_bar_writer import DailyBarWriterFromDataFrames


def create_data_portal(env, tempdir, sim_params, sids):
    if sim_params.data_frequency == "daily":
        path = os.path.join(tempdir.path, "testdaily.bcolz")
        assets = {}
        length = sim_params.days_in_period
        for sid_idx, sid in enumerate(sids):
            assets[sid] = pd.DataFrame({
                "open": (np.array(range(10, 10 + length)) + sid_idx) * 1000,
                "high": (np.array(range(15, 15 + length)) + sid_idx) * 1000,
                "low": (np.array(range(8, 8 + length)) + sid_idx) * 1000,
                "close": (np.array(range(10, 10 + length)) + sid_idx) * 1000,
                "volume": np.array(range(1, length + 1)) + sid_idx,
                "day": [day.value for day in sim_params.trading_days]
            }, index=sim_params.trading_days)

        DailyBarWriterFromDataFrames(assets).write(
            path,
            sim_params.trading_days,
            assets
        )

        return DataPortal(
            env,
            daily_equities_path=path,
            sim_params=sim_params,
            asset_finder=env.asset_finder
        )
    else:
        assets = {}

        minutes = env.minutes_for_days_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        length = len(minutes)

        for sid_idx, sid in enumerate(sids):
            assets[sid] = pd.DataFrame({
                "open": (np.array(range(10, 10 + length)) + sid_idx) * 1000,
                "high": (np.array(range(15, 15 + length)) + sid_idx) * 1000,
                "low": (np.array(range(8, 8 + length)) + sid_idx) * 1000,
                "close": (np.array(range(10, 10 + length)) + sid_idx) * 1000,
                "volume": np.array(range(1, length + 1)) + sid_idx,
                "minute": minutes
            }, index=minutes)

        MinuteBarWriterFromDataFrames().write(tempdir.path, assets)

        return DataPortal(
            env,
            minutes_equities_path=tempdir.path,
            sim_params=sim_params,
            asset_finder=env.asset_finder
        )


def create_data_portal_from_trade_history(env, tempdir, sim_params,
                                          trades_by_sid):
    if sim_params.data_frequency == "daily":
        path = os.path.join(tempdir.path, "testdaily.bcolz")
        assets = {}
        for sidint, trades in trades_by_sid.iteritems():
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            for trade in trades:
                opens.append(trade.open_price)
                highs.append(trade.high)
                lows.append(trade.low)
                closes.append(trade.close_price)
                volumes.append(trade.volume)
            assets[sidint] = pd.DataFrame({
                "open": np.array(opens),
                "high": np.array(highs),
                "low": np.array(lows),
                "close": np.array(closes),
                "volume": np.array(volumes),
                "day": [day.value for day in sim_params.trading_days]
            }, index=sim_params.trading_days)

        DailyBarWriterFromDataFrames(assets).write(
            path,
            sim_params.trading_days,
            assets
        )

        return DataPortal(
            env,
            daily_equities_path=path,
            sim_params=sim_params,
            asset_finder=env.asset_finder
        )
    else:
        assets = {}

        minutes = env.minutes_for_days_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        length = len(minutes)

        for sid_idx, sid in enumerate(sids):
            assets[sid] = pd.DataFrame({
                "open": np.array(range(10, 10 + length)) + sid_idx,
                "high": np.array(range(15, 15 + length)) + sid_idx,
                "low": np.array(range(8, 8 + length)) + sid_idx,
                "close": np.array(range(10, 10 + length)) + sid_idx,
                "volume": np.array(range(1, length + 1)) + sid_idx,
                "minute": minutes
            }, index=minutes)

        MinuteBarWriterFromDataFrames().write(tempdir.path, assets)

        return DataPortal(
            env,
            minutes_equities_path=tempdir.path,
            sim_params=sim_params,
            asset_finder=env.asset_finder
        )
