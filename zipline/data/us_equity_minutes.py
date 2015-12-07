from abc import (
    ABCMeta,
    abstractmethod,
)
import json
import os
from bcolz import ctable
from datetime import datetime
import numpy as np
from numpy import float64
from os.path import join
import pandas as pd
from pandas import read_csv
from six import with_metaclass

from zipline.finance.trading import TradingEnvironment
from zipline.utils import tradingcalendar

MINUTES_PER_DAY = 390

_writer_env = TradingEnvironment()

METADATA_FILENAME = 'metadata.json'


def write_metadata(directory, first_trading_day):
    metadata_path = os.path.join(directory, METADATA_FILENAME)

    metadata = {
        'first_trading_day': str(first_trading_day.date())
    }

    with open(metadata_path, 'w') as fp:
        json.dump(metadata, fp)


class BcolzMinuteBarWriter(with_metaclass(ABCMeta)):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.
    """
    @property
    def first_trading_day(self):
        return self._first_trading_day

    @abstractmethod
    def gen_frames(self, assets):
        """
        Return an iterator of pairs of (asset_id, pd.dataframe).
        """
        raise NotImplementedError()

    def write(self, directory, assets, sid_path_func=None):
        _iterator = self.gen_frames(assets)

        return self._write_internal(directory, _iterator,
                                    sid_path_func=sid_path_func)

    @staticmethod
    def full_minutes_for_days(env, dt1, dt2):
        start_date = env.normalize_date(dt1)
        end_date = env.normalize_date(dt2)

        all_minutes = []

        for day in env.days_in_range(start_date, end_date):
            minutes_in_day = pd.date_range(
                start=pd.Timestamp(
                    datetime(
                        year=day.year,
                        month=day.month,
                        day=day.day,
                        hour=9,
                        minute=31),
                    tz='US/Eastern').tz_convert('UTC'),
                periods=390,
                freq="min"
            )

            all_minutes.append(minutes_in_day)

        # flatten
        return pd.DatetimeIndex(
            np.concatenate(all_minutes), copy=False, tz='UTC'
        )

    def _write_internal(self, directory, iterator, sid_path_func=None):
        first_trading_day = self.first_trading_day

        write_metadata(directory, first_trading_day)

        first_open = pd.Timestamp(
            datetime(
                year=first_trading_day.year,
                month=first_trading_day.month,
                day=first_trading_day.day,
                hour=9,
                minute=31
            ), tz='US/Eastern').tz_convert('UTC')

        for asset_id, df in iterator:
            if sid_path_func is None:
                path = join(directory, "{0}.bcolz".format(asset_id))
            else:
                path = sid_path_func(directory, asset_id)

            os.makedirs(path)

            minutes = self.full_minutes_for_days(_writer_env,
                                                 first_open, df.index[-1])
            minutes_count = len(minutes)

            dt_col = np.zeros(minutes_count, dtype=np.uint32)
            open_col = np.zeros(minutes_count, dtype=np.uint32)
            high_col = np.zeros(minutes_count, dtype=np.uint32)
            low_col = np.zeros(minutes_count, dtype=np.uint32)
            close_col = np.zeros(minutes_count, dtype=np.uint32)
            vol_col = np.zeros(minutes_count, dtype=np.uint32)

            for row in df.iterrows():
                dt = row[0]
                idx = minutes.searchsorted(dt)

                dt_col[idx] = dt.value / 1e9
                open_col[idx] = row[1].loc["open"]
                high_col[idx] = row[1].loc["high"]
                low_col[idx] = row[1].loc["low"]
                close_col[idx] = row[1].loc["close"]
                vol_col[idx] = row[1].loc["volume"]

            ctable(
                columns=[
                    open_col,
                    high_col,
                    low_col,
                    close_col,
                    vol_col,
                    dt_col
                ],
                names=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "dt"
                ],
                rootdir=path,
                mode='w'
            )


class MinuteBarWriterFromDataFrames(BcolzMinuteBarWriter):
    _csv_dtypes = {
        'open': float64,
        'high': float64,
        'low': float64,
        'close': float64,
        'volume': float64,
    }

    def __init__(self, first_trading_day):
        self._first_trading_day = first_trading_day

    def gen_frames(self, assets):
        for asset in assets:
            df = assets[asset]
            yield asset, df.set_index("minute")


class MinuteBarWriterFromCSVs(BcolzMinuteBarWriter):
    """
    BcolzMinuteBarWriter constructed from a map of CSVs to assets.

    Parameters
    ----------
    asset_map: dict
        A map from asset_id -> path to csv with data for that asset.

    CSVs should have the following columns:
        minute : datetime64
        open : float64
        high : float64
        low : float64
        close : float64
        volume : int64
    """
    _csv_dtypes = {
        'open': float64,
        'high': float64,
        'low': float64,
        'close': float64,
        'volume': float64,
    }

    def __init__(self, asset_map, first_trading_day):
        self._asset_map = asset_map
        self._first_trading_day = first_trading_day

    def gen_frames(self, assets):
        """
        Read CSVs as DataFrames from our asset map.
        """
        dtypes = self._csv_dtypes

        for asset in assets:
            path = self._asset_map.get(asset)
            if path is None:
                raise KeyError("No path supplied for asset %s" % asset)
            df = read_csv(path, parse_dates=['minute'], dtype=dtypes)
            df = df.set_index("minute").tz_localize("UTC")

            yield asset, df


class BcolzMinuteBarReader(object):

    def __init__(self, rootdir, sid_path_func=None):
        self.rootdir = rootdir

        metadata = self._get_metadata()

        self.first_trading_day = pd.Timestamp(
            metadata['first_trading_day'], tz='UTC')
        mask = tradingcalendar.trading_days.slice_indexer(
            self.first_trading_day)
        self.trading_days = tradingcalendar.trading_days[mask]
        self.sid_path_func = sid_path_func

    def _get_metadata(self):
        with open(os.path.join(self.rootdir, METADATA_FILENAME)) as fp:
            return json.load(fp)
