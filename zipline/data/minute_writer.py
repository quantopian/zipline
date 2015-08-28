from abc import (
    ABCMeta,
    abstractmethod,
)
from bcolz import ctable
from datetime import datetime
from functools32 import lru_cache
import numpy as np
from numpy import float64
from os.path import join
import pandas as pd
from pandas import read_csv
from six import with_metaclass

from zipline.finance.trading import TradingEnvironment
from zipline.utils import tradingcalendar

MINUTES_FOR_DAY = 390


class BcolzMinuteBarWriter(with_metaclass(ABCMeta)):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.
    """
    @abstractmethod
    def gen_frames(self, assets):
        """
        Return an iterator of pairs of (asset_id, pd.dataframe).
        """
        raise NotImplementedError()

    def write(self, directory, assets):
        _iterator = self.gen_frames(assets)

        return self._write_internal(directory, _iterator)

    def full_minutes_for_days(self, dt1, dt2):
        cols = []
        days = tradingcalendar.get_trading_days(dt1, dt2)
        for day in days:
            cols.append(self.get_minutes_col_for_day(day))

        return np.hstack(cols)

    @lru_cache(maxsize=None)
    def get_minutes_col_for_day(self, day):
        """
        Get the market minutes as uint32 seconds since the EPOCH

        This is called once for the processing of every day, since all sids
        will share this value.
        """
        minutes = (TradingEnvironment.instance().market_minutes_for_day(day).asi8 / int(1e9)).\
            astype(np.uint32)

        num_to_add = MINUTES_PER_DAY - len(minutes)
        # Fill the missing minutes with corresponding market minutes on half
        # days, not using 0's so that searchsorted still works.
        if num_to_add:
            last_market_minute = minutes[-1]
            # Calculate the missing minutes by adding 60 seconds to the last
            # market minute returned by the zipline calendar then repeating
            # a range every 60 seconds.
            to_add = np.arange(last_market_minute + 60,
                               last_market_minute + (num_to_add + 1) * 60,
                               60,
                               dtype=np.uint32)
            assert num_to_add == len(to_add)
            minutes = np.append(minutes, to_add)
        return minutes

    def _write_internal(self, directory, iterator):
        first_open = pd.Timestamp(
            datetime(
                year=2002,
                month=1,
                day=2,
                hour=9,
                minute=31
            ), tz='US/Eastern').tz_convert('UTC')

        for asset_id, df in iterator:
            path = join(directory, "{0}.bcolz".format(asset_id))

            minutes = self.full_minutes_for_days(first_open, df.index[-1])

            import pdb; pdb.set_trace()
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
                open_col[idx] = row[1].iloc[0]
                high_col[idx] = row[1].iloc[1]
                low_col[idx] = row[1].iloc[2]
                close_col[idx] = row[1].iloc[3]
                vol_col[idx] = row[1].iloc[4]

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
                    "vol",
                    "dt"
                ],
                rootdir=path,
                mode='w'
            )


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

    def __init__(self, asset_map):
        self._asset_map = asset_map

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
