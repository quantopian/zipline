from numpy import (
    float64,
    uint32,
    int64,
)
from bcolz import ctable

from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter, OHLC, UINT32_MAX


class DailyBarWriterFromDataFrames(BcolzDailyBarWriter):
    _csv_dtypes = {
        "open": float64,
        "high": float64,
        "low": float64,
        "close": float64,
        "volume": float64,
    }

    def __init__(self, asset_map):
        self._asset_map = asset_map

    def gen_tables(self, assets):
        for asset in assets:
            yield asset, ctable.fromdataframe(assets[asset])

    def to_uint32(self, array, colname):
        arrmax = array.max()
        if colname in OHLC:
            self.check_uint_safe(arrmax * 1000, colname)
            return (array * 1000).astype(uint32)
        elif colname == "volume":
            self.check_uint_safe(arrmax, colname)
            return array.astype(uint32)
        elif colname == "day":
            nanos_per_second = 1000 * 1000 * 1000
            self.check_uint_safe(arrmax.view(int64) / nanos_per_second, colname)
            return (array.view(int64) / nanos_per_second).astype(uint32)

    @staticmethod
    def check_uint_safe(value, colname):
        if value >= UINT32_MAX:
            raise ValueError(
                "Value %s from column '%s' is too large" % (value, colname)
            )
