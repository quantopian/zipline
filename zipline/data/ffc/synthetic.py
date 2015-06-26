"""
Synthetic data loaders for testing.
"""
from abc import abstractmethod
from bcolz import ctable

from numpy import (
    arange,
    array,
    full,
    iinfo,
    uint32,
)
from pandas import (
    DataFrame,
    Timestamp,
)
from sqlite3 import connect as sqlite3_connect

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)
from zipline.data.ffc.base import FFCLoader
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarWriter,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
)


UINT_32_MAX = iinfo(uint32).max


def nanos_to_seconds(nanos):
    return nanos / (1000 * 1000 * 1000)


class SyntheticDataLoader(FFCLoader):
    """
    DataLoader subclass that builds synthetic data based only on the shape of
    the desired output.  Keeps a log of all calls to load_columns() for use in
    testing.
    """
    def __init__(self, known_assets, adjustments):
        """
        known_assets: #TODO: Explain
        adjustments: #TODO:  Explain
        """
        self._log = []
        self._known_assets = known_assets
        self._adjustments = {}

    def _adjustments_for_dates(self, column, dates):
        adjustments = self._adjustments.get(column, {})
        out = {}
        for idx, dt in enumerate(dates):
            adjustments_for_dt = adjustments.get(dt, None)
            if adjustments_for_dt is not None:
                out[dt] = adjustments_for_dt
        return out

    def load_adjusted_array(self, columns, dates, assets):
        """
        Load each column with self.make_column.
        """
        self._log.append((columns, dates, assets))

        nrows = len(dates)
        ncols = len(assets)
        return [
            adjusted_array(
                self.make_baseline(col, nrows, ncols),
                NOMASK,
                self._adjustments_for_dates(col, dates),
            )
            for col in columns
        ]

    @abstractmethod
    def make_baseline(self, column, nrows, ncols):
        """
        Returns an ndarray of shape nrows, ncols for the given column.
        """
        pass


class ConstantLoader(SyntheticDataLoader):
    """
    SyntheticDataLoader that returns a constant value for each column.
    """

    def __init__(self, known_assets, adjustments, constants):
        super(ConstantLoader, self).__init__(
            known_assets=known_assets,
            adjustments=adjustments,
        )
        self._constants = constants

    def make_baseline(self, column, nrows, ncols):
        return full(
            (nrows, ncols),
            self._constants[column],
            dtype=column.dtype,
        )


class SyntheticDailyBarWriter(BcolzDailyBarWriter):
    """
    Bcolz writer that creates synthetic data based on asset lifetime metadata.

    For a given asset/date/column combination, we generate a corresponding raw
    value using the formula:

    data(asset, date, column) = (100,000 * asset_id)
                              + (10,000 * column_num)
                              + (date - Jan 1 2000).days  # ~6000 for 2015
    where:
        column_num('open') = 0
        column_num('high') = 1
        column_num('low') = 2
        column_num('close') = 3
        column_num('volume') = 4

    We use days since Jan 1, 2000 to guarantee that there are no collisions
    while also the produced values smaller than UINT32_MAX / 1000.

    Parameters
    ----------
    asset_info : DataFrame
        DataFrame with asset_id as index and 'start_date'/'end_date' columns.
    calendar : DatetimeIndex
        Dates to use to compute offsets.
    """
    OHLCV = ('open', 'high', 'low', 'close', 'volume')
    PSEUDO_EPOCH = Timestamp('2000-01-01', tz='UTC')

    def __init__(self, asset_info, calendar):
        super(SyntheticDailyBarWriter, self).__init__()
        self._asset_info = asset_info
        self._frames = {}
        for asset_id in asset_info.index:
            start, end = asset_info.ix[asset_id, ['start_date', 'end_date']]
            asset_dates = calendar[
                calendar.get_loc(start):calendar.get_loc(end) + 1
            ]

            opens, highs, lows, closes, volumes = self._make_raw_data(
                asset_id,
                asset_dates,
            )
            days = asset_dates.asi8
            ids = full((len(asset_dates),), asset_id, dtype=uint32)
            self._frames[asset_id] = DataFrame(
                {
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes,
                    'id': ids,
                    'day': days,
                },
            )

    def _make_raw_data(self, asset_id, asset_dates):
        """
        Generate 'raw' data that encodes information about the asset.

        See class docstring for a description of the data format.
        """
        assert asset_dates[0] > self.PSEUDO_EPOCH
        OHLCV_COLUMN_COUNT = len(self.OHLCV)
        data = full(
            (len(asset_dates), OHLCV_COLUMN_COUNT),
            asset_id * (100 * 1000),
            dtype=uint32,
        )

        # Add 10,000 * column-index to each column.
        data += arange(OHLCV_COLUMN_COUNT) * (10 * 1000)

        # Add days since Jan 1 2001 for each row.
        data += (asset_dates - self.PSEUDO_EPOCH).days[:, None]

        # Return data as iterable of column arrays.
        return (data[:, i] for i in range(OHLCV_COLUMN_COUNT))

    @classmethod
    def expected_value(cls, asset_id, date, colname):
        """
        Check that the raw value for an asset/date/column triple is as
        expected.

        Used by tests to verify data written by a writer.
        """
        from_asset = asset_id * 100 * 1000
        from_colname = cls.OHLCV.index(colname) * (10 * 1000)
        from_date = (date - cls.PSEUDO_EPOCH).days
        return from_asset + from_colname + from_date

    # BEGIN SUPERCLASS INTERFACE
    def gen_ctables(self, dates, assets):
        for asset in assets:
            # Clamp stored data to the requested date range.
            frame = self._frames[asset].reset_index()
            yield asset, ctable.fromdataframe(frame)

    def to_timestamp(self, raw_dt):
        return Timestamp(raw_dt)

    def to_uint32(self, array, colname):
        if colname == 'day':
            return nanos_to_seconds(array)
        elif colname in {'open', 'high', 'low', 'close'}:
            # Data is stored as 1000 * raw value.
            assert array.max() < (UINT_32_MAX / 1000), "Test data overflow!"
            return array * 1000
        else:
            assert colname == 'volume', "Unknown column: %s" % colname
            return array
    # END SUPERCLASS INTERFACE


class NullAdjustmentReader(SQLiteAdjustmentReader):
    """
    A SQLiteAdjustmentReader that stores no adjustments and uses in-memory
    SQLite.
    """

    def __init__(self):
        conn = sqlite3_connect(':memory:')
        writer = SQLiteAdjustmentWriter(conn)
        empty = DataFrame({
            'sid': array([], dtype=uint32),
            'effective_date': array([], dtype=uint32),
            'ratio': array([], dtype=float),
        })
        writer.write(splits=empty, mergers=empty, dividends=empty)
        super(NullAdjustmentReader, self).__init__(conn)
