"""
Synthetic data loaders for testing.
"""
from bcolz import ctable

from numpy import (
    arange,
    array,
    float64,
    full,
    iinfo,
    uint32,
)
from pandas import (
    DataFrame,
    Timestamp,
)
from sqlite3 import connect as sqlite3_connect

from six import iteritems

from zipline.data.ffc.base import FFCLoader
from zipline.data.ffc.frame import DataFrameFFCLoader
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarWriter,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
    US_EQUITY_PRICING_BCOLZ_COLUMNS,
)


UINT_32_MAX = iinfo(uint32).max


def nanos_to_seconds(nanos):
    return nanos / (1000 * 1000 * 1000)


class MultiColumnLoader(FFCLoader):
    """
    FFCLoader that can delegate to sub-loaders.

    Parameters
    ----------
    loaders : dict
        Dictionary mapping columns -> loader
    """
    def __init__(self, loaders):
        self._loaders = loaders

    def load_adjusted_array(self, columns, mask):
        """
        Load by delegating to sub-loaders.
        """
        out = []
        for column in columns:
            try:
                loader = self._loaders[column]
            except KeyError:
                raise ValueError("Couldn't find loader for %s" % column)
            out.extend(loader.load_adjusted_array([column], mask))
        return out


class ConstantLoader(MultiColumnLoader):
    """
    Synthetic FFCLoader that returns a constant value for each column.

    Parameters
    ----------
    constants : dict
        Map from column to value(s) to use for that column.
        Values can be anything that can be passed as the first positional
        argument to a DataFrame of the same shape as `mask`.
    mask : pandas.DataFrame
        Mask indicating when assets existed.
        Indices of this frame are used to align input queries.

    Notes
    -----
    Adjustments are unsupported with ConstantLoader.
    """
    def __init__(self, constants, dates, assets):
        loaders = {}
        for column, const in iteritems(constants):
            frame = DataFrame(
                const,
                index=dates,
                columns=assets,
                dtype=column.dtype,
            )
            loaders[column] = DataFrameFFCLoader(
                column=column,
                baseline=frame,
                adjustments=None,
            )

        super(ConstantLoader, self).__init__(loaders)


class SyntheticDailyBarWriter(BcolzDailyBarWriter):
    """
    Bcolz writer that creates synthetic data based on asset lifetime metadata.

    For a given asset/date/column combination, we generate a corresponding raw
    value using the following formula for OHLCV columns:

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

    For 'day' and 'id', we use the standard format expected by the base class.

    Parameters
    ----------
    asset_info : DataFrame
        DataFrame with asset_id as index and 'start_date'/'end_date' columns.
    calendar : DatetimeIndex
        Calendar to use for constructing asset lifetimes.
    """
    OHLCV = ('open', 'high', 'low', 'close', 'volume')
    OHLC = ('open', 'high', 'low', 'close')
    PSEUDO_EPOCH = Timestamp('2000-01-01', tz='UTC')

    def __init__(self, asset_info, calendar):
        super(SyntheticDailyBarWriter, self).__init__()
        assert (
            # Using .value here to avoid having to care about UTC-aware dates.
            self.PSEUDO_EPOCH.value <
            calendar.min().value <=
            asset_info['start_date'].min().value
        )
        assert (asset_info['start_date'] < asset_info['end_date']).all()
        self._asset_info = asset_info
        self._calendar = calendar

    def _raw_data_for_asset(self, asset_id):
        """
        Generate 'raw' data that encodes information about the asset.

        See class docstring for a description of the data format.
        """
        # Get the dates for which this asset existed according to our asset
        # info.
        dates = self._calendar[
            self._calendar.slice_indexer(
                self.asset_start(asset_id), self.asset_end(asset_id)
            )
        ]

        data = full(
            (len(dates), len(US_EQUITY_PRICING_BCOLZ_COLUMNS)),
            asset_id * (100 * 1000),
            dtype=uint32,
        )

        # Add 10,000 * column-index to OHLCV columns
        data[:, :5] += arange(5) * (10 * 1000)

        # Add days since Jan 1 2001 for OHLCV columns.
        data[:, :5] += (dates - self.PSEUDO_EPOCH).days[:, None]

        frame = DataFrame(
            data,
            index=dates,
            columns=US_EQUITY_PRICING_BCOLZ_COLUMNS,
        )

        frame['day'] = nanos_to_seconds(dates.asi8)
        frame['id'] = asset_id

        return ctable.fromdataframe(frame)

    def asset_start(self, asset):
        ret = self._asset_info.loc[asset]['start_date']
        if ret.tz is None:
            ret = ret.tz_localize('UTC')
        assert ret.tzname() == 'UTC', "Unexpected non-UTC timestamp"
        return ret

    def asset_end(self, asset):
        ret = self._asset_info.loc[asset]['end_date']
        if ret.tz is None:
            ret = ret.tz_localize('UTC')
        assert ret.tzname() == 'UTC', "Unexpected non-UTC timestamp"
        return ret

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

    def expected_values_2d(self, dates, assets, colname):
        """
        Return an 2D array containing cls.expected_value(asset_id, date,
        colname) for each date/asset pair in the inputs.

        Values before/after an assets lifetime are filled with 0 for volume and
        NaN for price columns.
        """
        if colname == 'volume':
            dtype = uint32
            missing = 0
        else:
            dtype = float64
            missing = float('nan')

        data = full((len(dates), len(assets)), missing, dtype=dtype)
        for j, asset in enumerate(assets):
            start, end = self.asset_start(asset), self.asset_end(asset)
            for i, date in enumerate(dates):
                # No value expected for dates outside the asset's start/end
                # date.
                if not (start <= date <= end):
                    continue
                data[i, j] = self.expected_value(asset, date, colname)
        return data

    # BEGIN SUPERCLASS INTERFACE
    def gen_tables(self, assets):
        for asset in assets:
            yield asset, self._raw_data_for_asset(asset)

    def to_uint32(self, array, colname):
        if colname in {'open', 'high', 'low', 'close'}:
            # Data is stored as 1000 * raw value.
            assert array.max() < (UINT_32_MAX / 1000), "Test data overflow!"
            return array * 1000
        else:
            assert colname in ('volume', 'day'), "Unknown column: %s" % colname
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
