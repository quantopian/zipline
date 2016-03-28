"""
Synthetic data loaders for testing.
"""
from bcolz import ctable

from numpy import (
    arange,
    array,
    eye,
    float64,
    full,
    iinfo,
    uint32,
)
from numpy.random import RandomState
from pandas import DataFrame, Timestamp
from six import iteritems
from sqlite3 import connect as sqlite3_connect

from .base import PipelineLoader
from .frame import DataFrameLoader
from zipline.data.us_equity_pricing import (
    BcolzDailyBarWriter,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
    US_EQUITY_PRICING_BCOLZ_COLUMNS,
)
from zipline.utils.numpy_utils import (
    bool_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
)


UINT_32_MAX = iinfo(uint32).max


def nanos_to_seconds(nanos):
    return nanos / (1000 * 1000 * 1000)


class PrecomputedLoader(PipelineLoader):
    """
    Synthetic PipelineLoader that uses a pre-computed array for each column.

    Parameters
    ----------
    values : dict
        Map from column to values to use for that column.
        Values can be anything that can be passed as the first positional
        argument to a DataFrame whose indices are ``dates`` and ``sids``
    dates : iterable[datetime-like]
        Row labels for input data.  Can be anything that pd.DataFrame will
        coerce to a DatetimeIndex.
    sids : iterable[int-like]
        Column labels for input data.  Can be anything that pd.DataFrame will
        coerce to an Int64Index.

    Notes
    -----
    Adjustments are unsupported by this loader.
    """
    def __init__(self, constants, dates, sids):
        loaders = {}
        for column, const in iteritems(constants):
            frame = DataFrame(
                const,
                index=dates,
                columns=sids,
                dtype=column.dtype,
            )
            loaders[column] = DataFrameLoader(
                column=column,
                baseline=frame,
                adjustments=None,
            )

        self._loaders = loaders

    def load_adjusted_array(self, columns, dates, assets, mask):
        """
        Load by delegating to sub-loaders.
        """
        out = {}
        for col in columns:
            try:
                loader = self._loaders[col]
            except KeyError:
                raise ValueError("Couldn't find loader for %s" % col)
            out.update(
                loader.load_adjusted_array([col], dates, assets, mask)
            )
        return out


class EyeLoader(PrecomputedLoader):
    """
    A PrecomputedLoader that emits arrays containing 1s on the diagonal and 0s
    elsewhere.

    Parameters
    ----------
    columns : list[BoundColumn]
        Columns that this loader should know about.
    dates : iterable[datetime-like]
        Same as PrecomputedLoader.
    sids : iterable[int-like]
        Same as PrecomputedLoader
    """
    def __init__(self, columns, dates, sids):
        shape = (len(dates), len(sids))
        super(EyeLoader, self).__init__(
            {column: eye(shape, dtype=column.dtype) for column in columns},
            dates,
            sids,
        )


class SeededRandomLoader(PrecomputedLoader):
    """
    A PrecomputedLoader that emits arrays randomly-generated with a given seed.

    Parameters
    ----------
    seed : int
        Seed for numpy.random.RandomState.
    columns : list[BoundColumn]
        Columns that this loader should know about.
    dates : iterable[datetime-like]
        Same as PrecomputedLoader.
    sids : iterable[int-like]
        Same as PrecomputedLoader
    """

    def __init__(self, seed, columns, dates, sids):
        self._seed = seed
        super(SeededRandomLoader, self).__init__(
            {c: self.values(c.dtype, dates, sids) for c in columns},
            dates,
            sids,
        )

    def values(self, dtype, dates, sids):
        """
        Make a random array of shape (len(dates), len(sids)) with ``dtype``.
        """
        shape = (len(dates), len(sids))
        return {
            datetime64ns_dtype: self._datetime_values,
            float64_dtype: self._float_values,
            int64_dtype: self._int_values,
            bool_dtype: self._bool_values,
        }[dtype](shape)

    @property
    def state(self):
        """
        Make a new RandomState from our seed.

        This ensures that every call to _*_values produces the same output
        every time for a given SeededRandomLoader instance.
        """
        return RandomState(self._seed)

    def _float_values(self, shape):
        """
        Return uniformly-distributed floats between -0.0 and 100.0.
        """
        return self.state.uniform(low=0.0, high=100.0, size=shape)

    def _int_values(self, shape):
        """
        Return uniformly-distributed integers between 0 and 100.
        """
        return (self.state.random_integers(low=0, high=100, size=shape)
                .astype('int64'))  # default is system int

    def _datetime_values(self, shape):
        """
        Return uniformly-distributed dates in 2014.
        """
        start = Timestamp('2014', tz='UTC').asm8
        offsets = self.state.random_integers(
            low=0,
            high=364,
            size=shape,
        ).astype('timedelta64[D]')
        return start + offsets

    def _bool_values(self, shape):
        """
        Return uniformly-distributed True/False values.
        """
        return self.state.randn(*shape) < 0


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
        data[:, :5] += arange(5, dtype=uint32) * (10 * 1000)

        # Add days since Jan 1 2001 for OHLCV columns.
        data[:, :5] += (dates - self.PSEUDO_EPOCH).days[:, None].astype(uint32)

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
        writer = SQLiteAdjustmentWriter(conn, None, None)
        empty = DataFrame({
            'sid': array([], dtype=uint32),
            'effective_date': array([], dtype=uint32),
            'ratio': array([], dtype=float),
        })
        empty_dividends = DataFrame({
            'sid': array([], dtype=uint32),
            'amount': array([], dtype=float64),
            'record_date': array([], dtype='datetime64[ns]'),
            'ex_date': array([], dtype='datetime64[ns]'),
            'declared_date': array([], dtype='datetime64[ns]'),
            'pay_date': array([], dtype='datetime64[ns]'),
        })
        writer.write(splits=empty, mergers=empty, dividends=empty_dividends)
        super(NullAdjustmentReader, self).__init__(conn)
