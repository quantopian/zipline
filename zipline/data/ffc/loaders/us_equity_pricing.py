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
from abc import (
    ABCMeta,
    abstractmethod,
)
from contextlib import contextmanager
from errno import ENOENT
from os import remove
from os.path import exists

from bcolz import (
    carray,
    ctable,
)
from click import progressbar
from numpy import (
    array,
    float64,
    floating,
    full,
    iinfo,
    integer,
    issubdtype,
    uint32,
)
from pandas import (
    DatetimeIndex,
    read_csv,
    Timestamp,
)
from six import (
    iteritems,
    string_types,
    with_metaclass,
)
import sqlite3


from zipline.data.ffc.base import FFCLoader
from zipline.data.ffc.loaders._equities import (
    _compute_row_slices,
    _read_bcolz_data,
)
from zipline.data.ffc.loaders._adjustments import load_adjustments_from_sqlite
from zipline.lib.adjusted_array import (
    adjusted_array,
)
from zipline.errors import NoFurtherDataError

OHLC = frozenset(['open', 'high', 'low', 'close'])
US_EQUITY_PRICING_BCOLZ_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'day', 'id'
]
DAILY_US_EQUITY_PRICING_DEFAULT_FILENAME = 'daily_us_equity_pricing.bcolz'
SQLITE_ADJUSTMENT_COLUMNS = frozenset(['effective_date', 'ratio', 'sid'])
SQLITE_ADJUSTMENT_COLUMN_DTYPES = {
    'effective_date': integer,
    'ratio': floating,
    'sid': integer,
}
SQLITE_ADJUSTMENT_TABLENAMES = frozenset(['splits', 'dividends', 'mergers'])

UINT32_MAX = iinfo(uint32).max


@contextmanager
def passthrough(obj):
    yield obj


class BcolzDailyBarWriter(with_metaclass(ABCMeta)):
    """
    Class capable of writing daily OHLCV data to disk in a format that can be
    read efficiently by BcolzDailyOHLCVReader.

    See Also
    --------
    BcolzDailyBarReader : Consumer of the data written by this class.
    """

    @abstractmethod
    def gen_tables(self, assets):
        """
        Return an iterator of pairs of (asset_id, bcolz.ctable).
        """
        raise NotImplementedError()

    @abstractmethod
    def to_uint32(self, array, colname):
        """
        Convert raw column values produced by gen_tables into uint32 values.

        Parameters
        ----------
        array : np.array
            An array of raw values.
        colname : str, {'open', 'high', 'low', 'close', 'volume', 'day'}
            The name of the column being loaded.

        For output being read by the default BcolzOHLCVReader, data should be
        stored in the following manner:

        - Pricing columns (Open, High, Low, Close) should be stored as 1000 *
          as-traded dollar value.
        - Volume should be the as-traded volume.
        - Dates should be stored as seconds since midnight UTC, Jan 1, 1970.
        """
        raise NotImplementedError()

    def write(self, filename, calendar, assets, show_progress=False):
        """
        Parameters
        ----------
        filename : str
            The location at which we should write our output.
        calendar : pandas.DatetimeIndex
            Calendar to use to compute asset calendar offsets.
        assets : pandas.Int64Index
            The assets for which to write data.
        show_progress : bool
            Whether or not to show a progress bar while writing.

        Returns
        -------
        table : bcolz.ctable
            The newly-written table.
        """
        _iterator = self.gen_tables(assets)
        if show_progress:
            pbar = progressbar(
                _iterator,
                length=len(assets),
                item_show_func=lambda i: i if i is None else str(i[0]),
                label="Merging asset files:",
            )
            with pbar as pbar_iterator:
                return self._write_internal(filename, calendar, pbar_iterator)
        return self._write_internal(filename, calendar, _iterator)

    def _write_internal(self, filename, calendar, iterator):
        """
        Internal implementation of write.

        `iterator` should be an iterator yielding pairs of (asset, ctable).
        """
        total_rows = 0
        first_row = {}
        last_row = {}
        calendar_offset = {}

        # Maps column name -> output carray.
        columns = {
            k: carray(array([], dtype=uint32))
            for k in US_EQUITY_PRICING_BCOLZ_COLUMNS
        }

        for asset_id, table in iterator:
            nrows = len(table)
            for column_name in columns:
                if column_name == 'id':
                    # We know what the content of this column is, so don't
                    # bother reading it.
                    columns['id'].append(full((nrows,), asset_id))
                    continue
                columns[column_name].append(
                    self.to_uint32(table[column_name][:], column_name)
                )

            # Bcolz doesn't support ints as keys in `attrs`, so convert
            # assets to strings for use as attr keys.
            asset_key = str(asset_id)

            # Calculate the index into the array of the first and last row
            # for this asset. This allows us to efficiently load single
            # assets when querying the data back out of the table.
            first_row[asset_key] = total_rows
            last_row[asset_key] = total_rows + nrows - 1
            total_rows += nrows

            # Calculate the number of trading days between the first date
            # in the stored data and the first date of **this** asset. This
            # offset used for output alignment by the reader.

            # HACK: Index with a list so that we get back an array we can pass
            # to self.to_uint32.  We could try to extract this in the loop
            # above, but that makes the logic a lot messier.
            asset_first_day = self.to_uint32(table['day'][[0]], 'day')[0]
            calendar_offset[asset_key] = calendar.get_loc(
                Timestamp(asset_first_day, unit='s', tz='UTC'),
            )

        # This writes the table to disk.
        full_table = ctable(
            columns=[
                columns[colname]
                for colname in US_EQUITY_PRICING_BCOLZ_COLUMNS
            ],
            names=US_EQUITY_PRICING_BCOLZ_COLUMNS,
            rootdir=filename,
            mode='w',
        )
        full_table.attrs['first_row'] = first_row
        full_table.attrs['last_row'] = last_row
        full_table.attrs['calendar_offset'] = calendar_offset
        full_table.attrs['calendar'] = calendar.asi8.tolist()
        return full_table


class DailyBarWriterFromCSVs(BcolzDailyBarWriter):
    """
    BcolzDailyBarWriter constructed from a map from csvs to assets.

    Parameters
    ----------
    asset_map : dict
        A map from asset_id -> path to csv with data for that asset.

    CSVs should have the following columns:
        day : datetime64
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

    def gen_tables(self, assets):
        """
        Read CSVs as DataFrames from our asset map.
        """
        dtypes = self._csv_dtypes
        for asset in assets:
            path = self._asset_map.get(asset)
            if path is None:
                raise KeyError("No path supplied for asset %s" % asset)
            data = read_csv(path, parse_dates=['day'], dtype=dtypes)
            yield asset, ctable.fromdataframe(data)

    def to_uint32(self, array, colname):
        arrmax = array.max()
        if colname in OHLC:
            self.check_uint_safe(arrmax * 1000, colname)
            return (array * 1000).astype(uint32)
        elif colname == 'volume':
            self.check_uint_safe(arrmax, colname)
            return array.astype(uint32)
        elif colname == 'day':
            nanos_per_second = (1000 * 1000 * 1000)
            self.check_uint_safe(arrmax.view(int) / nanos_per_second, colname)
            return (array.view(int) / nanos_per_second).astype(uint32)

    @staticmethod
    def check_uint_safe(value, colname):
        if value >= UINT32_MAX:
            raise ValueError(
                "Value %s from column '%s' is too large" % (value, colname)
            )


class BcolzDailyBarReader(object):
    """
    Reader for raw pricing data written by BcolzDailyOHLCVWriter.

    A Bcolz CTable is comprised of Columns and Attributes.

    Columns
    -------
    The table with which this loader interacts contains the following columns:

    ['open', 'high', 'low', 'close', 'volume', 'day', 'id'].

    The data in these columns is interpreted as follows:

    - Price columns ('open', 'high', 'low', 'close') are interpreted as 1000 *
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    - Id is the asset id of the row.

    The data in each column is grouped by asset and then sorted by day within
    each asset block.

    The table is built to represent a long time range of data, e.g. ten years
    of equity data, so the lengths of each asset block is not equal to each
    other. The blocks are clipped to the known start and end date of each asset
    to cut down on the number of empty values that would need to be included to
    make a regular/cubic dataset.

    When read across the open, high, low, close, and volume with the same
    index should represent the same asset and day.

    Attributes
    ----------
    The table with which this loader interacts contains the following
    attributes:

    first_row : dict
        Map from asset_id -> index of first row in the dataset with that id.
    last_row : dict
        Map from asset_id -> index of last row in the dataset with that id.
    calendar_offset : dict
        Map from asset_id -> calendar index of first row.
    calendar : list[int64]
        Calendar used to compute offsets, in asi8 format (ns since EPOCH).

    We use first_row and last_row together to quickly find ranges of rows to
    load when reading an asset's data into memory.

    We use calendar_offset and calendar to orient loaded blocks within a
    range of queried dates.
    """
    def __init__(self, table):
        if isinstance(table, string_types):
            table = ctable(rootdir=table, mode='r')

        self._table = table
        self._calendar = DatetimeIndex(table.attrs['calendar'], tz='UTC')
        self._first_rows = {
            int(asset_id): start_index
            for asset_id, start_index in iteritems(table.attrs['first_row'])
        }
        self._last_rows = {
            int(asset_id): end_index
            for asset_id, end_index in iteritems(table.attrs['last_row'])
        }
        self._calendar_offsets = {
            int(id_): offset
            for id_, offset in iteritems(table.attrs['calendar_offset'])
        }

    def _compute_slices(self, start_idx, end_idx, assets):
        """
        Compute the raw row indices to load for each asset on a query for the
        given dates after applying a shift.

        Parameters
        ----------
        start_idx : int
            Index of first date for which we want data.
        end_idx : int
            Index of last date for which we want data.
        assets : pandas.Int64Index
            Assets for which we want to compute row indices

        Returns
        -------
        A 3-tuple of (first_rows, last_rows, offsets):
        first_rows : np.array[intp]
            Array with length == len(assets) containing the index of the first
            row to load for each asset in `assets`.
        last_rows : np.array[intp]
            Array with length == len(assets) containing the index of the last
            row to load for each asset in `assets`.
        offset : np.array[intp]
            Array with length == (len(asset) containing the index in a buffer
            of length `dates` corresponding to the first row of each asset.

            The value of offset[i] will be 0 if asset[i] existed at the start
            of a query.  Otherwise, offset[i] will be equal to the number of
            entries in `dates` for which the asset did not yet exist.
        """
        # The core implementation of the logic here is implemented in Cython
        # for efficiency.
        return _compute_row_slices(
            self._first_rows,
            self._last_rows,
            self._calendar_offsets,
            start_idx,
            end_idx,
            assets,
        )

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        # Assumes that the given dates are actually in calendar.
        start_idx = self._calendar.get_loc(start_date)
        end_idx = self._calendar.get_loc(end_date)
        first_rows, last_rows, offsets = self._compute_slices(
            start_idx,
            end_idx,
            assets,
        )
        return _read_bcolz_data(
            self._table,
            (end_idx - start_idx + 1, len(assets)),
            [column.name for column in columns],
            first_rows,
            last_rows,
            offsets,
        )


class SQLiteAdjustmentWriter(object):
    """
    Writer for data to be read by SQLiteAdjustmentWriter

    Parameters
    ----------
    conn_or_path : str or sqlite3.Connection
        A handle to the target sqlite database.
    overwrite : bool, optional, default=False
        If True and conn_or_path is a string, remove any existing files at the
        given path before connecting.

    See Also
    --------
    SQLiteAdjustmentReader
    """

    def __init__(self, conn_or_path, overwrite=False):
        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
        elif isinstance(conn_or_path, str):
            if overwrite and exists(conn_or_path):
                try:
                    remove(conn_or_path)
                except OSError as e:
                    if e.errno != ENOENT:
                        raise
            self.conn = sqlite3.connect(conn_or_path)
        else:
            raise TypeError("Unknown connection type %s" % type(conn_or_path))

    def write_frame(self, tablename, frame):
        if frozenset(frame.columns) != SQLITE_ADJUSTMENT_COLUMNS:
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    SQLITE_ADJUSTMENT_COLUMNS,
                    frame.columns.tolist(),
                )
            )
        elif tablename not in SQLITE_ADJUSTMENT_TABLENAMES:
            raise ValueError(
                "Adjustment table %s not in %s" % (
                    tablename, SQLITE_ADJUSTMENT_TABLENAMES
                )
            )

        expected_dtypes = SQLITE_ADJUSTMENT_COLUMN_DTYPES
        actual_dtypes = frame.dtypes
        for colname, expected in iteritems(expected_dtypes):
            actual = actual_dtypes[colname]
            if not issubdtype(actual, expected):
                raise TypeError(
                    "Expected data of type {expected} for column '{colname}', "
                    "but got {actual}.".format(
                        expected=expected,
                        colname=colname,
                        actual=actual,
                    )
                )
        return frame.to_sql(tablename, self.conn)

    def write(self, splits, mergers, dividends):
        """
        Writes data to a SQLite file to be read by SQLiteAdjustmentReader.

        Parameters
        ----------
        splits : pandas.DataFrame
            Dataframe containing split data.
        mergers : pandas.DataFrame
            DataFrame containing merger data.
        dividends : pandas.DataFrame
            DataFrame containing dividend data.

        Notes
        -----
        DataFrame input (`splits`, `mergers`, and `dividends`) should all have
        the following columns:

        effective_date : int
            The date, represented as seconds since Unix epoch, on which the
            adjustment should be applied.
        ratio : float
            A value to apply to all data earlier than the effective date.
        sid : int
            The asset id associated with this adjustment.

        The ratio column is interpreted as follows:
        - For all adjustment types, multiply price fields ('open', 'high',
          'low', and 'close') by the ratio.
        - For **splits only**, **divide** volume by the adjustment ratio.

        Dividend ratios should be calculated as
        1.0 - (dividend_value / "close on day prior to dividend ex_date").

        Returns
        -------
        None

        See Also
        --------
        SQLiteAdjustmentReader : Consumer for the data written by this class
        """
        self.write_frame('splits', splits)
        self.write_frame('mergers', mergers)
        self.write_frame('dividends', dividends)
        self.conn.execute(
            "CREATE INDEX splits_sids "
            "ON splits(sid)"
        )
        self.conn.execute(
            "CREATE INDEX splits_effective_date "
            "ON splits(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX mergers_sids "
            "ON mergers(sid)"
        )
        self.conn.execute(
            "CREATE INDEX mergers_effective_date "
            "ON mergers(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX dividends_sid "
            "ON dividends(sid)"
        )
        self.conn.execute(
            "CREATE INDEX dividends_effective_date "
            "ON dividends(effective_date)"
        )

    def close(self):
        self.conn.close()


class SQLiteAdjustmentReader(object):
    """
    Loads adjustments based on corporate actions from a SQLite database.

    Expects data written in the format output by `SQLiteAdjustmentWriter`.

    Parameters
    ----------
    conn : str or sqlite3.Connection
        Connection from which to load data.
    """

    def __init__(self, conn):
        if isinstance(conn, str):
            conn = sqlite3.connect(conn)
        self.conn = conn

    def load_adjustments(self, columns, dates, assets):
        return load_adjustments_from_sqlite(
            self.conn,
            [column.name for column in columns],
            dates,
            assets,
        )


class USEquityPricingLoader(FFCLoader):
    """
    FFCLoader for US Equity Pricing

    Delegates loading of baselines and adjustments.
    """

    def __init__(self, raw_price_loader, adjustments_loader):
        self.raw_price_loader = raw_price_loader
        # HACK: Pull the calendar off our raw_price_loader so that we can
        # backshift dates.
        self._calendar = self.raw_price_loader._calendar
        self.adjustments_loader = adjustments_loader

    def load_adjusted_array(self, columns, dates, assets, mask):
        # load_adjusted_array is called with dates on which the user's algo
        # will be shown data, which means we need to return the data that would
        # be known at the start of each date.  We assume that the latest data
        # known on day N is the data from day (N - 1), so we shift all query
        # dates back by a day.
        start_date, end_date = _shift_dates(
            self._calendar, dates[0], dates[-1], shift=1,
        )

        raw_arrays = self.raw_price_loader.load_raw_arrays(
            columns,
            start_date,
            end_date,
            assets,
        )

        adjustments = self.adjustments_loader.load_adjustments(
            columns,
            dates,
            assets,
        )

        return [
            adjusted_array(raw_array, mask, col_adjustments)
            for raw_array, col_adjustments in zip(raw_arrays, adjustments)
        ]


def _shift_dates(dates, start_date, end_date, shift):
    try:
        start = dates.get_loc(start_date)
    except KeyError:
        if start_date < dates[0]:
            raise NoFurtherDataError(
                msg=(
                    "Modeling Query requested data starting on {query_start}, "
                    "but first known date is {calendar_start}"
                ).format(
                    query_start=str(start_date),
                    calendar_start=str(dates[0]),
                )
            )
        else:
            raise ValueError("Query start %s not in calendar" % start_date)

    # Make sure that shifting doesn't push us out of the calendar.
    if start < shift:
        raise NoFurtherDataError(
            msg=(
                "Modeling Query requested data from {shift}"
                " days before {query_start}, but first known date is only "
                "{start} days earlier."
            ).format(shift=shift, query_start=start_date, start=start),
        )

    try:
        end = dates.get_loc(end_date)
    except KeyError:
        if end_date > dates[-1]:
            raise NoFurtherDataError(
                msg=(
                    "Modeling Query requesting data up to {query_end}, "
                    "but last known date is {calendar_end}"
                ).format(
                    query_end=end_date,
                    calendar_end=dates[-1],
                )
            )
        else:
            raise ValueError("Query end %s not in calendar" % end_date)
    return dates[start - shift], dates[end - shift]
