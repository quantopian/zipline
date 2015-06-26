
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
    array_equal,
    full,
    uint32,
)
from pandas import DatetimeIndex
from six import (
    iteritems,
    string_types,
    with_metaclass,
)
import sqlite3


from zipline.data.ffc.base import FFCLoader
from zipline.data.ffc.loaders._us_equity_pricing import (
    _compute_row_slices,
    _read_bcolz_data,
    load_adjustments_from_sqlite,
)

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)


US_EQUITY_PRICING_BCOLZ_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'day', 'id'
]
DAILY_US_EQUITY_PRICING_DEFAULT_FILENAME = 'daily_us_equity_pricing.bcolz'
SQLITE_ADJUSTMENT_COLUMNS = frozenset(['effective_date', 'ratio', 'sid'])
SQLITE_ADJUSTMENT_TABLENAMES = frozenset(['splits', 'dividends', 'mergers'])


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
    def gen_ctables(self, dates, assets):
        """
        Return an iterator of pairs of (asset_id, bcolz.ctable).
        """
        raise NotImplementedError()

    @abstractmethod
    def to_timestamp(self, raw_dt):
        """
        Convert a raw date entry produced by gen_ctables into a pandas
        Timestamp.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_uint32(self, array, colname):
        """
        Convert raw column values produced by gen_ctables into uint32 values.

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
        _iterator = self.gen_ctables(calendar, assets)
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
            calendar_offset[asset_key] = calendar.get_loc(
                self.to_timestamp(table['day'][0])
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

    def _compute_slices(self, dates, assets):
        """
        Compute the raw row indices to load for each asset on a query for the
        given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            Dates of the query on which we want to compute row indices.
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
        query_start = self._calendar.get_loc(dates[0])
        query_stop = self._calendar.get_loc(dates[-1])

        # Sanity check that the requested date range matches our calendar.
        # This could be removed in the future if it's materially affecting
        # performance.
        query_dates = self._calendar[query_start:query_stop + 1]
        if not array_equal(query_dates.values, dates.values):
            raise ValueError("Incompatible calendars!")

        # The core implementation of the logic here is implemented in Cython
        # for efficiency.
        return _compute_row_slices(
            self._first_rows,
            self._last_rows,
            self._calendar_offsets,
            query_start,
            query_stop,
            assets,
        )

    def load_raw_arrays(self, columns, dates, assets):
        first_rows, last_rows, offsets = self._compute_slices(dates, assets)
        return _read_bcolz_data(
            self._table,
            (len(dates), len(assets)),
            [column.name for column in columns],
            first_rows,
            last_rows,
            offsets,
        )


# Note (ssanderson): This probably could just be a function, but it's a class
# right now for symmetry with with BcolzWriter above.  It's possible that this
# should be further abstracted into an iterator over per-sid frames.
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
    SQLiteAdjustmentWriter.write
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
        return load_adjustments_from_sqlite(self.conn, columns, dates, assets)


class USEquityPricingLoader(FFCLoader):

    def __init__(self, raw_price_loader, adjustments_loader):
        self.raw_price_loader = raw_price_loader
        self.adjustments_loader = adjustments_loader

    def load_adjusted_array(self, columns, dates, assets):
        raw_arrays = self.raw_price_loader.load_raw_arrays(
            columns,
            dates,
            assets,
        )
        adjustments = self.adjustments_loader.load_adjustments(
            columns,
            dates,
            assets,
        )

        return [
            adjusted_array(raw_array, NOMASK, col_adjustments)
            for raw_array, col_adjustments in zip(raw_arrays, adjustments)
        ]
