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
from errno import ENOENT
from os import remove
from os.path import exists
import sqlite3

from bcolz import (
    carray,
    ctable,
    open as open_ctable,
)
from click import progressbar
from numpy import (
    array,
    int64,
    float64,
    floating,
    full,
    iinfo,
    integer,
    issubdtype,
    nan,
    uint32,
)
from pandas import (
    DataFrame,
    DatetimeIndex,
    read_csv,
    Timestamp,
)
from six import (
    iteritems,
    with_metaclass,
)

from zipline.utils.input_validation import coerce_string, preprocess

from ._equities import _compute_row_slices, _read_bcolz_data
from ._adjustments import load_adjustments_from_sqlite

import logbook
logger = logbook.Logger('UsEquityPricing')

OHLC = frozenset(['open', 'high', 'low', 'close'])
US_EQUITY_PRICING_BCOLZ_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'day', 'id'
]
SQLITE_ADJUSTMENT_COLUMNS = frozenset(['effective_date', 'ratio', 'sid'])
SQLITE_ADJUSTMENT_COLUMN_DTYPES = {
    'effective_date': integer,
    'ratio': floating,
    'sid': integer,
}
SQLITE_ADJUSTMENT_TABLENAMES = frozenset(['splits', 'dividends', 'mergers'])


SQLITE_DIVIDEND_PAYOUT_COLUMNS = frozenset(
    ['sid',
     'ex_date',
     'declared_date',
     'pay_date',
     'record_date',
     'amount'])
SQLITE_DIVIDEND_PAYOUT_COLUMN_DTYPES = {
    'sid': integer,
    'ex_date': integer,
    'declared_date': integer,
    'record_date': integer,
    'pay_date': integer,
    'amount': float,
}


SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMNS = frozenset(
    ['sid',
     'ex_date',
     'declared_date',
     'record_date',
     'pay_date',
     'payment_sid',
     'ratio'])
SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMN_DTYPES = {
    'sid': integer,
    'ex_date': integer,
    'declared_date': integer,
    'record_date': integer,
    'pay_date': integer,
    'payment_sid': integer,
    'ratio': float,
}
UINT32_MAX = iinfo(uint32).max


class NoDataOnDate(Exception):
    """
    Raised when a spot price can be found for the sid and date.
    """
    pass


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
                    columns['id'].append(full((nrows,), asset_id, uint32))
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
            self.check_uint_safe(arrmax.view(int64) / nanos_per_second,
                                 colname)
            return (array.view(int64) / nanos_per_second).astype(uint32)

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
    @preprocess(table=coerce_string(open_ctable, mode='r'))
    def __init__(self, table):

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
        # Cache of fully read np.array for the carrays in the daily bar table.
        # raw_array does not use the same cache, but it could.
        # Need to test keeping the entire array in memory for the course of a
        # process first.
        self._spot_cols = {}

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

    def _spot_col(self, colname):
        """
        Get the colname from daily_bar_table and read all of it into memory,
        caching the result.

        Parameters
        ----------
        colname : string
            A name of a OHLCV carray in the daily_bar_table

        Returns
        -------
        array (uint32)
            Full read array of the carray in the daily_bar_table with the
            given colname.
        """
        try:
            col = self._spot_cols[colname]
        except KeyError:
            col = self._spot_cols[colname] = self._table[colname][:]
        return col

    def sid_day_index(self, sid, day):
        """
        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64-like
            Midnight of the day for which data is requested.

        Returns
        -------
        int
            Index into the data tape for the given sid and day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
        """
        day_loc = self._calendar.get_loc(day)
        offset = day_loc - self._calendar_offsets[sid]
        if offset < 0:
            raise NoDataOnDate(
                "No data on or before day={0} for sid={1}".format(
                    day, sid))
        ix = self._first_rows[sid] + offset
        if ix > self._last_rows[sid]:
            raise NoDataOnDate(
                "No data on or after day={0} for sid={1}".format(
                    day, sid))
        return ix

    def spot_price(self, sid, day, colname):
        """
        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64-like
            Midnight of the day for which data is requested.
        colname : string
            The price field. e.g. ('open', 'high', 'low', 'close', 'volume')

        Returns
        -------
        float
            The spot price for colname of the given sid on the given day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
            Returns -1 if the day is within the date range, but the price is
            0.
        """
        ix = self.sid_day_index(sid, day)
        price = self._spot_col(colname)[ix]
        if price == 0:
            return -1
        if colname != 'volume':
            return price * 0.001
        else:
            return price


class SQLiteAdjustmentWriter(object):
    """
    Writer for data to be read by SQLiteAdjustmentReader

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

    def __init__(self, conn_or_path, calendar, daily_bar_reader,
                 overwrite=False):
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

        self._daily_bar_reader = daily_bar_reader
        self._calendar = calendar

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

    def write_dividend_payouts(self, frame):
        """
        Write dividend payout data to SQLite table `dividend_payouts`.
        """
        if frozenset(frame.columns) != SQLITE_DIVIDEND_PAYOUT_COLUMNS:
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    sorted(SQLITE_DIVIDEND_PAYOUT_COLUMNS),
                    sorted(frame.columns.tolist()),
                )
            )

        expected_dtypes = SQLITE_DIVIDEND_PAYOUT_COLUMN_DTYPES
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
        return frame.to_sql('dividend_payouts', self.conn)

    def write_stock_dividend_payouts(self, frame):
        if frozenset(frame.columns) != SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMNS:
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    sorted(SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMNS),
                    sorted(frame.columns.tolist()),
                )
            )

        expected_dtypes = SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMN_DTYPES
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
        return frame.to_sql('stock_dividend_payouts', self.conn)

    def calc_dividend_ratios(self, dividends):
        """
        Calculate the ratios to apply to equities when looking back at pricing
        history so that the price is smoothed over the ex_date, when the market
        adjusts to the change in equity value due to upcoming dividend.

        Returns
        -------
        DataFrame
            A frame in the same format as splits and mergers, with keys
            - sid, the id of the equity
            - effective_date, the date in seconds on which to apply the ratio.
            - ratio, the ratio to apply to backwards looking pricing data.
        """
        ex_dates = dividends.ex_date.values

        sids = dividends.sid.values
        amounts = dividends.amount.values

        ratios = full(len(amounts), nan)

        daily_bar_reader = self._daily_bar_reader

        calendar = self._calendar

        effective_dates = full(len(amounts), -1, dtype=int64)

        for i, amount in enumerate(amounts):
            sid = sids[i]
            ex_date = ex_dates[i]
            day_loc = calendar.get_loc(ex_date)
            prev_close_date = calendar[day_loc - 1]
            try:
                prev_close = daily_bar_reader.spot_price(
                    sid, prev_close_date, 'close')
                if prev_close != 0.0:
                    ratio = 1.0 - amount / prev_close
                    ratios[i] = ratio
                    # only assign effective_date when data is found
                    effective_dates[i] = ex_date
            except NoDataOnDate:
                logger.warn("Couldn't compute ratio for dividend %s" % {
                    'sid': sid,
                    'ex_date': ex_date,
                    'amount': amount,
                })
                continue

        # Create a mask to filter out indices in the effective_date, sid, and
        # ratio vectors for which a ratio was not calculable.
        effective_mask = effective_dates != -1
        effective_dates = effective_dates[effective_mask]
        effective_dates = effective_dates.astype('datetime64[ns]').\
            astype('datetime64[s]').astype(uint32)
        sids = sids[effective_mask]
        ratios = ratios[effective_mask]

        return DataFrame({
            'sid': sids,
            'effective_date': effective_dates,
            'ratio': ratios,
        })

    def write_dividend_data(self, dividends, stock_dividends=None):
        """
        Write both dividend payouts and the derived price adjustment ratios.
        """

        # First write the dividend payouts.
        dividend_payouts = dividends.copy()
        dividend_payouts['ex_date'] = dividend_payouts['ex_date'].values.\
            astype('datetime64[s]').astype(integer)
        dividend_payouts['record_date'] = \
            dividend_payouts['record_date'].values.astype('datetime64[s]').\
            astype(integer)
        dividend_payouts['declared_date'] = \
            dividend_payouts['declared_date'].values.astype('datetime64[s]').\
            astype(integer)
        dividend_payouts['pay_date'] = \
            dividend_payouts['pay_date'].values.astype('datetime64[s]').\
            astype(integer)

        self.write_dividend_payouts(dividend_payouts)

        if stock_dividends is not None:
            stock_dividend_payouts = stock_dividends.copy()
            stock_dividend_payouts['ex_date'] = \
                stock_dividend_payouts['ex_date'].values.\
                astype('datetime64[s]').astype(integer)
            stock_dividend_payouts['record_date'] = \
                stock_dividend_payouts['record_date'].values.\
                astype('datetime64[s]').astype(integer)
            stock_dividend_payouts['declared_date'] = \
                stock_dividend_payouts['declared_date'].\
                values.astype('datetime64[s]').astype(integer)
            stock_dividend_payouts['pay_date'] = \
                stock_dividend_payouts['pay_date'].\
                values.astype('datetime64[s]').astype(integer)
        else:
            stock_dividend_payouts = DataFrame({
                'sid': array([], dtype=uint32),
                'record_date': array([], dtype=uint32),
                'ex_date': array([], dtype=uint32),
                'declared_date': array([], dtype=uint32),
                'pay_date': array([], dtype=uint32),
                'payment_sid': array([], dtype=uint32),
                'ratio': array([], dtype=float),
            })

        self.write_stock_dividend_payouts(stock_dividend_payouts)

        # Second from the dividend payouts, calculate ratios.

        dividend_ratios = self.calc_dividend_ratios(dividends)

        self.write_frame('dividends', dividend_ratios)

    def write(self, splits, mergers, dividends, stock_dividends=None):
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
        DataFrame input (`splits`, `mergers`) should all have
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

        DataFrame input, 'dividends' should have the following columns:

        sid : int
            The asset id associated with this adjustment.
        ex_date : datetime64
            The date on which an equity must be held to be eligible to receive
            payment.
        declared_date : datetime64
            The date on which the dividend is announced to the public.
        pay_date : datetime64
            The date on which the dividend is distributed.
        record_date : datetime64
            The date on which the stock ownership is checked to determine
            distribution of dividends.
        amount : float
            The cash amount paid for each share.

        Dividend ratios are calculated as
        1.0 - (dividend_value / "close on day prior to dividend ex_date").


        DataFrame input, 'stock_dividends' should have the following columns:

        sid : int
            The asset id associated with this adjustment.
        ex_date : datetime64
            The date on which an equity must be held to be eligible to receive
            payment.
        declared_date : datetime64
            The date on which the dividend is announced to the public.
        pay_date : datetime64
            The date on which the dividend is distributed.
        record_date : datetime64
            The date on which the stock ownership is checked to determine
            distribution of dividends.
        payment_sid : int
            The asset id of the shares that should be paid instead of cash.
        ratio: float
            The ratio of currently held shares in the held sid that should
            be paid with new shares of the payment_sid.

        stock_dividends is optional.


        Returns
        -------
        None

        See Also
        --------
        SQLiteAdjustmentReader : Consumer for the data written by this class
        """
        self.write_frame('splits', splits)
        self.write_frame('mergers', mergers)
        self.write_dividend_data(dividends, stock_dividends)
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
        self.conn.execute(
            "CREATE INDEX dividend_payouts_sid "
            "ON dividend_payouts(sid)"
        )
        self.conn.execute(
            "CREATE INDEX dividends_payouts_ex_date "
            "ON dividend_payouts(ex_date)"
        )
        self.conn.execute(
            "CREATE INDEX stock_dividend_payouts_sid "
            "ON stock_dividend_payouts(sid)"
        )
        self.conn.execute(
            "CREATE INDEX stock_dividends_payouts_ex_date "
            "ON stock_dividend_payouts(ex_date)"
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

    @preprocess(conn=coerce_string(sqlite3.connect))
    def __init__(self, conn):
        self.conn = conn

    def load_adjustments(self, columns, dates, assets):
        return load_adjustments_from_sqlite(
            self.conn,
            [column.name for column in columns],
            dates,
            assets,
        )
