"""
HDF5 Pricing File Format
------------------------
At the top level, the file is keyed by country (to support regional
files containing multiple countries).

Within each country, there are 4 subgroups:

``/data``
^^^^^^^^^
Each field (OHLCV) is stored in a dataset as a 2D array, with a row per
sid and a column per session. This differs from the more standard
orientation of dates x sids, because it allows each compressed block to
contain contiguous values for the same sid, which allows for better
compression.

.. code-block:: none

   /data
     /open
     /high
     /low
     /close
     /volume

``/index``
^^^^^^^^^^
Contains two datasets, the index of sids (aligned to the rows of the
OHLCV 2D arrays) and index of sessions (aligned to the columns of the
OHLCV 2D arrays) to use for lookups.

.. code-block:: none

   /index
     /sid
     /day

``/lifetimes``
^^^^^^^^^^^^^^
Contains two datasets, start_date and end_date, defining the lifetime
for each asset, aligned to the sids index.

.. code-block:: none

   /lifetimes
     /start_date
     /end_date

``/currency``
^^^^^^^^^^^^^

Contains a single dataset, ``code``, aligned to the sids index, which contains
the listing currency of each sid.

Example
^^^^^^^
Sample layout of the full file with multiple countries.

.. code-block:: none

   |- /US
   |  |- /data
   |  |  |- /open
   |  |  |- /high
   |  |  |- /low
   |  |  |- /close
   |  |  |- /volume
   |  |
   |  |- /index
   |  |  |- /sid
   |  |  |- /day
   |  |
   |  |- /lifetimes
   |  |  |- /start_date
   |  |  |- /end_date
   |  |
   |  |- /currency
   |     |- /code
   |
   |- /CA
      |- /data
      |  |- /open
      |  |- /high
      |  |- /low
      |  |- /close
      |  |- /volume
      |
      |- /index
      |  |- /sid
      |  |- /day
      |
      |- /lifetimes
      |  |- /start_date
      |  |- /end_date
      |
      |- /currency
         |- /code
"""

from functools import partial

import h5py
import logging
import numpy as np
import pandas as pd
from functools import reduce

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
    NoDataForSid,
    NoDataOnDate,
)
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import bytes_array_to_native_str_object_array
from zipline.utils.pandas_utils import check_indexes_all_same

log = logging.getLogger("HDF5DailyBars")

VERSION = 0

DATA = "data"
INDEX = "index"
LIFETIMES = "lifetimes"
CURRENCY = "currency"
CODE = "code"

SCALING_FACTOR = "scaling_factor"

OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

FIELDS = (OPEN, HIGH, LOW, CLOSE, VOLUME)

DAY = "day"
SID = "sid"

START_DATE = "start_date"
END_DATE = "end_date"

# XXX is reserved for "transactions involving no currency".
MISSING_CURRENCY = "XXX"

DEFAULT_SCALING_FACTORS = {
    # Retain 3 decimal places for prices.
    OPEN: 1000,
    HIGH: 1000,
    LOW: 1000,
    CLOSE: 1000,
    # Volume is expected to be a whole integer.
    VOLUME: 1,
}


def coerce_to_uint32(a, scaling_factor):
    """
    Returns a copy of the array as uint32, applying a scaling factor to
    maintain precision if supplied.
    """
    return (a * scaling_factor).round().astype("uint32")


def days_and_sids_for_frames(frames):
    """
    Returns the date index and sid columns shared by a list of dataframes,
    ensuring they all match.

    Parameters
    ----------
    frames : list[pd.DataFrame]
        A list of dataframes indexed by day, with a column per sid.

    Returns
    -------
    days : np.array[datetime64[ns]]
        The days in these dataframes.
    sids : np.array[int64]
        The sids in these dataframes.

    Raises
    ------
    ValueError
        If the dataframes passed are not all indexed by the same days
        and sids.
    """
    if not frames:
        days = np.array([], dtype="datetime64[ns]")
        sids = np.array([], dtype="int64")
        return days, sids

    # Ensure the indices and columns all match.
    check_indexes_all_same(
        [frame.index for frame in frames],
        message="Frames have mismatched days.",
    )
    check_indexes_all_same(
        [frame.columns for frame in frames],
        message="Frames have mismatched sids.",
    )

    return frames[0].index.values, frames[0].columns.values


class HDF5DailyBarWriter:
    """
    Class capable of writing daily OHLCV data to disk in a format that
    can be read efficiently by HDF5DailyBarReader.

    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    date_chunk_size : int
        The number of days per chunk in the HDF5 file. If this is
        greater than the number of days in the data, the chunksize will
        match the actual number of days.

    See Also
    --------
    zipline.data.hdf5_daily_bars.HDF5DailyBarReader
    """

    def __init__(self, filename, date_chunk_size):
        self._filename = filename
        self._date_chunk_size = date_chunk_size

    def h5_file(self, mode):
        return h5py.File(self._filename, mode)

    def write(self, country_code, frames, currency_codes=None, scaling_factors=None):
        """
        Write the OHLCV data for one country to the HDF5 file.

        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        frames : dict[str, pd.DataFrame]
            A dict mapping each OHLCV field to a dataframe with a row
            for each date and a column for each sid. The dataframes need
            to have the same index and columns.
        currency_codes : pd.Series, optional
            Series mapping sids to 3-digit currency code values for those sids'
            listing currencies. If not passed, missing currencies will be
            written.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        if scaling_factors is None:
            scaling_factors = DEFAULT_SCALING_FACTORS

        # Note that this functions validates that all of the frames
        # share the same days and sids.
        days, sids = days_and_sids_for_frames(list(frames.values()))

        # XXX: We should make this required once we're using it everywhere.
        if currency_codes is None:
            currency_codes = pd.Series(index=sids, data=MISSING_CURRENCY)

        # Currency codes should match dataframe columns.
        check_sids_arrays_match(
            sids,
            currency_codes.index.values,
            message="currency_codes sids do not match data sids:",
        )

        # Write start and end dates for each sid.
        start_date_ixs, end_date_ixs = compute_asset_lifetimes(frames)

        if len(sids):
            chunks = (len(sids), min(self._date_chunk_size, len(days)))
        else:
            # h5py crashes if we provide chunks for empty data.
            chunks = None

        with self.h5_file(mode="a") as h5_file:
            # ensure that the file version has been written
            h5_file.attrs["version"] = VERSION

            country_group = h5_file.create_group(country_code)

            self._write_index_group(country_group, days, sids)
            self._write_lifetimes_group(
                country_group,
                start_date_ixs,
                end_date_ixs,
            )
            self._write_currency_group(country_group, currency_codes)
            self._write_data_group(
                country_group,
                frames,
                scaling_factors,
                chunks,
            )

    def write_from_sid_df_pairs(
        self, country_code, data, currency_codes=None, scaling_factors=None
    ):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        data : iterable[tuple[int, pandas.DataFrame]]
            The data chunks to write. Each chunk should be a tuple of
            sid and the data for that asset.
        currency_codes : pd.Series, optional
            Series mapping sids to 3-digit currency code values for those sids'
            listing currencies. If not passed, missing currencies will be
            written.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        data = list(data)
        if not data:
            empty_frame = pd.DataFrame(
                data=None,
                index=np.array([], dtype="datetime64[ns]"),
                columns=np.array([], dtype="int64"),
            )
            return self.write(
                country_code,
                {f: empty_frame.copy() for f in FIELDS},
                scaling_factors,
            )

        sids, frames = zip(*data)
        ohlcv_frame = pd.concat(frames)

        # Repeat each sid for each row in its corresponding frame.
        sid_ix = np.repeat(sids, [len(f) for f in frames])

        # Add id to the index, so the frame is indexed by (date, id).
        ohlcv_frame.set_index(sid_ix, append=True, inplace=True)

        frames = {field: ohlcv_frame[field].unstack() for field in FIELDS}

        return self.write(
            country_code=country_code,
            frames=frames,
            scaling_factors=scaling_factors,
            currency_codes=currency_codes,
        )

    def _write_index_group(self, country_group, days, sids):
        """Write /country/index."""
        index_group = country_group.create_group(INDEX)
        self._log_writing_dataset(index_group)

        index_group.create_dataset(SID, data=sids)

        # h5py does not support datetimes, so they need to be stored
        # as integers.
        index_group.create_dataset(DAY, data=days.astype(np.int64))

    def _write_lifetimes_group(self, country_group, start_date_ixs, end_date_ixs):
        """Write /country/lifetimes"""
        lifetimes_group = country_group.create_group(LIFETIMES)
        self._log_writing_dataset(lifetimes_group)

        lifetimes_group.create_dataset(START_DATE, data=start_date_ixs)
        lifetimes_group.create_dataset(END_DATE, data=end_date_ixs)

    def _write_currency_group(self, country_group, currencies):
        """Write /country/currency"""
        currency_group = country_group.create_group(CURRENCY)
        self._log_writing_dataset(currency_group)

        currency_group.create_dataset(
            CODE,
            data=currencies.values.astype(dtype="S3"),
        )

    def _write_data_group(self, country_group, frames, scaling_factors, chunks):
        """Write /country/data"""
        data_group = country_group.create_group(DATA)
        self._log_writing_dataset(data_group)

        for field in FIELDS:
            frame = frames[field]

            # Sort rows by increasing sid, and columns by increasing date.
            frame.sort_index(inplace=True)
            frame.sort_index(axis="columns", inplace=True)

            data = coerce_to_uint32(
                frame.T.fillna(0).values,
                scaling_factors[field],
            )

            dataset = data_group.create_dataset(
                field,
                compression="lzf",
                shuffle=True,
                data=data,
                chunks=chunks,
            )
            self._log_writing_dataset(dataset)

            dataset.attrs[SCALING_FACTOR] = scaling_factors[field]

            log.debug("Writing dataset {} to file {}", dataset.name, self._filename)

    def _log_writing_dataset(self, dataset):
        log.debug("Writing {} to file {}", dataset.name, self._filename)


def compute_asset_lifetimes(frames):
    """
    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        A dict mapping each OHLCV field to a dataframe with a row for
        each date and a column for each sid, as passed to write().

    Returns
    -------
    start_date_ixs : np.array[int64]
        The index of the first date with non-nan values, for each sid.
    end_date_ixs : np.array[int64]
        The index of the last date with non-nan values, for each sid.
    """
    # Build a 2D array (dates x sids), where an entry is True if all
    # fields are nan for the given day and sid.
    is_null_matrix = np.logical_and.reduce(
        [frames[field].isnull().values for field in FIELDS],
    )
    if not is_null_matrix.size:
        empty = np.array([], dtype="int64")
        return empty, empty.copy()

    # Offset of the first null from the start of the input.
    start_date_ixs = is_null_matrix.argmin(axis=0)
    # Offset of the last null from the **end** of the input.
    end_offsets = is_null_matrix[::-1].argmin(axis=0)
    # Offset of the last null from the start of the input
    end_date_ixs = is_null_matrix.shape[0] - end_offsets - 1

    return start_date_ixs, end_date_ixs


def convert_price_with_scaling_factor(a, scaling_factor):
    conversion_factor = 1.0 / scaling_factor

    zeroes = a == 0
    return np.where(zeroes, np.nan, a.astype("float64")) * conversion_factor


class HDF5DailyBarReader(CurrencyAwareSessionBarReader):
    """
    Parameters
    ---------
    country_group : h5py.Group
        The group for a single country in an HDF5 daily pricing file.
    """

    def __init__(self, country_group):
        self._country_group = country_group

        self._postprocessors = {
            OPEN: partial(
                convert_price_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(OPEN),
            ),
            HIGH: partial(
                convert_price_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(HIGH),
            ),
            LOW: partial(
                convert_price_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(LOW),
            ),
            CLOSE: partial(
                convert_price_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(CLOSE),
            ),
            VOLUME: lambda a: a,
        }

    @classmethod
    def from_file(cls, h5_file, country_code):
        """
        Construct from an h5py.File and a country code.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        if h5_file.attrs["version"] != VERSION:
            raise ValueError(
                "mismatched version: file is of version %s, expected %s"
                % (
                    h5_file.attrs["version"],
                    VERSION,
                ),
            )

        return cls(h5_file[country_code])

    @classmethod
    def from_path(cls, path, country_code):
        """
        Construct from a file path and a country code.

        Parameters
        ----------
        path : str
            The path to an HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        return cls.from_file(h5py.File(path), country_code)

    def _read_scaling_factor(self, field):
        return self._country_group[DATA][field].attrs[SCALING_FACTOR]

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """
        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        self._validate_timestamp(start_date)
        self._validate_timestamp(end_date)

        start = start_date.asm8
        end = end_date.asm8
        date_slice = self._compute_date_range_slice(start, end)
        n_dates = date_slice.stop - date_slice.start

        # Create a buffer into which we'll read data from the h5 file.
        # Allocate an extra row of space that will always contain null values.
        # We'll use that space to provide "data" for entries in ``assets`` that
        # are unknown to us.
        full_buf = np.zeros((len(self.sids) + 1, n_dates), dtype=np.uint32)
        # We'll only read values into this portion of the read buf.
        mutable_buf = full_buf[:-1]

        # Indexer that converts an array aligned to self.sids (which is what we
        # pull from the h5 file) into an array aligned to ``assets``.
        #
        # Unknown assets will have an index of -1, which means they'll always
        # pull from the last row of the read buffer. We allocated an extra
        # empty row above so that these lookups will cause us to fill our
        # output buffer with "null" values.
        sid_selector = self._make_sid_selector(assets)

        out = []
        for column in columns:
            # Zero the buffer to prepare to receive new data.
            mutable_buf.fill(0)

            dataset = self._country_group[DATA][column]

            # Fill the mutable portion of our buffer with data from the file.
            dataset.read_direct(
                mutable_buf,
                np.s_[:, date_slice],
            )

            # Select data from the **full buffer**. Unknown assets will pull
            # from the last row, which is always empty.
            out.append(self._postprocessors[column](full_buf[sid_selector].T))

        return out

    def _make_sid_selector(self, assets):
        """
        Build an indexer mapping ``self.sids`` to ``assets``.

        Parameters
        ----------
        assets : list[int]
            List of assets requested by a caller of ``load_raw_arrays``.

        Returns
        -------
        index : np.array[int64]
            Index array containing the index in ``self.sids`` for each location
            in ``assets``. Entries in ``assets`` for which we don't have a sid
            will contain -1. It is caller's responsibility to handle these
            values correctly.
        """
        assets = np.array(assets)
        sid_selector = self.sids.searchsorted(assets)
        unknown = np.in1d(assets, self.sids, invert=True)
        sid_selector[unknown] = -1
        return sid_selector

    def _compute_date_range_slice(self, start_date, end_date):
        # Get the index of the start of dates for ``start_date``.
        start_ix = self.dates.searchsorted(start_date)

        # Get the index of the start of the first date **after** end_date.
        end_ix = self.dates.searchsorted(end_date, side="right")

        return slice(start_ix, end_ix)

    def _validate_assets(self, assets):
        """Validate that asset identifiers are contained in the daily bars.

        Parameters
        ----------
        assets : array-like[int]
           The asset identifiers to validate.

        Raises
        ------
        NoDataForSid
            If one or more of the provided asset identifiers are not
            contained in the daily bars.
        """
        missing_sids = np.setdiff1d(assets, self.sids)

        if len(missing_sids):
            raise NoDataForSid(
                "Assets not contained in daily pricing file: {}".format(missing_sids)
            )

    def _validate_timestamp(self, ts):
        if ts.asm8 not in self.dates:
            raise NoDataOnDate(ts)

    @lazyval
    def dates(self):
        return self._country_group[INDEX][DAY][:].astype("datetime64[ns]")

    @lazyval
    def sids(self):
        return self._country_group[INDEX][SID][:].astype("int64", copy=False)

    @lazyval
    def asset_start_dates(self):
        return self.dates[self._country_group[LIFETIMES][START_DATE][:]]

    @lazyval
    def asset_end_dates(self):
        return self.dates[self._country_group[LIFETIMES][END_DATE][:]]

    @lazyval
    def _currency_codes(self):
        bytes_array = self._country_group[CURRENCY][CODE][:]
        return bytes_array_to_native_str_object_array(bytes_array)

    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[object]
            Array of currency codes for listing currencies of ``sids``.
        """
        # Find the index of requested sids in our stored sids.
        ixs = self.sids.searchsorted(sids, side="left")

        result = self._currency_codes[ixs]

        # searchsorted returns the index of the next lowest sid if the lookup
        # fails. Fill these sids with the special "missing" sentinel.
        not_found = self.sids[ixs] != sids

        result[not_found] = None

        return result

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return pd.Timestamp(self.dates[-1])

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        raise NotImplementedError(
            "HDF5 pricing does not yet support trading calendars."
        )

    @property
    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return pd.Timestamp(self.dates[0])

    @lazyval
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(self.dates)

    def get_value(self, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        """
        self._validate_assets([sid])
        self._validate_timestamp(dt)

        sid_ix = self.sids.searchsorted(sid)
        dt_ix = self.dates.searchsorted(dt.asm8)

        value = self._postprocessors[field](
            self._country_group[DATA][field][sid_ix, dt_ix]
        )

        # When the value is nan, this dt may be outside the asset's lifetime.
        # If that's the case, the proper NoDataOnDate exception is raised.
        # Otherwise (when there's just a hole in the middle of the data), the
        # nan is returned.
        if np.isnan(value):
            if dt.asm8 < self.asset_start_dates[sid_ix]:
                raise NoDataBeforeDate()

            if dt.asm8 > self.asset_end_dates[sid_ix]:
                raise NoDataAfterDate()

        return value

    def get_last_traded_dt(self, asset, dt):
        """Get the latest day on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded day.
        dt : pd.Timestamp
            The dt at which to start searching for the last traded day.

        Returns
        -------
        last_traded : pd.Timestamp
            The day of the last trade for the given asset, using the
            input dt as a vantage point.
        """
        sid_ix = self.sids.searchsorted(asset.sid)
        # Used to get a slice of all dates up to and including ``dt``.
        dt_limit_ix = self.dates.searchsorted(dt.asm8, side="right")

        # Get the indices of all dates with nonzero volume.
        nonzero_volume_ixs = np.ravel(
            np.nonzero(self._country_group[DATA][VOLUME][sid_ix, :dt_limit_ix])
        )

        if len(nonzero_volume_ixs) == 0:
            return pd.NaT

        return pd.Timestamp(self.dates[nonzero_volume_ixs][-1])


class MultiCountryDailyBarReader(CurrencyAwareSessionBarReader):
    """

    Parameters
    ---------
    readers : dict[str -> SessionBarReader]
        A dict mapping country codes to SessionBarReader instances to
        service each country.
    """

    def __init__(self, readers):
        self._readers = readers
        self._country_map = pd.concat(
            [
                pd.Series(index=reader.sids, data=country_code)
                for country_code, reader in readers.items()
            ]
        )

    @classmethod
    def from_file(cls, h5_file):
        """Construct from an h5py.File.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        """
        return cls(
            {
                country: HDF5DailyBarReader.from_file(h5_file, country)
                for country in h5_file.keys()
            }
        )

    @classmethod
    def from_path(cls, path):
        """
        Construct from a file path.

        Parameters
        ----------
        path : str
            Path to an HDF5 daily pricing file.
        """
        return cls.from_file(h5py.File(path))

    @property
    def countries(self):
        """A set-like object of the country codes supplied by this reader."""
        return self._readers.keys()

    def _country_code_for_assets(self, assets):
        country_codes = self._country_map.reindex(assets)

        # Series.get() returns None if none of the labels are in the index.
        if country_codes is not None:
            unique_country_codes = country_codes.dropna().unique()
            num_countries = len(unique_country_codes)
        else:
            num_countries = 0

        if num_countries == 0:
            raise ValueError("At least one valid asset id is required.")
        elif num_countries > 1:
            raise NotImplementedError(
                (
                    "Assets were requested from multiple countries ({}),"
                    " but multi-country reads are not yet supported."
                ).format(list(unique_country_codes))
            )

        return unique_country_codes.item()

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """

        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        country_code = self._country_code_for_assets(assets)

        return self._readers[country_code].load_raw_arrays(
            columns,
            start_date,
            end_date,
            assets,
        )

    @property
    def last_available_dt(self):
        """

        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return max(reader.last_available_dt for reader in self._readers.values())

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        raise NotImplementedError(
            "HDF5 pricing does not yet support trading calendars."
        )

    @property
    def first_trading_day(self):
        """

        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return min(reader.first_trading_day for reader in self._readers.values())

    @property
    def sessions(self):
        """

        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(
            reduce(
                np.union1d,
                (reader.dates for reader in self._readers.values()),
            ),
        )

    def get_value(self, sid, dt, field):
        """Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        NoDataForSid
            If the given sid is not valid.
        """
        try:
            country_code = self._country_code_for_assets([sid])
        except ValueError as exc:
            raise NoDataForSid(
                "Asset not contained in daily pricing file: {}".format(sid)
            ) from exc
        return self._readers[country_code].get_value(sid, dt, field)

    def get_last_traded_dt(self, asset, dt):
        """Get the latest day on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded day.
        dt : pd.Timestamp
            The dt at which to start searching for the last traded day.

        Returns
        -------
        last_traded : pd.Timestamp
            The day of the last trade for the given asset, using the
            input dt as a vantage point.
        """
        country_code = self._country_code_for_assets([asset.sid])
        return self._readers[country_code].get_last_traded_dt(asset, dt)

    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Assumes that a sid's prices are always quoted in a single currency.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[S3]
            Array of currency codes for listing currencies of ``sids``.
        """
        country_code = self._country_code_for_assets(sids)
        return self._readers[country_code].currency_codes(sids)


def check_sids_arrays_match(left, right, message):
    """Check that two 1d arrays of sids are equal"""
    if len(left) != len(right):
        raise ValueError(
            "{}:\nlen(left) ({}) != len(right) ({})".format(
                message, len(left), len(right)
            )
        )

    diff = left != right
    if diff.any():
        (bad_locs,) = np.where(diff)
        raise ValueError("{}:\n Indices with differences: {}".format(message, bad_locs))
