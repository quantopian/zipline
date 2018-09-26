"""
Reader and writer for the HDF5 daily pricing file format.

At the top level, the file is keyed by country (to support regional
files containing multiple countries).

###

Within each country, there are 3 subgroups:

1) /data
     /open
     /high
     /low
     /close
     /volume

Each field (OHLCV) is stored in a dataset as a 2D array, with a row per
sid and a column per session. This differs from the more standard
orientation of dates x sids, because it allows each compressed block to
contain contiguous values for the same sid, which allows for better
compression.

2) /index
     /sid
     /day

Contains two datasets, the index of sids (aligned to the rows of the
OHLCV 2D arrays) and index of sessions (aligned to the columns of the
OHLCV 2D arrays) to use for lookups.

3) /lifetimes
     /start_date
     /end_date

Contains two datasets, start_date and end_date, defining the lifetime
for each asset, aligned to the sids index.

###

Sample layout of the full file with multiple countries:

    /US
      /data
        /open
        /high
        /low
        /close
        /volume
      /index
        /sid
        /day
      /lifetimes
        /start_date
        /end_date
    /CA
      /data
        /open
        /high
        /low
        /close
        /volume
      /index
        /sid
        /day
      /lifetimes
        /start_date
        /end_date
"""


from functools import partial

import h5py
import logbook
import numpy as np
import pandas as pd
from six import iteritems
from six.moves import reduce

from zipline.data.bar_reader import NoDataBeforeDate, NoDataAfterDate
from zipline.data.session_bars import SessionBarReader
from zipline.utils.memoize import lazyval
from zipline.utils.pandas_utils import check_indexes_all_same


log = logbook.Logger('HDF5DailyBars')


DATA = 'data'
INDEX = 'index'
LIFETIMES = 'lifetimes'

SCALING_FACTOR = 'scaling_factor'

OPEN = 'open'
HIGH = 'high'
LOW = 'low'
CLOSE = 'close'
VOLUME = 'volume'

FIELDS = (OPEN, HIGH, LOW, CLOSE, VOLUME)

DAY = 'day'
SID = 'sid'

START_DATE = 'start_date'
END_DATE = 'end_date'


DEFAULT_SCALING_FACTORS = {
    # Retain 3 decimal places for prices.
    OPEN: 1000,
    HIGH: 1000,
    LOW: 1000,
    CLOSE: 1000,
    # Volume is expected to be a whole integer.
    VOLUME: 1,
}


def coerce_to_uint32(a, field, scaling_factor):
    """
    Returns a copy of the array as uint32, applying a scaling factor to
    maintain precision if supplied.
    """
    return (a * scaling_factor).astype('uint32')


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

    # Ensure the indices and columns all match.
    check_indexes_all_same(
        [frame.index for frame in frames],
        message='Frames have mistmatched days.',
    )
    check_indexes_all_same(
        [frame.columns for frame in frames],
        message='Frames have mismatched sids.',
    )

    return frames[0].index.values, frames[0].columns.values


class HDF5DailyBarWriter(object):
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

    def write(self, country_code, frames, scaling_factors=None):
        """Write the OHLCV data for one country to the HDF5 file.

        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        frames : dict[str, pd.DataFrame]
            A dict mapping each OHLCV field to a dataframe with a row
            for each date and a column for each sid. The dataframes need
            to have the same index and columns.
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

        with self.h5_file(mode='a') as h5_file:
            country_group = h5_file.create_group(country_code)

            data_group = country_group.create_group(DATA)
            index_group = country_group.create_group(INDEX)
            lifetimes_group = country_group.create_group(LIFETIMES)

            # Note that this functions validates that all of the frames
            # share the same days and sids.
            days, sids = days_and_sids_for_frames(list(frames.values()))

            # Write sid and date indices.
            index_group.create_dataset(SID, data=sids)

            # h5py does not support datetimes, so they need to be stored
            # as integers.
            index_group.create_dataset(DAY, data=days.astype(np.int64))

            log.debug(
                'Wrote {} group to file {}',
                index_group.name,
                self._filename,
            )

            # Write start and end dates for each sid.
            start_date_ixs, end_date_ixs = compute_asset_lifetimes(frames)

            lifetimes_group.create_dataset(START_DATE, data=start_date_ixs)
            lifetimes_group.create_dataset(END_DATE, data=end_date_ixs)

            for field in FIELDS:
                frame = frames[field]

                # Sort rows by increasing sid, and columns by increasing date.
                frame.sort_index(inplace=True)
                frame.sort_index(axis='columns', inplace=True)

                data = coerce_to_uint32(
                    frame.T.fillna(0).values,
                    field,
                    scaling_factors[field],
                )

                dataset = data_group.create_dataset(
                    field,
                    compression='lzf',
                    shuffle=True,
                    data=data,
                    chunks=(
                        len(sids),
                        min(self._date_chunk_size, len(days))
                    ),
                )

                dataset.attrs[SCALING_FACTOR] = scaling_factors[field]

                log.debug(
                    'Writing dataset {} to file {}',
                    dataset.name, self._filename
                )

    def write_from_sid_df_pairs(self,
                                country_code,
                                data,
                                scaling_factors=None):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        data : iterable[tuple[int, pandas.DataFrame]]
            The data chunks to write. Each chunk should be a tuple of
            sid and the data for that asset.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        ohlcv_frame = pd.concat([df for sid, df in data])

        # Add id to the index, so the frame is indexed by (date, id).
        ohlcv_frame.set_index('id', append=True, inplace=True)

        frames = {
            field: ohlcv_frame[field].unstack()
            for field in FIELDS
        }

        return self.write(country_code, frames, scaling_factors)


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

    # Offset of the first null from the start of the input.
    start_date_ixs = is_null_matrix.argmin(axis=0)
    # Offset of the *last null from the **end** of the input.
    end_offsets = is_null_matrix[::-1].argmin(axis=0)
    # Offset of the last null from the start of the input
    end_date_ixs = is_null_matrix.shape[0] - end_offsets - 1

    return start_date_ixs, end_date_ixs


def convert_price_with_scaling_factor(a, scaling_factor):
    conversion_factor = (1.0 / scaling_factor)

    zeroes = (a == 0)
    return np.where(zeroes, np.nan, a.astype('float64')) * conversion_factor


class HDF5DailyBarReader(SessionBarReader):
    """
    Parameters
    ---------
    country_group : h5py.Group
        The group for a single country in an HDF5 daily pricing file.
    """
    def __init__(self, country_group):
        self._country_group = country_group

        self._postprocessors = {
            OPEN: partial(convert_price_with_scaling_factor,
                          scaling_factor=self._read_scaling_factor(OPEN)),
            HIGH: partial(convert_price_with_scaling_factor,
                          scaling_factor=self._read_scaling_factor(HIGH)),
            LOW: partial(convert_price_with_scaling_factor,
                         scaling_factor=self._read_scaling_factor(LOW)),
            CLOSE: partial(convert_price_with_scaling_factor,
                           scaling_factor=self._read_scaling_factor(CLOSE)),
            VOLUME: lambda a: a,
        }

    @classmethod
    def from_file(cls, h5_file, country_code):
        """
        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        return cls(h5_file[country_code])

    @classmethod
    def from_path(cls, path, country_code):
        """
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

    def load_raw_arrays(self,
                        columns,
                        start_date,
                        end_date,
                        assets):
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
        start = start_date.asm8
        end = end_date.asm8

        sid_selector = self.sids.searchsorted(assets)
        date_slice = self._compute_date_range_slice(start, end)

        nrows = date_slice.stop - date_slice.start
        ncols = len(assets)

        buf = np.zeros((ncols, nrows), dtype=np.uint32)

        out = []
        for column in columns:
            # Zero the buffer to prepare it to receive new data.
            buf.fill(0)

            dataset = self._country_group[DATA][column]

            dataset.read_direct(buf, np.s_[:, date_slice.start:date_slice.stop])  # noqa

            out.append(
                self._postprocessors[column](
                    buf[sid_selector].T
                )
            )

        return out

    def _compute_date_range_slice(self, start_date, end_date):
        # Get the index of the start of dates for ``start_date``.
        start_ix = self.dates.searchsorted(start_date)

        # Get the index of the start of the first date **after** end_date.
        end_ix = self.dates.searchsorted(end_date, side='right')

        return slice(start_ix, end_ix)

    @lazyval
    def dates(self):
        return self._country_group[INDEX][DAY][:].astype('datetime64[ns]')

    @lazyval
    def sids(self):
        sids = self._country_group[INDEX][SID][:]
        return sids.astype(int)

    @lazyval
    def asset_start_dates(self):
        # TODO: This needs test coverage.
        return self.dates[self._country_group[LIFETIMES][START_DATE][:]]

    @lazyval
    def asset_end_dates(self):
        # TODO: This needs test coverage.
        return self.dates[self._country_group[LIFETIMES][END_DATE][:]]

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return pd.Timestamp(self.dates[-1], tz='UTC')

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        raise NotImplementedError(
            'HDF5 pricing does not yet support trading calendars.'
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
        return pd.Timestamp(self.dates[0], tz='UTC')

    @lazyval
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(self.dates, utc=True)

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
                # TODO: This needs test coverage.
                raise NoDataAfterDate()

        return value

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest day on or before ``dt`` in which ``asset`` traded.

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
        dt_limit_ix = self.dates.searchsorted(dt.asm8, side='right')

        # Get the indices of all dates with nonzero volume.
        nonzero_volume_ixs = np.ravel(
            np.nonzero(self._country_group[DATA][VOLUME][sid_ix, :dt_limit_ix])
        )

        if len(nonzero_volume_ixs) == 0:
            return pd.NaT

        return pd.Timestamp(self.dates[nonzero_volume_ixs][-1], tz='UTC')


class MultiCountryDailyBarReader(SessionBarReader):
    """
    Parameters
    ---------
    readers : dict[str -> SessionBarReader]
        A dict mapping country codes to SessionBarReader instances to
        service each country.
    """
    def __init__(self, readers):
        self._readers = readers
        self._country_map = pd.concat([
            pd.Series(index=reader.sids, data=country_code)
            for country_code, reader in iteritems(readers)
        ])

    def _country_code_for_assets(self, assets):
        country_codes = self._country_map[assets]

        if country_codes.isnull().any():
            raise ValueError(
                'Assets not contained in daily pricing file: {}'.format(
                    list(country_codes[country_codes.isnull()].index)
                )
            )

        unique_country_codes = country_codes.unique()

        if len(unique_country_codes) > 1:
            raise NotImplementedError(
                (
                    'Assets were requested from multiple countries ({}),'
                    ' but multi-country reads are not yet supported.'
                ).format(list(unique_country_codes))
            )

        return np.asscalar(unique_country_codes)

    def load_raw_arrays(self,
                        columns,
                        start_date,
                        end_date,
                        assets):
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
        return max(
            reader.last_available_dt for reader in self._readers.values()
        )

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        raise NotImplementedError(
            'HDF5 pricing does not yet support trading calendars.'
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
        return min(
            reader.first_trading_day for reader in self._readers.values()
        )

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
                (reader.dates for reader in self.readers.values()),
            ),
            utc=True,
        )

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
        country_code = self._country_code_for_assets([sid])
        return self._readers[country_code].get_value(sid, dt, field)

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest day on or before ``dt`` in which ``asset`` traded.

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
