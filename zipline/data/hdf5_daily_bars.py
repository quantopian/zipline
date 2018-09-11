from functools import partial

import h5py
import logbook
import numpy as np
import pandas as pd

from zipline.data.session_bars import SessionBarReader
from zipline.utils.memoize import lazyval


log = logbook.Logger('HDF5DailyBars')


DATA = 'data'
INDEX = 'index'

SCALING_FACTOR = 'scaling_factor'

OPEN = 'open'
HIGH = 'high'
LOW = 'low'
CLOSE = 'close'
VOLUME = 'volume'

DAY = 'day'
SID = 'sid'


scaling_factors = {
    # Retain 3 decimal places for prices.
    OPEN: 1000,
    HIGH: 1000,
    LOW: 1000,
    CLOSE: 1000,
    # Volume is expected to be a whole integer.
    VOLUME: 1,
}


def coerce_to_uint32(a, field):
    """
    Returns a copy of the array as uint32, applying a scaling factor to
    maintain precision if necessary.
    """
    return (a * scaling_factors[field]).astype('uint32')


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

    def __init__(self, filename, date_chunk_size, driver=None):
        self._filename = filename
        self._date_chunk_size = date_chunk_size

    def _h5_file(self, mode):
        return h5py.File(self._filename, mode)

    def write(self, country_code, frame):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        frame : pd.DataFrame
            A dataframe of OHLCV data with a (sids, dates) index.
        """
        with self._h5_file(mode='a') as h5_file:
            country_group = h5_file.create_group(country_code)

            data_group = country_group.create_group(DATA)
            index_group = country_group.create_group(INDEX)

            # Sort rows by sid, then by date.
            frame = frame.sort_index()

            sids = frame.index.levels[0].values
            index_group.create_dataset(SID, data=sids)

            # h5py does not support datetimes, so they need to be stored
            # as integers.
            days = frame.index.levels[1].astype(np.int64)
            index_group.create_dataset(DAY, data=days)

            log.debug(
                'Wrote {} group to file {}',
                index_group.name,
                self._filename,
            )

            for field in (OPEN, HIGH, LOW, CLOSE, VOLUME):
                data = coerce_to_uint32(
                    frame[field].unstack().fillna(0).values,
                    field,
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

                dataset.attrs['scaling_factor'] = scaling_factors[field]

                log.debug(
                    'Writing dataset {} to file {}',
                    dataset.name, self._filename
                )

        return self._h5_file(mode='r')


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
        start_dt: Timestamp
           Beginning of the window range.
        end_dt: Timestamp
           End of the window range.
        sids : list of int
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

        sid_selector = self._sids.searchsorted(assets)

        date_slice = self._compute_date_range_slice(start, end)
        nrows = date_slice.stop - date_slice.start
        out = []
        for column in columns:
            dataset = self._country_group[DATA][column]
            ncols = dataset.shape[0]
            shape = (ncols, nrows)
            buf = np.full(shape, 0, dtype=np.uint32)
            dataset.read_direct(buf, np.s_[:, date_slice.start:date_slice.stop])  # noqa
            buf = buf[sid_selector].T
            out.append(self._postprocessors[column](buf))

        return out

    def _compute_date_range_slice(self, start_date, end_date):
        # Get the index of the start of dates for ``start_date``.
        start_ix = self._dates.searchsorted(start_date)

        # Get the index of the start of the first date **after** end_date.
        end_ix = self._dates.searchsorted(end_date, side='right')

        return slice(start_ix, end_ix)

    def _requested_dates(self, start_date, end_date):
        start_ix = self._dates.searchsorted(start_date)
        end_ix = self._dates.searchsorted(end_date, side='right')
        return self._dates[start_ix:end_ix]

    @lazyval
    def _dates(self):
        return self._country_group[INDEX][DAY][:].astype('datetime64[ns]')

    @lazyval
    def _sids(self):
        sids = self._country_group[INDEX][SID][:]
        return sids.astype(int)

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return pd.Timestamp(self._dates[-1], tz='UTC')

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """

        # TODO: Write this as an attr on the country group, which can be
        #       read to reconstruct the calendar here.
        return None

    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return pd.Timestamp(self._dates[0], tz='UTC')

    @property
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unionining the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(self._dates, utc=True)

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
        sid_ix = self._sids.searchsorted(sid)
        dt_ix = self._dates.searchsorted(dt.asm8)

        return self._postprocessors[field](
            self._country_group[DATA][field][sid_ix, dt_ix]
        )

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest minute on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded minute.
        dt : pd.Timestamp
            The minute at which to start searching for the last traded minute.

        Returns
        -------
        last_traded : pd.Timestamp
            The dt of the last trade for the given asset, using the input
            dt as a vantage point.
        """
        pass
