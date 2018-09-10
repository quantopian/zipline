from functools import partial

import pandas as pd
import numpy as np

from zipline.data.session_bars import SessionBarReader


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


def convert_price_with_scaling_factor(a, scaling_factor):
    conversion_factor = (1.0 / scaling_factor)

    zeroes = (a == 0)
    return np.where(zeroes, np.nan, a.astype('float64')) * conversion_factor


class HDF5DailyBarReader(SessionBarReader):

    def __init__(self, f, calendar):
        self._file = f
        self._calendar = calendar

        self._postprocessors = {
            country_code: {
                OPEN: partial(
                    convert_price_with_scaling_factor,
                    scaling_factor=self._read_scaling_factor(
                        country_code,
                        OPEN,
                    )
                ),
                HIGH: partial(
                    convert_price_with_scaling_factor,
                    scaling_factor=self._read_scaling_factor(
                        country_code,
                        HIGH,
                    )
                ),
                LOW: partial(
                    convert_price_with_scaling_factor,
                    scaling_factor=self._read_scaling_factor(
                        country_code,
                        LOW,
                    )
                ),
                CLOSE: partial(
                    convert_price_with_scaling_factor,
                    scaling_factor=self._read_scaling_factor(
                        country_code,
                        CLOSE,
                    )
                ),
                VOLUME: lambda a: a,
            }
            for country_code in self._file
        }

    def _read_scaling_factor(self, country_code, field):
        return self._file[country_code][DATA][field].attrs[SCALING_FACTOR]

    def load_raw_arrays(self,
                        country_code,
                        columns,
                        start_date,
                        end_date,
                        assets):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code where the exchange is located.
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

        sid_selector = self._sids(country_code).searchsorted(assets)

        date_slice = self._compute_date_range_slice(country_code, start, end)
        nrows = date_slice.stop - date_slice.start
        out = []
        for column in columns:
            dataset = self._file[country_code][DATA][column]
            ncols = dataset.shape[0]
            shape = (ncols, nrows)
            buf = np.full(shape, 0, dtype=np.uint32)
            dataset.read_direct(buf, np.s_[:, date_slice.start:date_slice.stop])  # noqa
            buf = buf[sid_selector].T
            out.append(self._postprocessors[country_code][column](buf))

        return out

    def _compute_date_range_slice(self, country_code, start_date, end_date):
        dates = self._dates(country_code)

        # Get the index of the start of dates for ``start_date``.
        start_ix = dates.searchsorted(start_date)

        # Get the index of the start of the first date **after** end_date.
        end_ix = dates.searchsorted(end_date, side='right')

        return slice(start_ix, end_ix)

    def _requested_dates(self, country_code, start_date, end_date):
        dates = self._dates(country_code)

        start_ix = dates.searchsorted(start_date)
        end_ix = dates.searchsorted(end_date, side='right')
        return dates[start_ix:end_ix]

    def _dates(self, country_code):
        return self._file[country_code][INDEX][DAY][:].astype('datetime64[ns]')

    def _sids(self, country_code):
        sids = self._file[country_code][INDEX][SID][:]
        return sids.astype(int)

    def last_available_dt(self, country_code):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code where the exchange is located.

        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return pd.Timestamp(self._dates(country_code)[-1], tz='UTC')

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        return self._calendar

    def first_trading_day(self, country_code):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code where the exchange is located.

        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return pd.Timestamp(self._dates(country_code)[0], tz='UTC')

    @property
    def sessions(self, country_code):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code where the exchange is located.

        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unionining the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(self._dates(country_code), utc=True)

    def get_value(self, country_code, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code where the exchange is located.
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
        sid_ix = self._sids(country_code).searchsorted(sid)
        dt_ix = self._dates(country_code).searchsorted(dt.asm8)

        return self._postprocessors[country_code][field](
            self._file[country_code][DATA][field][sid_ix, dt_ix]
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
