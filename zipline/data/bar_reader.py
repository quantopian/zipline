from abc import ABCMeta, abstractproperty, abstractmethod

from six import with_metaclass


class BarReader(with_metaclass(ABCMeta)):
    @property
    def data_frequency(self):
        return self._data_frequency

    @abstractproperty
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        pass

    @abstractproperty
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        pass

    @abstractmethod
    def get_value(self, sid, dt, colname):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.  Could be a minute,
            or a session label.
        colname : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.
        """
        pass

    @abstractmethod
    def load_raw_arrays(self, columns, start_dt, end_dt, sids):
        """
        Parameters
        ----------
        fields : list of str
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
        pass


class NoDataOnDate(Exception):
    """
    Raised when a spot price can be found for the sid and date.
    """
    pass
