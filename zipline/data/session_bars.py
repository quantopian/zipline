# Copyright 2016 Quantopian, Inc.
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
from abc import ABCMeta, abstractmethod, abstractproperty
from six import with_metaclass


class NoDataOnDate(Exception):
    """
    Raised when a spot price can be found for the sid and date.
    """
    pass


class NoDataBeforeDate(Exception):
    pass


class NoDataAfterDate(Exception):
    pass


class SessionBarReader(with_metaclass(ABCMeta)):
    """
    Reader for OHCLV pricing data at a session frequency.
    """
    _data_frequency = 'session'

    @property
    def data_frequency(self):
        return self._data_frequency

    @abstractmethod
    def load_raw_arrays(self, columns, start_date, end_date, assets):
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

    @abstractmethod
    def get_value(self, sid, session, colname):
        """
        Retrieve the value at the given coordinates.

        This function shares the same input and semantics as the
        ``MinuteBarReaders``'s ``get_value``, in the future, the names may be
        made consistent.

        Parameters
        ----------
        sid : int
            The asset identifier.
        session : pd.Timestamp
            The session label for the desired data point.
        colname : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        See Also
        --------
        zipline.minute_bars.MinuteBarReader.get_value
        """
        pass

    @abstractproperty
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unionining the range for all assets) which the
           reader can provide.
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

    @abstractproperty
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        pass
