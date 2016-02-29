#
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

from abc import (
    ABCMeta,
    abstractmethod,
)


class TradingSchedule(object):
    """
    A TradingSchedule defines the execution timing of a TradingAlgorithm.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def data_availability_time(self, date):
        """
        Given a UTC-canonicalized date, returns a time by-which all data from
        the previous date is available to the algorithm.

        Parameters
        ----------
        date : Timestamp
            The UTC-canonicalized calendar date whose data availability time
            is needed.

        Returns
        -------
        Timestamp or None
            The data availability time on the given date, or None if there is
            no data availability time for that date.
        """
        raise NotImplementedError()

    @abstractmethod
    def start_and_end(self, date):
        """
        Given a UTC-canonicalized date, returns a tuple of timestamps of the
        start and end of the algorithm trading session for that date.

        Parameters
        ----------
        date : Timestamp
            The UTC-canonicalized algorithm trading session date whose start
            and end are needed.

        Returns
        -------
        (Timestamp, Timestamp)
            The start and end for the given date.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_execution_time(self, dt):
        """
        Calculates if a TradingAlgorithm using this TradingSchedule should be
        executed at time dt.

        Parameters
        ----------
        dt : Timestamp
            The time being queried.

        Returns
        -------
        bool
            True if the TradingAlgorithm should be executed at dt,
            otherwise False.
        """
        raise NotImplementedError()

    @abstractmethod
    def last_n_minutes(self, end, n):
        """
        Calculates a trailing window of trading minutes back from the
        given end.

        Parameters
        ----------
        end : Timestamp
            The end of the trailing window.
        n : int
            The number of minutes needed.
        """
        raise NotImplementedError()
