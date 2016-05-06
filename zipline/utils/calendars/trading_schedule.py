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
    abstractproperty,
)
from six import with_metaclass

from .exchange_calendar import get_calendar
from .calendar_helpers import (
    next_scheduled_day,
    previous_scheduled_day,
    next_open_and_close,
    previous_open_and_close,
    scheduled_day_distance,
    minutes_for_day,
    days_in_range,
    minutes_for_days_in_range,
    add_scheduled_days,
    all_scheduled_minutes,
    next_scheduled_minute,
    previous_scheduled_minute,
    minute_window,
)


class TradingSchedule(with_metaclass(ABCMeta)):
    """
    A TradingSchedule defines the execution timing of a TradingAlgorithm.
    """

    def next_execution_day(self, date):
        return next_scheduled_day(
            date,
            last_trading_day=self.last_execution_day,
            is_scheduled_day_hook=self.is_executing_on_day,
        )

    def previous_execution_day(self, date):
        return previous_scheduled_day(
            date,
            first_trading_day=self.first_execution_day,
            is_scheduled_day_hook=self.is_executing_on_day,
        )

    def next_start_and_end(self, date):
        return next_open_and_close(
            date,
            open_and_close_hook=self.start_and_end,
            next_scheduled_day_hook=self.next_execution_day,
        )

    def previous_start_and_end(self, date):
        return previous_open_and_close(
            date,
            open_and_close_hook=self.start_and_end,
            previous_scheduled_day_hook=self.previous_execution_day,
        )

    def execution_day_distance(self, first_date, second_date):
        return scheduled_day_distance(
            first_date, second_date,
            all_days=self.all_execution_days,
        )

    def execution_minutes_for_day(self, day):
        return minutes_for_day(
            day,
            open_and_close_hook=self.start_and_end,
        )

    def execution_days_in_range(self, start, end):
        return days_in_range(
            start, end,
            all_days=self.all_execution_days,
        )

    def execution_minutes_for_days_in_range(self, start, end):
        return minutes_for_days_in_range(
            start, end,
            days_in_range_hook=self.execution_days_in_range,
            minutes_for_day_hook=self.execution_minutes_for_day,
        )

    def add_execution_days(self, n, date):
        return add_scheduled_days(
            n, date,
            next_scheduled_day_hook=self.next_execution_day,
            previous_scheduled_day_hook=self.previous_execution_day,
            all_trading_days=self.all_execution_days,
        )

    def next_execution_minute(self, start):
        return next_scheduled_minute(
            start,
            is_scheduled_day_hook=self.is_executing_on_day,
            open_and_close_hook=self.start_and_end,
            next_open_and_close_hook=self.next_start_and_end,
        )

    def previous_execution_minute(self, start):
        return previous_scheduled_minute(
            start,
            is_scheduled_day_hook=self.is_executing_on_day,
            open_and_close_hook=self.start_and_end,
            previous_open_and_close_hook=self.previous_start_and_end,
        )

    def execution_minute_window(self, start, count, step=1):
        return minute_window(
            start, count, step,
            schedule=self.schedule,
            is_scheduled_minute_hook=self.is_executing_on_minute,
            session_date_hook=self.session_date,
            minutes_for_date_hook=self.execution_minutes_for_day,
        )

    @abstractproperty
    def day(self):
        """
        A CustomBusinessDay defining those days on which the algorithm is
        trading.
        """
        raise NotImplementedError()

    @abstractproperty
    def tz(self):
        """
        The native timezone for this TradingSchedule.
        """
        raise NotImplementedError()

    @abstractproperty
    def first_execution_day(self):
        """
        The first possible day of trading in this TradingSchedule.
        """
        raise NotImplementedError()

    @abstractproperty
    def last_execution_day(self):
        """
        The last possible day of trading in this TradingSchedule.
        """
        raise NotImplementedError()

    @abstractmethod
    def trading_sessions(self, start, end):
        """
        Calculates all of the trading sessions between the given
        start and end.

        Parameters
        ----------
        start : Timestamp
        end : Timestamp

        Returns
        -------
        DataFrame
            A DataFrame, with a DatetimeIndex of trading dates, containing
            columns of trading starts and ends in this TradingSchedule.
        """
        raise NotImplementedError()

    @property
    def all_execution_days(self):
        return self.schedule.index

    @property
    def all_execution_minutes(self):
        return all_scheduled_minutes(self.all_execution_days,
                                     self.execution_minutes_for_days_in_range)

    def trading_dates(self, start, end):
        """
        Calculates the dates of all of the trading sessions between the given
        start and end.

        Parameters
        ----------
        start : Timestamp
        end : Timestamp

        Returns
        -------
        DatetimeIndex
            A DatetimeIndex containing the dates of the desired trading
            sessions.
        """
        return self.trading_sessions(start, end).index

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
    def is_executing_on_minute(self, dt):
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
    def is_executing_on_day(self, dt):
        """
        Calculates if a TradingAlgorithm using this TradingSchedule would
        execute on the day of dt.

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
    def session_date(self, dt):
        """
        Given a time, returns the UTC-canonicalized date of the trading
        session in which the time belongs. If the time is not in a trading
        session (while algorithm isn't trading), returns the date of the next
        exchange session after the time.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        Timestamp
            The date of the exchange session in which dt belongs.
        """
        raise NotImplementedError()

    @abstractproperty
    def early_ends(self):
        """
        Returns a DatetimeIndex containing the session dates on-which there is
        an early end to trading.
        """
        raise NotImplementedError()


class ExchangeTradingSchedule(TradingSchedule):
    """
    A TradingSchedule that functions as a wrapper around an ExchangeCalendar.
    """

    def __init__(self, cal):
        """
        Docstring goes here, Jimmy

        Parameters
        ----------
        cal : ExchangeCalendar
            The ExchangeCalendar to be represented by this
            ExchangeTradingSchedule.
        """
        self._exchange_calendar = cal
        super(ExchangeTradingSchedule, self).__init__()

    @property
    def all_execution_days(self):
        return self._exchange_calendar.all_trading_days

    @property
    def all_execution_minutes(self):
        return self._exchange_calendar.all_trading_minutes

    @property
    def day(self):
        return self._exchange_calendar.day

    @property
    def tz(self):
        return self._exchange_calendar.tz

    @property
    def schedule(self):
        return self._exchange_calendar.schedule

    @property
    def first_execution_day(self):
        return self._exchange_calendar.first_trading_day

    @property
    def last_execution_day(self):
        return self._exchange_calendar.last_trading_day

    def trading_sessions(self, start, end):
        """
        See TradingSchedule definition.
        """
        return self._exchange_calendar.trading_days(start, end)

    def data_availability_time(self, date):
        """
        See TradingSchedule definition.
        """
        calendar_open, _ = self._exchange_calendar.open_and_close(date)
        return calendar_open

    def start_and_end(self, date):
        """
        See TradingSchedule definition.
        """
        return self._exchange_calendar.open_and_close(date)

    def is_executing_on_minute(self, dt):
        """
        See TradingSchedule definition.
        """
        return self._exchange_calendar.is_open_on_minute(dt)

    def is_executing_on_day(self, dt):
        """
        See TradingSchedule definition.
        """
        return self._exchange_calendar.is_open_on_day(dt)

    def session_date(self, dt):
        """
        See TradingSchedule definition.
        """
        return self._exchange_calendar.session_date(dt)

    @property
    def early_ends(self):
        return self._exchange_calendar.early_closes


default_nyse_schedule = ExchangeTradingSchedule(cal=get_calendar('NYSE'))
