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

from datetime import time
from itertools import chain

from dateutil.relativedelta import (
    MO,
    TH,
)
from pandas import (
    date_range,
    DateOffset,
    Timestamp,
    Timedelta,
)
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    sunday_to_monday,
    USLaborDay,
    USPresidentsDay,
    USThanksgivingDay,
)
from pandas.tseries.offsets import Day
from pytz import timezone

from .exchange_calendar import ExchangeCalendar
from .calendar_helpers import normalize_date

# Useful resources for making changes to this file:
# http://www.nyse.com/pdfs/closings.pdf
# http://www.stevemorse.org/jcal/whendid.html

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)

US_EASTERN = timezone('US/Eastern')
NYSE_OPEN = time(9, 31)
NYSE_CLOSE = time(16)
NYSE_STANDARD_EARLY_CLOSE = time(13)
# Does the market open or close on a different calendar day, compared to the
# calendar day assigned by the exchang to this session?
NYSE_OPEN_OFFSET = 0
NYSE_CLOSE_OFFSET = 0

# Closings
USNewYearsDay = Holiday(
    'New Years Day',
    month=1,
    day=1,
    # When Jan 1 is a Sunday, NYSE observes the subsequent Monday.  When Jan 1
    # Saturday (as in 2005 and 2011), no holiday is observed.
    observance=sunday_to_monday
)
USMemorialDay = Holiday(
    # NOTE: The definition for Memorial Day is incorrect as of pandas 0.16.0.
    # See https://github.com/pydata/pandas/issues/9760.
    'Memorial Day',
    month=5,
    day=25,
    offset=DateOffset(weekday=MO(1)),
)
USMartinLutherKingJrAfter1998 = Holiday(
    'Dr. Martin Luther King Jr. Day',
    month=1,
    day=1,
    # The NYSE didn't observe MLK day as a holiday until 1998.
    start_date=Timestamp('1998-01-01'),
    offset=DateOffset(weekday=MO(3)),
)
USIndependenceDay = Holiday(
    'July 4th',
    month=7,
    day=4,
    observance=nearest_workday,
)
Christmas = Holiday(
    'Christmas',
    month=12,
    day=25,
    observance=nearest_workday,
)

# Half Days
MonTuesThursBeforeIndependenceDay = Holiday(
    # When July 4th is a Tuesday, Wednesday, or Friday, the previous day is a
    # half day.
    'Mondays, Tuesdays, and Thursdays Before Independence Day',
    month=7,
    day=3,
    days_of_week=(MONDAY, TUESDAY, THURSDAY),
    start_date=Timestamp("1995-01-01"),
)
FridayAfterIndependenceDayExcept2013 = Holiday(
    # When July 4th is a Thursday, the next day is a half day (except in 2013,
    # when, for no explicable reason, Wednesday was a half day instead).
    "Fridays after Independence Day that aren't in 2013",
    month=7,
    day=5,
    days_of_week=(FRIDAY,),
    observance=lambda dt: None if dt.year == 2013 else dt,
    start_date=Timestamp("1995-01-01"),
)
USBlackFridayBefore1993 = Holiday(
    'Black Friday',
    month=11,
    day=1,
    # Black Friday was not observed until 1992.
    start_date=Timestamp('1992-01-01'),
    end_date=Timestamp('1993-01-01'),
    offset=[DateOffset(weekday=TH(4)), Day(1)],
)
USBlackFridayInOrAfter1993 = Holiday(
    'Black Friday',
    month=11,
    day=1,
    start_date=Timestamp('1993-01-01'),
    offset=[DateOffset(weekday=TH(4)), Day(1)],
)
# These have the same definition, but are used in different places because the
# NYSE closed at 2:00 PM on Christmas Eve until 1993.
ChristmasEveBefore1993 = Holiday(
    'Christmas Eve',
    month=12,
    day=24,
    end_date=Timestamp('1993-01-01'),
    # When Christmas is a Saturday, the 24th is a full holiday.
    days_of_week=(MONDAY, TUESDAY, WEDNESDAY, THURSDAY),
)
ChristmasEveInOrAfter1993 = Holiday(
    'Christmas Eve',
    month=12,
    day=24,
    start_date=Timestamp('1993-01-01'),
    # When Christmas is a Saturday, the 24th is a full holiday.
    days_of_week=(MONDAY, TUESDAY, WEDNESDAY, THURSDAY),
)


# http://en.wikipedia.org/wiki/Aftermath_of_the_September_11_attacks
September11Closings = date_range('2001-09-11', '2001-09-16', tz='UTC')

# http://en.wikipedia.org/wiki/Hurricane_sandy
HurricaneSandyClosings = date_range(
    '2012-10-29',
    '2012-10-30',
    tz='UTC'
)

# National Days of Mourning
# - President Richard Nixon - April 27, 1994
# - President Ronald W. Reagan - June 11, 2004
# - President Gerald R. Ford - Jan 2, 2007
USNationalDaysofMourning = [
    Timestamp('1994-04-27', tz='UTC'),
    Timestamp('2004-06-11', tz='UTC'),
    Timestamp('2007-01-02', tz='UTC'),
]


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the NYSE.

    See NYSEExchangeCalendar for full description.
    """
    rules = [
        USNewYearsDay,
        USMartinLutherKingJrAfter1998,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        USIndependenceDay,
        USLaborDay,
        USThanksgivingDay,
        USIndependenceDay,
        Christmas,
    ]


class NYSE2PMCloseCalendar(AbstractHolidayCalendar):
    """
    Holiday Calendar for 2PM closes for NYSE
    """
    rules = [
        ChristmasEveBefore1993,
        USBlackFridayBefore1993,
    ]


class NYSEEarlyCloseCalendar(AbstractHolidayCalendar):
    """
    Regular early close calendar for NYSE
    """
    rules = [
        MonTuesThursBeforeIndependenceDay,
        FridayAfterIndependenceDayExcept2013,
        USBlackFridayInOrAfter1993,
        ChristmasEveInOrAfter1993,
    ]


class NYSEExchangeCalendar(ExchangeCalendar):
    """
    Exchange calendar for NYSE

    Open Time: 9:31 AM, US/Eastern
    Close Time: 4:00 PM, US/Eastern

    Regularly-Observed Holidays:
    - New Years Day (observed on monday when Jan 1 is a Sunday)
    - Martin Luther King Jr. Day (3rd Monday in January, only after 1998)
    - Washington's Birthday (aka President's Day, 3rd Monday in February)
    - Good Friday (two days before Easter Sunday)
    - Memorial Day (last Monday in May)
    - Independence Day (observed on the nearest weekday to July 4th)
    - Labor Day (first Monday in September)
    - Thanksgiving (fourth Thursday in November)
    - Christmas (observed on nearest weekday to December 25)

    NOTE: The NYSE does not observe the following US Federal Holidays:
    - Columbus Day
    - Veterans Day

    Regularly-Observed Early Closes:
    - July 3rd (Mondays, Tuesdays, and Thursdays, 1995 onward)
    - July 5th (Fridays, 1995 onward, except 2013)
    - Christmas Eve (except on Fridays, when the exchange is closed entirely)
    - Day After Thanksgiving (aka Black Friday, observed from 1992 onward)

    NOTE: Until 1993, the standard early close time for the NYSE was 2:00 PM.
    From 1993 onward, it has been 1:00 PM.

    Additional Irregularities:
    - Closed from 9/11/2001 to 9/16/2001 due to terrorist attacks in NYC.
    - Closed on 10/29/2012 and 10/30/2012 due to Hurricane Sandy.
    - Closed on 4/27/1994 due to Richard Nixon's death.
    - Closed on 6/11/2004 due to Ronald Reagan's death.
    - Closed on 1/2/2007 due to Gerald Ford's death.
    - Closed at 1:00 PM on Wednesday, July 3rd, 2013
    - Closed at 1:00 PM on Friday, December 31, 1999
    - Closed at 1:00 PM on Friday, December 26, 1997
    - Closed at 1:00 PM on Friday, December 26, 2003

    NOTE: The exchange was **not** closed early on Friday December 26, 2008,
    nor was it closed on Friday December 26, 2014. The next Thursday Christmas
    will be in 2025.  If someone is still maintaining this code in 2025, then
    we've done alright...and we should check if it's a half day.
    """

    exchange_name = 'NYSE'
    native_timezone = US_EASTERN
    open_time = NYSE_OPEN
    close_time = NYSE_CLOSE
    open_offset = NYSE_OPEN_OFFSET
    close_offset = NYSE_CLOSE_OFFSET

    holidays_calendar = NYSEHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = [
        (NYSE_STANDARD_EARLY_CLOSE, NYSEEarlyCloseCalendar()),
        (time(14), NYSE2PMCloseCalendar()),
    ]

    holidays_adhoc = list(chain(
        September11Closings,
        HurricaneSandyClosings,
        USNationalDaysofMourning,
    ))

    special_opens_adhoc = ()
    special_closes_adhoc = [
        (NYSE_STANDARD_EARLY_CLOSE, ('1997-12-26',
                                     '1999-12-31',
                                     '2003-12-26',
                                     '2013-07-03')),
    ]

    @property
    def name(self):
        """
        The name of this exchange calendar.
        E.g.: 'NYSE', 'LSE', 'CME Energy'
        """
        return self.exchange_name

    @property
    def tz(self):
        """
        The native timezone of the exchange.
        """
        return self.native_timezone

    def is_open_on_minute(self, dt):
        """
        Is the exchange open (accepting orders) at @dt.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at the given dt, otherwise False.
        """
        # Retrieve the exchange session relevant for this datetime
        session = self.session_date(dt)
        # Retrieve the open and close for this exchange session
        open, close = self.open_and_close(session)
        # Is @dt within the trading hours for this exchange session
        return open <= dt and dt <= close

    def is_open_on_day(self, dt):
        """
        Is the exchange open (accepting orders) anytime during the calendar day
        containing @dt.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at any time during the day containing @dt
        """
        dt_normalized = normalize_date(dt)
        return dt_normalized in self.schedule.index

    def trading_days(self, start, end):
        """
        Calculates all of the exchange sessions between the given
        start and end, inclusive.

        SD: Should @start and @end are UTC-canonicalized, as our exchange
        sessions are. If not, then it's not clear how this method should behave
        if @start and @end are both in the middle of the day. Here, I assume we
        need to map @start and @end to session.

        Parameters
        ----------
        start : Timestamp
        end : Timestamp

        Returns
        -------
        DatetimeIndex
            A DatetimeIndex populated with all of the trading days between
            the given start and end.
        """
        start_session = self.session_date(start)
        end_session = self.session_date(end)
        # Increment end_session by one day, beucase .loc[s:e] return all values
        # in the DataFrame up to but not including `e`.
        # end_session += Timedelta(days=1)
        return self.schedule.loc[start_session:end_session]

    def open_and_close(self, dt):
        """
        Given a datetime, returns a tuple of timestamps of the
        open and close of the exchange session containing the datetime.

        SD: Should we accept an arbitrary datetime, or should we first map it
        to and exchange session using session_date. Need to check what the
        consumers expect. Here, I assume we need to map it to a session.

        Parameters
        ----------
        dt : Timestamp
            A dt in a session whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp)
            The open and close for the given dt.
        """
        session = self.session_date(dt)
        return self._get_open_and_close(session)

    def _get_open_and_close(self, session_date):
        """
        Retrieves the open and close for a given session.

        Parameters
        ----------
        session_date : Timestamp
            The canonicalized session_date whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp) or (None, None)
            The open and close for the given dt, or Nones if the given date is
            not a session.
        """
        # Return a tuple of nones if the given date is not a session.
        if session_date not in self.schedule.index:
            return (None, None)

        o_and_c = self.schedule.loc[session_date]
        # `market_open` and `market_close` should be timezone aware, but pandas
        # 0.16.1 does not appear to support this:
        # http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#datetime-with-tz  # noqa
        return (o_and_c['market_open'].tz_localize('UTC'),
                o_and_c['market_close'].tz_localize('UTC'))

    def session_date(self, dt):
        """
        Given a datetime, returns the UTC-canonicalized date of the exchange
        session in which the time belongs. If the time is not in an exchange
        session (while the market is closed), returns the date of the next
        exchange session after the time.

        Parameters
        ----------
        dt : Timestamp
            A timezone-aware Timestamp.

        Returns
        -------
        Timestamp
            The date of the exchange session in which dt belongs.
        """
        while not self.is_open_on_day(dt):
            dt += Timedelta(days=1)
        return normalize_date(dt)
