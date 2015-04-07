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

from zipline.utils.tradingcalendar import ExchangeCalendar

# Useful resources for making changes to this file:
# http://www.nyse.com/pdfs/closings.pdf
# http://www.stevemorse.org/jcal/whendid.html

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)

US_EASTERN = timezone('US/Eastern')
NYSE_OPEN = time(9, 31)
NYSE_CLOSE = time(16)
NYSE_STANDARD_EARLY_CLOSE = time(13)

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
    - July 3rd (Mondays, Tuesdays, and Thursdays)
    - July 5th (Fridays, except 2013)
    - Christmas Eve (except on Fridays, when the exchange is closed entirely)
    - Day After Thanksgiving (Black Friday)

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

    native_timezone = US_EASTERN
    open_time = NYSE_OPEN
    close_time = NYSE_CLOSE

    holidays_calendar = NYSEHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = [
        (NYSE_STANDARD_EARLY_CLOSE, NYSEEarlyCloseCalendar()),
        (time(14), NYSE2PMCloseCalendar()),
    ]

    holidays_adhoc = chain(
        September11Closings,
        HurricaneSandyClosings,
        USNationalDaysofMourning,
    )

    special_opens_adhoc = ()
    special_closes_adhoc = [
        # SS: Normally I'm strongly against this formatting style, but it's
        # WAAAY clearer here than indenting the list of dates another level.
        (NYSE_STANDARD_EARLY_CLOSE, ('1997-12-26',
                                     '1999-12-31',
                                     '2003-12-26',
                                     '2013-07-03')),
    ]
