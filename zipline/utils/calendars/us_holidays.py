from pandas import (
    Timestamp,
    DateOffset,
    date_range,
)

from pandas.tseries.holiday import (
    Holiday,
    sunday_to_monday,
    nearest_workday,
)

from dateutil.relativedelta import (
    MO,
    TH
)
from pandas.tseries.offsets import Day

from zipline.utils.calendars.trading_calendar import (
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
)

# These have the same definition, but are used in different places because the
# NYSE closed at 2:00 PM on Christmas Eve until 1993.
from zipline.utils.pandas_utils import july_5th_holiday_observance

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
USNewYearsDay = Holiday(
    'New Years Day',
    month=1,
    day=1,
    # When Jan 1 is a Sunday, US markets observe the subsequent Monday.
    # When Jan 1 is a Saturday (as in 2005 and 2011), no holiday is observed.
    observance=sunday_to_monday
)
USMartinLutherKingJrAfter1998 = Holiday(
    'Dr. Martin Luther King Jr. Day',
    month=1,
    day=1,
    # The US markets didn't observe MLK day as a holiday until 1998.
    start_date=Timestamp('1998-01-01'),
    offset=DateOffset(weekday=MO(3)),
)
USMemorialDay = Holiday(
    # NOTE: The definition for Memorial Day is incorrect as of pandas 0.16.0.
    # See https://github.com/pydata/pandas/issues/9760.
    'Memorial Day',
    month=5,
    day=25,
    offset=DateOffset(weekday=MO(1)),
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
    observance=july_5th_holiday_observance,
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
BattleOfGettysburg = Holiday(
    # All of the floor traders in Chicago were sent to PA
    'Markets were closed during the battle of Gettysburg',
    month=7,
    day=(1, 2, 3),
    start_date=Timestamp("1863-07-01"),
    end_date=Timestamp("1863-07-03")
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
