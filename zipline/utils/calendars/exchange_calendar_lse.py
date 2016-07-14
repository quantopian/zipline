from datetime import time
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
    DateOffset,
    MO,
    weekend_to_monday,
    GoodFriday,
    EasterMonday,
)
from pytz import timezone

from .trading_calendar import (
    TradingCalendar,
    MONDAY,
    TUESDAY,
)

# New Year's Day
LSENewYearsDay = Holiday(
    "New Year's Day",
    month=1,
    day=1,
    observance=weekend_to_monday,
)
# Early May bank holiday
MayBank = Holiday(
    "Early May Bank Holiday",
    month=5,
    offset=DateOffset(weekday=MO(1)),
)
# Spring bank holiday
SpringBank = Holiday(
    "Spring Bank Holiday",
    month=5,
    day=31,
    offset=DateOffset(weekday=MO(-1)),
)
# Summer bank holiday
SummerBank = Holiday(
    "Summer Bank Holiday",
    month=8,
    day=31,
    offset=DateOffset(weekday=MO(-1)),
)
# Christmas
Christmas = Holiday(
    "Christmas",
    month=12,
    day=25,
)
# If christmas day is Saturday Monday 27th is a holiday
# If christmas day is sunday the Tuesday 27th is a holiday
WeekendChristmas = Holiday(
    "Weekend Christmas",
    month=12,
    day=27,
    days_of_week=(MONDAY, TUESDAY),
)
# Boxing day
BoxingDay = Holiday(
    "Boxing Day",
    month=12,
    day=26,
)
# If boxing day is saturday then Monday 28th is a holiday
# If boxing day is sunday then Tuesday 28th is a holiday
WeekendBoxingDay = Holiday(
    "Weekend Boxing Day",
    month=12,
    day=28,
    days_of_week=(MONDAY, TUESDAY),
)


class LSEHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the LSE.

    See NYSEExchangeCalendar for full description.
    """
    rules = [
        LSENewYearsDay,
        GoodFriday,
        EasterMonday,
        MayBank,
        SpringBank,
        SummerBank,
        Christmas,
        WeekendChristmas,
        BoxingDay,
        WeekendBoxingDay,
    ]


class LSEExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for the London Stock Exchange

    Open Time: 8:00 AM, GMT
    Close Time: 4:30 PM, GMT

    Regularly-Observed Holidays:
    - New Years Day (observed on first business day on/after)
    - Good Friday
    - Easter Monday
    - Early May Bank Holiday (first Monday in May)
    - Spring Bank Holiday (last Monday in May)
    - Summer Bank Holiday (last Monday in May)
    - Christmas Day
    - Dec. 27th (if Christmas is on a weekend)
    - Boxing Day
    - Dec. 28th (if Boxing Day is on a weekend)
    """

    name = 'LSE'
    tz = timezone('Europe/London')
    open_time = time(8, 1)
    close_time = time(16, 30)
    open_offset = 0
    close_offset = 0

    holidays_calendar = LSEHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = ()

    holidays_adhoc = ()

    special_opens_adhoc = ()
    special_closes_adhoc = ()
