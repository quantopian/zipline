from datetime import time
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
    DateOffset,
    MO,
    weekend_to_monday,
    GoodFriday,
)
from pytz import timezone

from zipline.utils.calendars.trading_calendar import TradingCalendar
from zipline.utils.calendars.us_holidays import Christmas
from zipline.utils.calendars.exchange_calendar_lse import (
    WeekendChristmas,
    BoxingDay,
    WeekendBoxingDay,
)

# New Year's Day
TSXNewYearsDay = Holiday(
    "New Year's Day",
    month=1,
    day=1,
    observance=weekend_to_monday,
)
# Ontario Family Day
FamilyDay = Holiday(
    "Family Day",
    month=2,
    day=1,
    offset=DateOffset(weekday=MO(3)),
    start_date='2008-01-01',
)
# Victoria Day
VictoriaDay = Holiday(
    'Victoria Day',
    month=5,
    day=25,
    offset=DateOffset(weekday=MO(-1)),
)
# Canada Day
CanadaDay = Holiday(
    'Canada Day',
    month=7,
    day=1,
    observance=weekend_to_monday,
)
# Civic Holiday
CivicHoliday = Holiday(
    'Civic Holiday',
    month=8,
    day=1,
    offset=DateOffset(weekday=MO(1)),
)
# Labor Day
LaborDay = Holiday(
    'Labor Day',
    month=9,
    day=1,
    offset=DateOffset(weekday=MO(1)),
)
# Thanksgiving
Thanksgiving = Holiday(
    'Thanksgiving',
    month=10,
    day=1,
    offset=DateOffset(weekday=MO(2)),
)


class TSXHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the TSX.

    See NYSEExchangeCalendar for full description.
    """
    rules = [
        TSXNewYearsDay,
        FamilyDay,
        GoodFriday,
        VictoriaDay,
        CanadaDay,
        CivicHoliday,
        LaborDay,
        Thanksgiving,
        Christmas,
        WeekendChristmas,
        BoxingDay,
        WeekendBoxingDay,
    ]


class TSXExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for the Toronto Stock Exchange

    Open Time: 9:30 AM, EST
    Close Time: 4:00 PM, EST

    Regularly-Observed Holidays:
    - New Years Day (observed on first business day on/after)
    - Family Day (Third Monday in February after 2008)
    - Good Friday
    - Victoria Day (Monday before May 25th)
    - Canada Day (July 1st, observed first business day after)
    - Civic Holiday (First Monday in August)
    - Labor Day (First Monday in September)
    - Thanksgiving (Second Monday in October)
    - Christmas Day
    - Dec. 27th (if Christmas is on a weekend)
    - Boxing Day
    - Dec. 28th (if Boxing Day is on a weekend)
    """

    name = 'TSX'
    tz = timezone('Canada/Atlantic')
    open_time = time(9, 31)
    close_time = time(16)
    open_offset = 0
    close_offset = 0

    holidays_calendar = TSXHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = ()

    holidays_adhoc = ()

    special_opens_adhoc = ()
    special_closes_adhoc = ()
