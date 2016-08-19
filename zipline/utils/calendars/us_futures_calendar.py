from datetime import time

from pandas.tseries.holiday import GoodFriday
from pytz import timezone

from zipline.utils.calendars import TradingCalendar
from zipline.utils.calendars.trading_calendar import HolidayCalendar
from zipline.utils.calendars.us_holidays import (
    USNewYearsDay,
    Christmas
)


class QuantopianUSFuturesCalendar(TradingCalendar):
    """Synthetic calendar for trading US futures.

    This calendar is a superset of all of the US futures exchange
    calendars provided by Zipline (CFE, CME, ICE), and is intended for
    trading across all of these exchanges.

    Notes
    -----
    Open Time: 6:00 PM, US/Eastern
    Close Time: 6:00 PM, US/Eastern

    Regularly-Observed Holidays:
    - New Years Day
    - Good Friday
    - Christmas

    In order to align the hours of each session, we ignore the Sunday
    CME Pre-Open hour (5-6pm).
    """
    @property
    def name(self):
        return "us_futures"

    @property
    def tz(self):
        return timezone('US/Eastern')

    @property
    def open_time(self):
        return time(18, 1)

    @property
    def close_time(self):
        return time(18)

    @property
    def open_offset(self):
        return -1

    @property
    def regular_holidays(self):
        return HolidayCalendar([
            USNewYearsDay,
            GoodFriday,
            Christmas,
        ])
