from datetime import time

from pandas import Timestamp
from pandas.tseries.holiday import GoodFriday
from pytz import timezone

from zipline.utils.calendars import TradingCalendar
from zipline.utils.calendars.trading_calendar import (
    HolidayCalendar, end_default
)
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
    # XXX: Override the default TradingCalendar start and end dates with ones
    # further in the future. This is a stopgap for memory issues caused by
    # upgrading to pandas 18. This calendar is the most severely affected,
    # since it has the most total minutes of any of the zipline calendars.
    def __init__(self,
                 start=Timestamp('2000-01-01', tz='UTC'),
                 end=end_default):
        super(QuantopianUSFuturesCalendar, self).__init__(start=start, end=end)

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
