from datetime import time

from pandas.tseries.holiday import (
    USPresidentsDay,
    USLaborDay,
    USThanksgivingDay,
    GoodFriday
)
from pytz import timezone

from zipline.utils.calendars import TradingCalendar
from zipline.utils.calendars.trading_calendar import HolidayCalendar
from zipline.utils.calendars.us_holidays import (
    USMartinLutherKingJrAfter1998,
    USMemorialDay,
    USBlackFridayInOrAfter1993,
    USNewYearsDay,
    USIndependenceDay,
    Christmas
)


class CFEExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for the CBOE Futures Exchange (CFE).

    http://cfe.cboe.com/aboutcfe/expirationcalendar.aspx

    Open Time: 8:30am, America/Chicago
    Close Time: 3:15pm, America/Chicago

    (We are ignoring extended trading hours for now)
    """
    @property
    def name(self):
        return "CFE"

    @property
    def tz(self):
        return timezone("America/Chicago")

    @property
    def open_time(self):
        return time(8, 31)

    @property
    def close_time(self):
        return time(15, 15)

    @property
    def regular_holidays(self):
        return HolidayCalendar([
            USNewYearsDay,
            USMartinLutherKingJrAfter1998,
            USPresidentsDay,
            GoodFriday,
            USIndependenceDay,
            USMemorialDay,
            USLaborDay,
            USThanksgivingDay,
            Christmas
        ])

    @property
    def special_closes(self):
        return [(
            time(12, 15),
            HolidayCalendar([
                USBlackFridayInOrAfter1993,
            ])
        )]
