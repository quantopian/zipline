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

from pandas.tseries.holiday import (
    USPresidentsDay,
    USLaborDay,
    USThanksgivingDay,
    GoodFriday
)
from pytz import timezone

# Useful resources for making changes to this file:
# http://www.cmegroup.com/tools-information/holiday-calendar.html

from .trading_calendar import TradingCalendar, HolidayCalendar
from .us_holidays import (
    USNewYearsDay,
    Christmas,
    ChristmasEveBefore1993,
    ChristmasEveInOrAfter1993,
    USBlackFridayInOrAfter1993,
    USNationalDaysofMourning,
    USMartinLutherKingJrAfter1998,
    USMemorialDay,
    USIndependenceDay)


class CMEExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for CME

    Open Time: 5:00 PM, America/Chicago
    Close Time: 5:00 PM, America/Chicago

    Regularly-Observed Holidays:
    - New Years Day
    - Good Friday
    - Christmas
    """
    @property
    def name(self):
        return "CME"

    @property
    def tz(self):
        return timezone('America/Chicago')

    @property
    def open_time(self):
        return time(17, 1)

    @property
    def close_time(self):
        return time(17)

    @property
    def open_offset(self):
        return -1

    @property
    def regular_holidays(self):
        # The CME has different holiday rules depending on the type of
        # instrument. For example, http://www.cmegroup.com/tools-information/holiday-calendar/files/2016-4th-of-july-holiday-schedule.pdf # noqa
        # shows that Equity, Interest Rate, FX, Energy, Metals & DME Products
        # close at 1200 CT on July 4, 2016, while Grain, Oilseed & MGEX
        # Products and Livestock, Dairy & Lumber products are completely
        # closed.

        # For now, we will treat the CME as having a single calendar, and just
        # go with the most conservative hours - and treat July 4 as an early
        # close at noon.
        return HolidayCalendar([
            USNewYearsDay,
            GoodFriday,
            Christmas,
        ])

    @property
    def adhoc_holidays(self):
        return USNationalDaysofMourning

    @property
    def special_closes(self):
        return [(
            time(12),
            HolidayCalendar([
                USMartinLutherKingJrAfter1998,
                USPresidentsDay,
                USMemorialDay,
                USLaborDay,
                USIndependenceDay,
                USThanksgivingDay,
                USBlackFridayInOrAfter1993,
                ChristmasEveBefore1993,
                ChristmasEveInOrAfter1993,
            ])
        )]
