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

from pandas.tseries.holiday import AbstractHolidayCalendar
from pytz import timezone

# Useful resources for making changes to this file:
# http://www.cmegroup.com/tools-information/holiday-calendar.html

from .trading_calendar import TradingCalendar

from .us_holidays import (
    USNewYearsDay,
    Christmas,
    ChristmasEveBefore1993,
    ChristmasEveInOrAfter1993,
    FridayAfterIndependenceDayExcept2013,
    MonTuesThursBeforeIndependenceDay,
    USBlackFridayInOrAfter1993,
    September11Closings,
    USNationalDaysofMourning
)

US_CENTRAL = timezone('America/Chicago')
CME_OPEN = time(17)
CME_CLOSE = time(16)

# The CME seems to have different holiday rules depending on the type
# of instrument.  For example, http://www.cmegroup.com/tools-information/holiday-calendar/files/2016-4th-of-july-holiday-schedule.pdf # noqa
# shows that Equity, Interest Rate, FX, Energy, Metals & DME Products close at
# 1200 CT on July 4, 2016, while Grain, Oilseed & MGEX Products and Livestock,
# Dairy & Lumber products are completely closed.

# For now, we will treat the CME as having a single calendar, and just go with
# the most conservative hours - and treat July 4 as an early close at noon.
CME_STANDARD_EARLY_CLOSE = time(12)

# Does the market open or close on a different calendar day, compared to the
# calendar day assigned by the exchange to this session?
CME_OPEN_OFFSET = -1
CME_CLOSE_OFFSET = -0


class CMEHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the CME.

    See CMEExchangeCalendar for full description.
    """
    rules = [
        USNewYearsDay,
        Christmas,
    ]


class CMEEarlyCloseCalendar(AbstractHolidayCalendar):
    """
    Regular early close calendar for NYSE
    """
    rules = [
        MonTuesThursBeforeIndependenceDay,
        FridayAfterIndependenceDayExcept2013,
        USBlackFridayInOrAfter1993,
        ChristmasEveBefore1993,
        ChristmasEveInOrAfter1993,
    ]


class CMEExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for CME

    Open Time: 5:00 PM, America/Chicago
    Close Time: 5:00 PM, America/Chicago

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

    NOTE: For the following US Federal Holidays, part of the CME is closed
    (Foreign Exchange, Interest Rates) but Commodities, GSCI, Weather & Real
    Estate is open.  Thus, we don't treat these as holidays.
    - Columbus Day
    - Veterans Day

    Regularly-Observed Early Closes:
    - Christmas Eve (except on Fridays, when the exchange is closed entirely)
    - Day After Thanksgiving (aka Black Friday, observed from 1992 onward)

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

    name = "CME"
    tz = US_CENTRAL
    open_time = CME_OPEN
    close_time = CME_CLOSE
    open_offset = CME_OPEN_OFFSET
    close_offset = CME_CLOSE_OFFSET

    holidays_calendar = CMEHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = []

    holidays_adhoc = list(chain(
        September11Closings,
        USNationalDaysofMourning,
    ))

    special_opens_adhoc = ()
    special_closes_adhoc = []
