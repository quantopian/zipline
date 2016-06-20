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

from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    GoodFriday,
    USLaborDay,
    USPresidentsDay,
    USThanksgivingDay,
)
from pytz import timezone

from .trading_calendar import TradingCalendar

from .us_holidays import (
    USNewYearsDay,
    USMartinLutherKingJrAfter1998,
    USMemorialDay,
    USIndependenceDay,
    Christmas,
    MonTuesThursBeforeIndependenceDay,
    FridayAfterIndependenceDayExcept2013,
    USBlackFridayBefore1993,
    USBlackFridayInOrAfter1993,
    September11Closings,
    HurricaneSandyClosings,
    USNationalDaysofMourning,
    ChristmasEveBefore1993,
    ChristmasEveInOrAfter1993,
)

# Useful resources for making changes to this file:
# http://www.nyse.com/pdfs/closings.pdf
# http://www.stevemorse.org/jcal/whendid.html

US_EASTERN = timezone('US/Eastern')
NYSE_OPEN = time(9, 31)
NYSE_CLOSE = time(16)
NYSE_STANDARD_EARLY_CLOSE = time(13)

# Whether market opens or closes on a different calendar day, compared to the
# calendar day assigned by the exchange to this session.
NYSE_OPEN_OFFSET = 0
NYSE_CLOSE_OFFSET = 0


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


class NYSEExchangeCalendar(TradingCalendar):
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

    name = "NYSE"
    tz = US_EASTERN
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
