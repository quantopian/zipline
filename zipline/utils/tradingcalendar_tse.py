#
# Copyright 2013 Quantopian, Inc.
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


import pandas as pd
import pytz

from datetime import datetime
from dateutil import rrule
from zipline.utils.tradingcalendar import TradingCalendar

from datetime import timedelta

#Days in Environment but not in Calendar (using ^GSPTSE as bm_symbol):
#--------------------------------------------------------------------
#Used http://web.tmxmoney.com/pricehistory.php?qm_page=61468&qm_symbol=^TSX
#to check whether exchange was open on these days.
#1994-07-01     - July 1st, Yahoo Finance has Volume = 0
#1996-07-01     - July 1st, Yahoo Finance has Volume = 0
#1996-08-05     - Civic Holiday, Yahoo Finance has Volume = 0
#1997-07-01     - July 1st, Yahoo Finance has Volume = 0
#1997-08-04     - Civic Holiday, Yahoo Finance has Volume = 0
#2001-05-21     - Victoria day, Yahoo Finance has Volume = 0
#2004-10-11     - Closed, Thanksgiving - Confirmed closed
#2004-12-28     - Closed, Boxing Day - Confirmed closed
#2012-10-08     - Closed, Thanksgiving - Confirmed closed

#Days in Calendar but not in Environment using ^GSPTSE as bm_symbol:
#--------------------------------------------------------------------
#Used http://web.tmxmoney.com/pricehistory.php?qm_page=61468&qm_symbol=^TSX
#to check whether exchange was open on these days.
#2000-06-28     - No data this far back, can't confirm
#2000-08-28     - No data this far back, can't confirm
#2000-08-29     - No data this far back, can't confirm
#2001-09-11     - TSE Open for 71 min.
#2002-02-01     - Confirm TSE Open
#2002-06-14     - Confirm TSE Open
#2002-07-02     - Confirm TSE Open
#2002-11-11     - TSX website has no data for 2 weeks in 2002
#2003-07-07     - Confirm TSE Open
#2003-12-16     - Confirm TSE Open


class TseTradingCalendar(TradingCalendar):

    DEFAULT_OPEN_TIME = timedelta(hours=9, minutes=31)
    DEFAULT_CLOSE_TIME = timedelta(hours=16)
    EARLY_CLOSE_TIME = timedelta(hours=13)

    DEFAULT_OPEN_CLOSE_TIMES = (DEFAULT_OPEN_TIME, DEFAULT_CLOSE_TIME)
    EARLY_OPEN_CLOSE_TIMES = (DEFAULT_OPEN_TIME, EARLY_CLOSE_TIME)

    def __init__(self, start=pd.Timestamp('1994-01-01', tz='UTC'), end=None):
            super(TseTradingCalendar, self).__init__(start, end, 'US/Eastern')

    def get_non_trading_ruleset(self, start, end):
        non_trading_rules = []

        weekends = rrule.rrule(
            rrule.YEARLY,
            byweekday=(rrule.SA, rrule.SU),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(weekends)

        new_years = rrule.rrule(
            rrule.MONTHLY,
            byyearday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(new_years)

        new_years_sunday = rrule.rrule(
            rrule.MONTHLY,
            byyearday=2,
            byweekday=rrule.MO,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(new_years_sunday)

        new_years_saturday = rrule.rrule(
            rrule.MONTHLY,
            byyearday=3,
            byweekday=rrule.MO,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(new_years_saturday)

        # Family day in Ontario, starting in 2008, third monday of February
        family_day = rrule.rrule(
            rrule.MONTHLY,
            bymonth=2,
            byweekday=(rrule.MO(3)),
            cache=True,
            dtstart=datetime(2008, 1, 1, tzinfo=pytz.utc),
            until=end
        )
        non_trading_rules.append(family_day)

        good_friday = rrule.rrule(
            rrule.DAILY,
            byeaster=-2,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(good_friday)

        #Monday prior to May 25th.
        victoria_day = rrule.rrule(
            rrule.MONTHLY,
            bymonth=5,
            byweekday=rrule.MO,
            bymonthday=[24, 23, 22, 21, 20, 19, 18],
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(victoria_day)

        july_1st = rrule.rrule(
            rrule.MONTHLY,
            bymonth=7,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(july_1st)

        july_1st_sunday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=7,
            bymonthday=2,
            byweekday=rrule.MO,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(july_1st_sunday)

        july_1st_saturday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=7,
            bymonthday=3,
            byweekday=rrule.MO,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(july_1st_saturday)

        civic_holiday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=8,
            byweekday=rrule.MO(1),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(civic_holiday)

        labor_day = rrule.rrule(
            rrule.MONTHLY,
            bymonth=9,
            byweekday=(rrule.MO(1)),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(labor_day)

        thanksgiving = rrule.rrule(
            rrule.MONTHLY,
            bymonth=10,
            byweekday=(rrule.MO(2)),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(thanksgiving)

        christmas = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=25,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(christmas)

        # If Christmas is a Sunday then the 26th, a Monday is observed.
        # (but that would be boxing day), so the 27th is also observed.
        christmas_sunday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=27,
            byweekday=rrule.TU,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(christmas_sunday)

        # If Christmas is a Saturday then the 27th, a monday is observed.
        christmas_saturday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=27,
            byweekday=rrule.MO,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(christmas_saturday)

        boxing_day = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=26,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(boxing_day)

        #if boxing day is a sunday, the Christmas was saturday.
        # Christmas is observed on the 27th, a month and boxing day is observed
        # on the 28th, a tuesday.
        boxing_day_sunday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=28,
            byweekday=rrule.TU,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(boxing_day_sunday)

        # If boxing day is a Saturday then the 28th, a monday is observed.
        boxing_day_saturday = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=28,
            byweekday=rrule.MO,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(boxing_day_saturday)

        non_trading_ruleset = rrule.rruleset()

        for rule in non_trading_rules:
            non_trading_ruleset.rrule(rule)

        return non_trading_ruleset

    def get_exceptional_non_trading_days(self):
        non_trading_days = []

        # Add September 11th closings
        # The TSX was open for 71 minutes on September 11, 2011.
        # It was closed on the 12th and reopened on the 13th.
        # http://www.cbc.ca/news2/interactives/map-tsx/
        #
        #    September 2001
        # Su Mo Tu We Th Fr Sa
        #                    1
        #  2  3  4  5  6  7  8
        #  9 10 11 12 13 14 15
        # 16 17 18 19 20 21 22
        # 23 24 25 26 27 28 29
        # 30

        non_trading_days.append(
            datetime(2001, 9, 12, tzinfo=pytz.utc))

        return non_trading_days

    def get_early_closes(self, start, end):
        # TSX closed at 1:00 PM on december 24th.

        start = self.canonicalize_datetime(start)
        end = self.canonicalize_datetime(end)

        start = max(start, datetime(1993, 1, 1, tzinfo=pytz.utc))
        end = max(end, datetime(1993, 1, 1, tzinfo=pytz.utc))

        # Not included here are early closes prior to 1993
        # or unplanned early closes

        early_close_rules = []

        christmas_eve = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=24,
            byweekday=(rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR),
            cache=True,
            dtstart=start,
            until=end
        )
        early_close_rules.append(christmas_eve)

        early_close_ruleset = rrule.rruleset()

        for rule in early_close_rules:
            early_close_ruleset.rrule(rule)
        early_closes = early_close_ruleset.between(start, end, inc=True)

        early_closes.sort()
        return pd.DatetimeIndex(early_closes)

    def get_exceptional_open_and_close_times(self, day):
        return self.EARLY_OPEN_CLOSE_TIMES

    def get_default_open_close_times(self):
        return self.DEFAULT_OPEN_CLOSE_TIMES
