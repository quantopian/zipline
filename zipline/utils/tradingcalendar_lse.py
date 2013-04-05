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


# References:
# http://www.londonstockexchange.com
# /about-the-exchange/company-overview/business-days/business-days.htm
# http://en.wikipedia.org/wiki/Bank_holiday
# http://www.adviceguide.org.uk/england/work_e/work_time_off_work_e/
# bank_and_public_holidays.htm

import pytz

import pandas as pd

from datetime import datetime
from dateutil import rrule
from zipline.utils.tradingcalendar import end

start = datetime(2002, 1, 1, tzinfo=pytz.utc)

non_trading_rules = []
# Weekends
weekends = rrule.rrule(
    rrule.YEARLY,
    byweekday=(rrule.SA, rrule.SU),
    cache=True,
    dtstart=start,
    until=end
)
non_trading_rules.append(weekends)
# New Year's Day
new_year = rrule.rrule(
    rrule.MONTHLY,
    byyearday=1,
    cache=True,
    dtstart=start,
    until=end
)
# If new years day is on Saturday then Monday 3rd is a holiday
# If new years day is on Sunday then Monday 2nd is a holiday
weekend_new_year = rrule.rrule(
    rrule.MONTHLY,
    bymonth=1,
    bymonthday=[2, 3],
    byweekday=(rrule.MO),
    cache=True,
    dtstart=start,
    until=end
)
non_trading_rules.append(new_year)
non_trading_rules.append(weekend_new_year)
# Good Friday
good_friday = rrule.rrule(
    rrule.DAILY,
    byeaster=-2,
    cache=True,
    dtstart=start,
    until=end
)
non_trading_rules.append(good_friday)
# Easter Monday
easter_monday = rrule.rrule(
    rrule.DAILY,
    byeaster=1,
    cache=True,
    dtstart=start,
    until=end
)
non_trading_rules.append(easter_monday)
# Early May Bank Holiday (1st Monday in May)
may_bank = rrule.rrule(
    rrule.MONTHLY,
    bymonth=5,
    byweekday=(rrule.MO(1)),
    cache=True,
    dtstart=start,
    until=end
)
non_trading_rules.append(may_bank)
# Spring Bank Holiday (Last Monday in May)
spring_bank = rrule.rrule(
    rrule.MONTHLY,
    bymonth=5,
    byweekday=(rrule.MO(-1)),
    cache=True,
    dtstart=datetime(2003, 1, 1, tzinfo=pytz.utc),
    until=end
)
non_trading_rules.append(spring_bank)
# Summer Bank Holiday (Last Monday in August)
summer_bank = rrule.rrule(
    rrule.MONTHLY,
    bymonth=8,
    byweekday=(rrule.MO(-1)),
    cache=True,
    dtstart=start,
    until=end
)
non_trading_rules.append(summer_bank)
# Christmas Day
christmas = rrule.rrule(
    rrule.MONTHLY,
    bymonth=12,
    bymonthday=25,
    cache=True,
    dtstart=start,
    until=end
)
# If christmas day is Saturday Monday 27th is a holiday
# If christmas day is sunday the Tuesday 27th is a holiday
weekend_christmas = rrule.rrule(
    rrule.MONTHLY,
    bymonth=12,
    bymonthday=27,
    byweekday=(rrule.MO, rrule.TU),
    cache=True,
    dtstart=start,
    until=end
)

non_trading_rules.append(christmas)
non_trading_rules.append(weekend_christmas)
# Boxing Day
boxing_day = rrule.rrule(
    rrule.MONTHLY,
    bymonth=12,
    bymonthday=26,
    cache=True,
    dtstart=start,
    until=end
)
# If boxing day is saturday then Monday 28th is a holiday
# If boxing day is sunday then Tuesday 28th is a holiday
weekend_boxing_day = rrule.rrule(
    rrule.MONTHLY,
    bymonth=12,
    bymonthday=28,
    byweekday=(rrule.MO, rrule.TU),
    cache=True,
    dtstart=start,
    until=end
)

non_trading_rules.append(boxing_day)
non_trading_rules.append(weekend_boxing_day)

non_trading_ruleset = rrule.rruleset()

# In 2002 May bank holiday was moved to 4th June to follow the Queens
# Golden Jubilee
non_trading_ruleset.exdate(datetime(2002, 9, 27, tzinfo=pytz.utc))
non_trading_ruleset.rdate(datetime(2002, 6, 3, tzinfo=pytz.utc))
non_trading_ruleset.rdate(datetime(2002, 6, 4, tzinfo=pytz.utc))
# TODO: not sure why Feb 18 2008 is not available in the yahoo data
non_trading_ruleset.rdate(datetime(2008, 2, 18, tzinfo=pytz.utc))
# In 2011 The Friday before Mayday was the Royal Wedding
non_trading_ruleset.rdate(datetime(2011, 4, 29, tzinfo=pytz.utc))
# In 2012 May bank holiday was moved to 4th June to preceed the Queens
# Diamond Jubilee
non_trading_ruleset.exdate(datetime(2012, 5, 28, tzinfo=pytz.utc))
non_trading_ruleset.rdate(datetime(2012, 6, 4, tzinfo=pytz.utc))
non_trading_ruleset.rdate(datetime(2012, 6, 5, tzinfo=pytz.utc))

for rule in non_trading_rules:
    non_trading_ruleset.rrule(rule)

non_trading_days = non_trading_ruleset.between(start, end, inc=True)
non_trading_day_index = pd.DatetimeIndex(sorted(non_trading_days))

business_days = pd.DatetimeIndex(start=start, end=end,
                                 freq=pd.datetools.BDay())

trading_days = business_days - non_trading_day_index
