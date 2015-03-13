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

from zipline.utils.tradingcalendar import end, canonicalize_datetime

start = datetime(2002, 1, 1, tzinfo=pytz.utc)


def get_non_trading_days(start, end):
    non_trading_rules = []

    start = canonicalize_datetime(start)
    end = canonicalize_datetime(end)

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
    non_trading_rules.append(new_year)
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
    # Labour Day (1st of May)
    may_bank = rrule.rrule(
        rrule.MONTHLY,
        bymonth=5,
        bymonthday=1,
        cache=True,
        dtstart=start,
        until=end
    )
    non_trading_rules.append(may_bank)
    # Christmas Eve
    christmas_eve = rrule.rrule(
        rrule.MONTHLY,
        bymonth=12,
        bymonthday=24,
        cache=True,
        dtstart=start,
        until=end
    )
    non_trading_rules.append(christmas_eve)
    # Christmas Day
    christmas = rrule.rrule(
        rrule.MONTHLY,
        bymonth=12,
        bymonthday=25,
        cache=True,
        dtstart=start,
        until=end
    )
    non_trading_rules.append(christmas)
    # Boxing Day
    boxing_day = rrule.rrule(
        rrule.MONTHLY,
        bymonth=12,
        bymonthday=26,
        cache=True,
        dtstart=start,
        until=end
    )
    non_trading_rules.append(boxing_day)
    # New Year's Eve
    newyears_eve = rrule.rrule(
        rrule.MONTHLY,
        bymonth=12,
        bymonthday=31,
        cache=True,
        dtstart=start,
        until=end
    )
    non_trading_rules.append(newyears_eve)
    non_trading_ruleset = rrule.rruleset()
    for rule in non_trading_rules:
        non_trading_ruleset.rrule(rule)
    non_trading_days = non_trading_ruleset.between(start, end, inc=True)

    non_trading_days.sort()
    return pd.DatetimeIndex(non_trading_days)

non_trading_days = get_non_trading_days(start, end)
trading_day = pd.tseries.offsets.CDay(holidays=non_trading_days)


def get_trading_days(start, end, trading_day=trading_day):
    return pd.date_range(start=start.date(),
                         end=end.date(),
                         freq=trading_day).tz_localize('UTC')

trading_days = get_trading_days(start, end)


def get_early_closes(start, end):
    return []

early_closes = get_early_closes(start, end)


def get_open_and_close(day, early_closes):
    market_open = pd.Timestamp(
        datetime(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=8,
            minute=00),
        tz='Europe/Berlin').tz_convert('UTC')
    close_hour = 18
    market_close = pd.Timestamp(
        datetime(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=close_hour),
        tz='Europe/Berlin').tz_convert('UTC')

    return market_open, market_close


def get_open_and_closes(trading_days, early_closes):
    open_and_closes = pd.DataFrame(index=trading_days,
                                   columns=('market_open', 'market_close'))
    for day in trading_days:
        market_open = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=9,
                minute=31),
            tz='Europe/Berlin').tz_convert('UTC')
        # 1 PM if early close, 4 PM otherwise
        close_hour = 18
        market_close = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=close_hour),
            tz='Europe/Berlin').tz_convert('UTC')

        open_and_closes.loc[day, 'market_open'] = market_open
        open_and_closes.loc[day, 'market_close'] = market_close

    return open_and_closes

open_and_closes = get_open_and_closes(trading_days, early_closes)
