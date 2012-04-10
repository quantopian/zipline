from collections import namedtuple

import pytz
import calendar
from dateutil import rrule
from datetime import datetime, date, timedelta
from dateutil.relativedelta import *

# Datetime Tuple
d_tuple = namedtuple('dt', ['year', 'month', 'day', 'hour', 'minute', 'second', 'micros'])

WEEKDAYS = [rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR]

HOLIDAYS = {
    'new_years'    : datetime(2008 , 1  , 1 ),
    'mlk_day'      : datetime(2008 , 1  , 21),
    'presidents'   : datetime(2008 , 2  , 18),
    'good_friday'  : datetime(2008 , 3  , 21),
    'memorial_day' : datetime(2008 , 5  , 26),
    'july_4th'     : datetime(2008 , 7  , 4 ),
    'labor_day'    : datetime(2008 , 9  , 1 ),
    'tgiving'      : datetime(2008 , 11 , 27),
    'christmas'    : datetime(2008 , 5  , 25),
}

# Create a rule to recur every weekday starting today
rule = rrule.rrule(
    rrule.DAILY,
    byweekday=WEEKDAYS,
)

# Precompute the rule, so that dates are cached.
rs = rrule.rruleset()
rs.rrule(rule)

# Add holidays as exclusion days
for holiday in HOLIDAYS.itervalues():
    rs.exdate(holiday)

def trading_days(after, before, inclusive=False):
    """
    Iterates over the NYSE trading days between the two given
    dates.
    """
    return rs.between(after, before, inc=inclusive)

if __name__ == '__main__':

    now = datetime.now()
    now30 = datetime.now() + timedelta(days=30)

    # Iterate over the trading days between any two arbitrary
    # days, excluding the preset holidays.
    for day in trading_days(now, now30):
        print day
