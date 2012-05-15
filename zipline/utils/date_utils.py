from collections import namedtuple

import time
import pytz
import iso8601
import calendar
from dateutil import rrule
from datetime import datetime, date, timedelta
from dateutil.relativedelta import *

# Datetime Tuple
# --------------
d_tuple = namedtuple('dt', ['year', 'month', 'day', 'hour', 'minute', 'second', 'micros'])

# iso8061 utility
# ---------------------
def parse_iso8061(date_string):
    dt = iso8601.parse_date(date_string)
    dt = dt.replace(tzinfo = pytz.utc)
    return dt

# Epoch utilities
# ---------------------
UNIX_EPOCH = datetime(1970, 1, 1, 0, 0, tzinfo = pytz.utc)
def EPOCH(utc_datetime):
    """
    The key is to ensure all the dates you are using are in the utc timezone 
    before you start converting. See http://pytz.sourceforge.net/ to learn how 
    to do that properly. By normalizing to utc, you eliminate the ambiguity of 
    daylight savings transitions. Then you can safely use timedelta to calculate 
    distance from the unix epoch, and then convert to seconds or milliseconds.
    
    Note that the resulting unix timestamp is itself in the UTC timezone. If you 
    wish to see the timestamp in a localized timezone, you will need to make 
    another conversion.
    
    Also note that this will only work for dates after 1970.
    """
    assert isinstance(utc_datetime, datetime)
    # utc only please
    assert utc_datetime.tzinfo == pytz.utc
    
    # how long since the epoch?
    delta = utc_datetime - UNIX_EPOCH
    seconds = delta.total_seconds()
    ms = seconds * 1000
    return ms
    
def UN_EPOCH(ms_since_epoch):
    seconds_since_epoch = ms_since_epoch / 1000
    delta = timedelta(seconds = seconds_since_epoch)
    dt = UNIX_EPOCH + delta
    return dt
    
def iso8061_to_epoch(datestring):
    dt = parse_iso8061(datestring)
    return EPOCH(dt)
    
def epoch_now():
    dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    return EPOCH(dt)

# UTC Datetime Subclasses
# -----------------------
def utcnow():
    return datetime.now(pytz.utc)

class utcdatetime(datetime):
    def __new__(cls, *args, **kwargs):
        kwargs['tzinfo'] = pytz.utc
        dt = datetime.__new__(cls, *args, **kwargs)
        return dt


    
# Datetime Calculations
# ---------------------

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
    cache=True,
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

    # Its now cached so if we do that traversal again it only
    # takes like 1e-5 seconds.
    tic = time.time()
    for day in trading_days(now, now30):
        print day
    print time.time() - tic
