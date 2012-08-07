import pytz

from datetime import datetime, timedelta
from dateutil import rrule
from zipline.utils.date_utils import utcnow

def market_opens(start, end, inclusive=False):
    """
    Returns all market opens between the start date and the end date.
    Must use utc-stamped datetimes.
    """
    return opens.between(start, end, inc=inclusive)

def market_closes(start, end, inclusive=False):
    """
    Returns all market closes between the start date and the end date.
    Must use utc-stamped datetimes.
    """
    return closes.between(start, end, inc=inclusive)

def trading_days_between(start, end):
    """
    Calculate the number of "complete" trading days between two
    events.  We define this as the number of market opens that
    occurred between start and end, with the caveat that we subtract 1
    from this total if end falls on the same day as the last market
    open and end occurs earlier in its own day than start.  This
    reflects the fact that we haven't completed a full day
    corresponding to the last market open.

    Examples:
    
    1.)
    start = Tuesday, Aug 7, 2012, 1:00 pm 
    end = Wednesday, Aug 8, 2012, 1:30 pm

    There is one market open between these dates, on the morning of
    Wednesday the 8th.  This falls on the same calendar day as end,
    but end is later in the day than start, so we count this as a full
    day.  The correct output is 1.

    2.)
    start = Tuesday, Aug 7, 2012, 1:30 pm 
    end = Wednesday, Aug 8, 2012, 1:00 pm
    
    There is one market open between these dayes, on the morning of
    Wednesday the 8th.  This falls on the same calendar day as end,
    and end is earlier in the day than start, so we do not count this
    day as completed.  The correct output is 0.

    3.)
    start = Tuesday, Aug 7, 2012, 1:00 pm 
    end = Saturday, Aug 11, 2012, 1:30 pm
    
    There are 3 market opens between these dates, occurring on
    Wednesday, Thursday, and Friday.  The last open is not on
    the same day as end, so we simply return 3

    4.)
    start = Tuesday, Aug 7, 2012, 1:30 pm 
    end = Monday, Aug, 13, 2012, 1:00 pm
    
    There are 4 market opens between these dates, occurring on
    Wednesday, Thursday, Friday, and the following Monday. The 
    last open occurs on the same calendar day as end, and end
    is earlier in the day than start, so we do not count the
    last market day as completed.  The correct output is 3 days.
    """
    # Calculate the number of opens between the events.
    opens = (market_opens(start, end))
    days_between = len(opens)
    if days_between == 0:
        return days_between
    
    # If end falls on the same day as an open, subtract 1 from the
    # total if end is earlier in its respective day than start.
    last_open = opens[-1]
    if last_open.date() == end.date() and earlier_in_day(end, start):
        days_between -=1
    
    return days_between

def earlier_in_day(d1, d2):
    """
    Return true if d1 falls earlier in its own day than d2.
    """
    return d1.time() < d2.time()

WEEKDAYS = [rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR]

# Recurrence rule that generates all market opens since Jan 1, 1970.
# This does not exclude holidays.
market_opens_with_holidays = rrule.rrule(
    rrule.DAILY,
    byweekday=WEEKDAYS,
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart=datetime(2000, 1, 1, tzinfo = pytz.utc),
    until=datetime(2014 , 1, 1, tzinfo = pytz.utc)
)

# Recurrence rule that generates all market closes since Jan 1, 1970.
# This does not exclude holidays.
market_closes_with_holidays = rrule.rrule(
    rrule.DAILY,
    byweekday=WEEKDAYS,
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart=datetime(2001, 1, 1, tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence rules for excluding the market open/close on new years.
new_years_opens = rrule.rrule(
    rrule.MONTHLY,
    byyearday = 1,
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
new_years_closes = rrule.rrule(
    rrule.MONTHLY,
    byyearday = 1,
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence rules for excluding MLK day. It is always the third
# monday in January.
mlk_opens = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 1,
    byweekday = (rrule.MO(3)),
    byhour = 14, 
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
mlk_closes = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 1,
    byweekday = (rrule.MO(+3)),
    byhour = 21, 
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
) 

# Recurrence rules for generating the market open/close for
# presidents' day.  Presidents' day always occurs on the third monday
# of February.
presidents_day_opens = rrule.rrule(
    rrule.MONTHLY, 
    bymonth = 2,
    byweekday = (rrule.MO(3)),
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
presidents_day_closes = rrule.rrule(
    rrule.MONTHLY, 
    bymonth = 2,
    byweekday = (rrule.MO(3)),
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence rules for generating the market open/close for good
# friday.  Good friday always falls 2 days before easter, which
# thankfully is a built-in refernce in this module.
good_friday_opens = rrule.rrule(
    rrule.DAILY,
    byeaster = -2,
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
good_friday_closes = rrule.rrule(
    rrule.DAILY,
    byeaster = -2,
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
) 

# Recurrence rules for generating the market open/close for memorial
# day. Memorial day always occurs on the last monday of May.
memorial_day_opens = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 5,
    byweekday = (rrule.MO(-1)),
    byhour = 14, 
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
memorial_day_closes = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 5,
    byweekday = (rrule.MO(-1)),
    byhour = 21, 
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence rules for generating the market open/close for July 4th.
july_4th_opens = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 6,
    bymonthday = 4,
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
july_4th_closes = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 6,
    bymonthday = 4,
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence rule for generating the market open/close for labor day.
# Labor day is always the first monday of September.
labor_day_opens = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 9,
    byweekday = (rrule.MO(1)),
    byhour = 14, 
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
labor_day_closes = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 9,
    byweekday = (rrule.MO(1)),
    byhour = 21, 
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence rule for generating the market open/close for
# thanksgiving.  Thanksgiving always falls on the fourth thursday in
# November. (Who decides how these holidays work!?!)
thanksgiving_opens = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 11,
    byweekday = (rrule.TH(-1)),
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
thanksgiving_closes = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 11,
    byweekday = (rrule.TH(-1)),
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# Recurrence relation for generating the market open/close for
# christmas.  Christmas always occurs on december 25th.

christmas_opens = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 12,
    bymonthday = 25,
    byhour = 14,
    byminute = 30,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)
christmas_closes = rrule.rrule(
    rrule.MONTHLY,
    bymonth = 12,
    bymonthday = 25,
    byhour = 21,
    byminute = 0,
    cache = True,
    dtstart = datetime(2000, 1,1,tzinfo = pytz.utc),
    until=datetime(2014, 1, 1, tzinfo = pytz.utc)
)

# All NYSE observed holidays.
holiday_opens = [
    new_years_opens,
    mlk_opens,
    presidents_day_opens,
    good_friday_opens,
    memorial_day_opens,
    july_4th_opens,
    labor_day_opens,
    thanksgiving_opens,
    christmas_opens
]
holiday_closes = [
    new_years_closes,
    mlk_closes,
    presidents_day_closes,
    good_friday_closes,
    memorial_day_closes,
    july_4th_closes,
    labor_day_closes,
    thanksgiving_closes,
    christmas_closes
]

# Valid market opens are given by all market opens minus holidays.
opens = rrule.rruleset(cache=True)
opens.rrule(market_opens_with_holidays)
for holiday_rule in holiday_opens:
    opens.exrule(holiday_rule)
open_count = opens.count()

closes = rrule.rruleset(cache=True)
closes.rrule(market_closes_with_holidays)
for holiday_rule in holiday_closes:
    closes.exrule(holiday_rule)
close_count = closes.count()

