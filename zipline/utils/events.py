
import pytz
import datetime
import pandas as pd
from pandas.tseries.frequencies import to_offset
from zipline.utils import tradingcalendar


class EventContainer(object):

    def __init__(self, *events):
        self.events = pd.Series(events)

    def add_event(self, freq='B', functions=[],
                  time_func=lambda dt: True,
                  start_dt='1990-01-01', tz='US/Eastern'):
        event = EventOffset(freq=freq,
                            start_dt=pd.Timestamp(start_dt, tz=tz),
                            time_func=time_func,
                            functions=functions)

        self.events[len(self.events.index)] = event

    def handle_data(self, context, data, dt):
        f = lambda event: event.handle_data(context, data, dt)
        self.events.apply(f)


class EventOffset(object):

    def __init__(self, freq='B', functions=[],
                 time_func=lambda dt: True,
                 start_dt=tradingcalendar.start):

        self.offset = to_offset(freq)
        self.time_func = time_func
        self.start_dt = pd.Timestamp(start_dt, tz=pytz.utc)
        self.event_dt = self.start_dt
        self._funcs = pd.Series(functions)

    def handle_data(self, context, data, dt):
        dt = dt.astimezone(pytz.utc)
        if dt < self.event_dt:
            return
        if self.time_func(dt):
            self._funcs.apply(lambda f: f(context, data))
            self.roll(dt)

    def roll(self, dt):
        self.event_dt = pd.Timestamp(self.offset.apply(dt), tz=pytz.utc)

    def next_dt(self, dt=None):
        if dt is None:
            dt = self.event_dt
        return pd.Timestamp(self.offset.apply(dt), tz=pytz.utc)

    def prev_dt(self, dt=None):
        if dt is None:
            dt = self.event_dt
        return pd.Timestamp(dt - self.offset, tz=pytz.utc)

    def add_func(self, func):
        self._funcs[len(self._funcs.index)] = func


#
# Common market timing utilities.
#


class AfterOpen(object):
    """
    Returns True once the market has been open for N minutes.
    """
    def __init__(self, minutes=0, hours=0):
        self.n = 60*hours + minutes

    def __call__(self, dt, exact=False):
        n = minutes_since_open(dt)
        if exact:
            return n == self.n
        return n >= self.n


class BeforeClose(object):
    """
    A rule to enter after the market has been open for N minutes.
    """
    def __init__(self, minutes=0, hours=0):
        self.n = 60*hours + minutes

    def __call__(self, dt, exact=False):
        n = minutes_until_close(dt)
        if exact:
            return n == self.n
        return n >= self.n


class AtTime(object):
    """
    Rule to enter at a specific time only.
    """
    def __init__(self, hour=None, minute=None, tz='US/Eastern'):
        self.tz = pytz.timezone(tz)
        self.time = datetime.time(hour, minute, tzinfo=self.tz)

    def __call__(self, dt):
        dt = self.tz.normalize(dt)
        return dt.timetz() == self.time


class BetweenTimes(object):

    def __init__(self, time1=None, time2=None, tz='US/Eastern'):
        """
        :params:
            time1 and time2: tuples
            lower and upper bounds of the entry times.
            tz: timezone obj or string id
        e.g.
        BetweenTimes((9, 31), (10, 0)) evaluates
        to True from 9:31 to 9:59 Eastern time.
        """
        self.tz = pytz.timezone(tz)
        self.t1 = datetime.time(*time1, tzinfo=self.tz)
        self.t2 = datetime.time(*time2, tzinfo=self.tz)

    def __call__(self, dt):
        dt = self.tz.normalize(dt).timetz()
        return self.t1 <= dt < self.t2


def minutes_until_close(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    open_close = tradingcalendar.open_and_closes.T[ref]
    market_close = open_close['market_close']
    return (market_close - dt).seconds / 60


def minutes_since_open(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    open_close = tradingcalendar.open_and_closes.T[ref]
    market_open = open_close['market_open']
    return (dt - market_open).seconds / 60

#
# User facing functions for consistency
#


def market_open(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    return dt == tradingcalendar.open_and_closes.T[ref]['market_open']


def market_close(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    return dt == tradingcalendar.open_and_closes.T[ref]['market_close']


def at_time(hour, minute, tz='US/Eastern'):
    return AtTime(hour=hour, minute=minute, tz=tz)


def between_times(time1=None, time2=None, tz='US/Eastern'):
    return BetweenTimes(time1=time1, time2=time2, tz=tz)


def after_open(minutes=0, hours=0):
    return AfterOpen(minutes=minutes, hours=hours)


def before_close(minutes=0, hours=0):
    return BeforeClose(minutes=minutes, hours=hours)
