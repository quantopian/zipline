
import pytz
import datetime

from pandas.tseries.frequencies import to_offset

from zipline.utils import tradingcalendar
from zipline.finance.trading import TradingEnvironment


ENV = TradingEnvironment.instance()


class EventContainer(object):
    """
    Container class to keep track of events and call
    their associated functions.
    """

    events = []

    def add_event(self, rule=None, func=None,
                  freq=None, start_dt=tradingcalendar.start):

        event = EventOffset(
            rule=rule,
            func=func,
            freq=freq,
            start_dt=start_dt)
        self.events.append(event)

    def handle_data(self, context, data, dt):
        for event in self.events:
            event.handle_data(context, data, dt)


class EventOffset(object):
    """
    Facilitates scheduling function calls at frequencies
    other than the backtest frequency.
    """

    def __init__(self, rule=None, func=None,
                 freq=None, start_dt=tradingcalendar.start):
        """
        Params:
            rule: callable <defalult=None>
                  A callable that must accept a datetime arg
                  and return a boolean value. This is only called
                  if the offset has been satisfied.
                  The freq arg is obeyed in the case where the rule is None.

            func: callable or array of callables. <default=None>
                  function(s) to be called with (context, data) arguments
                  when the offset and rule are satisfied.

            freq: An object that can be converted to a pandas offset
                  via pandas.tseries.frequencies.to_offset

            start_dt: tz-aware datetime obj
                  The first datetime the event can occur.
        """
        if rule is None:
            self._rule = lambda dt: True
        else:
            self._rule = rule
        if hasattr(func, '__iter__'):
            self._funcs = list(func)
        else:
            self._funcs = [func]
        self.offset = to_offset(freq)
        self.start_dt = start_dt
        self.event_dt = self.start_dt

    def handle_data(self, context, data, dt):
        if dt < self.event_dt:
            return
        if self._rule(dt):
            for func in self._funcs:
                func(context, data)
            self.event_dt = self.offset.apply(dt)


#
# Common market timing utilities.
#


class AfterOpen(object):
    """
    Returns True once the market has been open for N minutes.
    """
    def __init__(self, minutes=0, hours=0):
        self.n = 60*hours + minutes

    def __call__(self, dt):
        n = minutes_since_open(dt)
        return n >= self.n


class BeforeClose(object):
    """
    Returns True for the last N minutes of the trading day.
    """
    def __init__(self, minutes=0, hours=0):
        self.n = 60*hours + minutes

    def __call__(self, dt):
        n = minutes_until_close(dt)
        return n < self.n


class AtTime(object):
    """
    Returns True at a specific time only.
    """
    def __init__(self, hour=None, minute=None, tz='US/Eastern'):
        self.tz = pytz.timezone(tz)
        self.time = datetime.time(hour, minute, tzinfo=self.tz)

    def __call__(self, dt):
        dt = self.tz.normalize(dt)
        return dt.timetz() == self.time


class BetweenTimes(object):
    """
    Returns True in the interval [time1, time2)
    """

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
    """
    returns the number of minutes remaining
    in the trading day.
    """
    _, close = ENV.get_open_and_close(dt)
    return (close - dt).seconds / 60


def minutes_since_open(dt):
    """
    returns the number of minutes since
    the market open.
    """
    open_dt, _ = ENV.get_open_and_close(dt)
    return (dt - open_dt).seconds / 60


def market_open(dt):
    """
    returns True if dt is equal to
    the market open on that day.
    """
    open_dt, _ = ENV.get_open_and_close(dt)
    return open_dt == dt


def market_close(dt):
    """
    returns True if dt is equal to
    the market close on that day.
    """
    _, close = ENV.get_open_and_close(dt)
    return close == dt
