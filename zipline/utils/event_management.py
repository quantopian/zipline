import pytz
from datetime import time, timedelta

import pandas as pd
from zipline.finance.trading import TradingEnvironment
from zipline.utils import tradingcalendar


class EventManager(object):
    """
    Manager for periodic events.

    Example:
    This algorithm will buy 10 shares at open and sell them at the close.
    ```
    from zipline.api import order, get_datetime
    from zipline.utils.event_management import(
        EventManager, at_market_open, at_market_close
    )

    def initialize(context):
        manager = EventManager()
        manager.add_event('market_open', at_market_open)
        manager.add_event('market_close', at_market_close)
        context.manager = manager
        context.sid = 'AAPL'
        context.amount = 10

    def handle_data(self, data):
        if context.manager.market_open(get_datetime()):
            order(context.sid, context.amount)
        if context.manager.market_close(get_datetime()):
            order(context.sid, -context.amount)
    """
    def add_event(self, name, rule,
                  period=1, max_daily_hits=1, skip_early_close_days=False):
        event = PeriodicEvent(
            rule=rule,
            period=period,
            max_daily_hits=max_daily_hits,
            skip_early_close_days=skip_early_close_days
        )
        self.__dict__[name] = event

    def remove_event(self, event_name):
        del self.__dict__[event_name]



class PeriodicEvent(object):


    def __init__(self,
                 rule=None,
                 period=1,
                 max_daily_hits=1,
                 skip_early_close_days=False):
        """
        :params:
            rule: function or class instance with a __call__ method.
                Must accept a datetime obj and return a boolean value.
                This will be used as the decision function for the intraday
                entry point when all other criteria has been met.

            period: integer <default=1>
                number of trading days between events

            max_daily_hits: integer <default=1>
                upper limit on the number of times per day
                the event is triggered.

            skip_early_close_days: boolean <default=False>
                if True, the event will not occur on days
                when the market is closing early.

        """
        self.period = period
        self.max_daily_hits = max_daily_hits
        self.remaining_hits = max_daily_hits
        self.env = TradingEnvironment().instance()
        self.next_event_date = self.env.first_trading_day
        self.market_open, self.market_close = \
            self.env.get_open_and_close(self.next_event_date)
        self.skip_early_close_days = skip_early_close_days
        self._rule = rule

    def __call__(self, dt):
        """
        Algo entry point, dt is the current algo datetime.
        All arguments are passed to the rule
        """
        dt = dt.astimezone(pytz.utc)
        # The datetime passed should never be greater than
        # self.market_close, if it is, the next_event_date is set
        # to that days date. In practice, this should only be
        # triggered when signal hasn't been called yet or if the call
        # is nested inside a code block.
        #
        # This rule seems less brittle than checking for strict
        # equality with env.first_trading_day.
        if dt >= self.market_close:
            self.next_event_date = pd.Timestamp(dt.date())
            self.market_open, self.market_close = \
                self.env.get_open_and_close(self.next_event_date)
            self.remaining_hits = self.max_daily_hits
        if dt < self.market_open:
            return False
        if self.skip_early_close_days:
            # Step next event forward one day if it's an early close day.
            if self.is_early_close(dt):
                self.next_event_date = self.env.next_trading_day(dt)
                self.market_open, self.market_close = \
                    self.env.get_open_and_close(self.next_event_date)
                self.remaining_hits = self.max_daily_hits
                return False
        decision = self._rule(dt)
        if decision:
            self.remaining_hits -= 1
            if self.remaining_hits <= 0:
                self.set_next_event_date(dt)
        return decision

    def set_next_event_date(self, dt):
        self.remaining_hits = self.max_daily_hits
        idx = self.env.get_index(dt) + self.period
        self.next_event_date = self.env.trading_days[idx]
        self.market_open, self.market_close = \
            self.env.get_open_and_close(self.next_event_date)

    def is_early_close(self, dt):
        # TODO: move this to TradingEnvironment
        ref_dt = self.env.trading_days[self.env.get_index(dt)]
        return ref_dt in self.env.early_closes


class EntryRule(object):
    """
    Base class for entry rules that need
    to be configured at initialization.
    """
    pass


class AfterOpen(EntryRule):
    """
    A rule to enter after the market has been open for N minutes.
    """
    def __init__(self, minutes=0, hours=0):
        self.delta_t = timedelta(minutes=60*hours + minutes)

    def __call__(self, dt):
        ref = tradingcalendar.canonicalize_datetime(dt)
        open_close = tradingcalendar.open_and_closes.T[ref]
        market_open = open_close['market_open']
        return dt >= market_open + self.delta_t


class BeforeClose(EntryRule):
    """
    A rule to enter in the last N minutes of the trading day.
    """
    def __init__(self, minutes=0, hours=0):
        self.delta_t = 60*hours + minutes

    def __call__(self, dt):
        ref = tradingcalendar.canonicalize_datetime(dt)
        open_close = tradingcalendar.open_and_closes.T[ref]
        market_close = open_close['market_close']
        return dt > market_close - timedelta(minutes=self.delta_t)


class AtTime(EntryRule):
    """
    Rule to enter at a specific time only.
    """
    def __init__(self, hour=None, minute=None, tz='US/Eastern'):
        self.tz = pytz.timezone(tz)
        self.time = time(hour, minute, tzinfo=self.tz)

    def __call__(self, dt):
        dt = self.tz.normalize(dt)
        return dt.timetz() == self.time


class BetweenTimes(EntryRule):
    """
    Rule to enter when the current time falls between two times.
    i.e. time1 <= current time < time2
    """

    def __init__(self, time1=None, time2=None, tz='US/Eastern'):
        """
        :params:
            time1 and time2: tuples
            lower and upper bounds of the entry times.
        e.g.
        BetweenTimes((9, 31), (10, 0)) evaluates
        to True from 9:31 to 9:59 Eastern time.
        """
        self.tz = pytz.timezone(tz)
        self.t1 = time(*time1, tzinfo=self.tz)
        self.t2 = time(*time2, tzinfo=self.tz)

    def __call__(self, dt):
        dt = self.tz.normalize(dt).timetz()
        return self.t1 <= dt < self.t2


def at_market_open(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    open_close = tradingcalendar.open_and_closes.T[ref]
    return dt == open_close['market_open']


def at_market_close(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    open_close = tradingcalendar.open_and_closes.T[ref]
    return dt == open_close['market_close']
