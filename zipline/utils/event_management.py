import pytz
from zipline.utils import tradingcalendar


class EventManager(object):
    """
    Manager for periodic events.
    """

    def __init__(self,
                 period=1,
                 rule_func=None,
                 max_daily_hits=1,
                 skip_early_close_days=False,
                 calendar=tradingcalendar):
        """
        :params:
            period: integer <default=1>
                number of trading days between events

            max_daily_hits: integer <default=1>
                upper limit on the number of times per day
                the event is triggered.

            rule_func: function (returns a boolean)
                decision function for the intraday entry point

            skip_early_close_days: boolean <default=False>
                if True, the event will not occur on days
                when the market is closing early.

            calendar: zipline module <default=tradingcalendar>
                Trading calendar to use, default is NYSE.
                See zipline.utils for choices
        """
        self.period = period
        self.max_daily_hits = max_daily_hits
        self.remaining_hits = max_daily_hits
        self.next_event_date = tradingcalendar.start
        open_close = tradingcalendar.open_and_closes.iloc[0]
        self.market_open = open_close['market_open']
        self.market_close = open_close['market_close']
        self.skip_early_close_days = skip_early_close_days
        self.calendar = calendar
        self._rule_func = rule_func

    def signal(self, dt, *args, **kwargs):
        """
        Entry point for the rule_func
        All arguments are passed to rule_func
        """
        dt = dt.astimezone(pytz.utc)
        if dt < self.market_open:
            return False
        if dt >= self.market_close:
            self.set_next_event_date(dt)
        decision = self._rule_func(dt, *args, **kwargs)
        if decision:
            self.remaining_hits -= 1
            if self.remaining_hits <= 0:
                self.set_next_event_date(dt)
        return decision

    def days_index(self, dt):
        dt = self.calendar.canonicalize_datetime(dt)
        return self.calendar.trading_days.searchsorted(dt)

    def open_and_close(self, dt):
        return self.calendar.open_and_closes.T[dt]

    def set_next_event_date(self, dt):
        self.remaining_hits = self.max_daily_hits
        trading_days = self.calendar.trading_days
        idx = self.days_index(dt) + self.period
        if self.skip_early_close_days:
            while trading_days[idx] in self.calendar.open_and_closes:
                idx += 1
        self.next_event_date = trading_days[idx]
        oc_times = self.open_and_close(self.next_event_date)
        self.market_open = oc_times['market_open']
        self.market_close = oc_times['market_close']
