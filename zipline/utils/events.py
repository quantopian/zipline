
import pytz
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
