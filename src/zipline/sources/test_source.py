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

"""A source to be used in testing."""

from datetime import timedelta
import itertools
from zipline.protocol import Event, DATASOURCE_TYPE


def create_trade(sid, price, amount, datetime, source_id="test_factory"):
    trade = Event()

    trade.source_id = source_id
    trade.type = DATASOURCE_TYPE.TRADE
    trade.sid = sid
    trade.dt = datetime
    trade.price = price
    trade.close_price = price
    trade.open_price = price
    trade.low = price * 0.95
    trade.high = price * 1.05
    trade.volume = amount

    return trade


def date_gen(start, end, trading_calendar, delta=timedelta(minutes=1), repeats=None):
    """Utility to generate a stream of dates."""

    daily_delta = not (delta.total_seconds() % timedelta(days=1).total_seconds())
    cur = start
    if daily_delta:
        # if we are producing daily timestamps, we
        # use midnight
        cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)

    def advance_current(cur):
        """Advances the current dt skipping non market days and minutes."""

        cur = cur + delta

        currently_executing = (daily_delta and (cur in trading_calendar.sessions)) or (
            trading_calendar.is_open_on_minute(cur)
        )

        if currently_executing:
            return cur
        else:
            if daily_delta:
                return trading_calendar.minute_to_session(cur).tz_localize(cur.tzinfo)
            else:
                return trading_calendar.session_open_close(
                    trading_calendar.minute_to_session(cur)
                )[0]

    # yield count trade events, all on trading days, and
    # during trading hours.
    while cur < end:
        if repeats:
            for _ in range(repeats):
                yield cur
        else:
            yield cur

        cur = advance_current(cur)


class SpecificEquityTrades:
    """Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    count  : integer representing number of trades
    sids   : list of values representing simulated internal sids
    start  : start date
    delta  : timedelta between internal events
    filter : filter to remove the sids
    """

    def __init__(
        self, trading_calendar, asset_finder, sids, start, end, delta, count=500
    ):
        self.trading_calendar = trading_calendar

        # Unpack config dictionary with default values.
        self.count = count
        self.start = start
        self.end = end
        self.delta = delta
        self.sids = sids
        self.generator = self.create_fresh_generator()

    def __iter__(self):
        return self

    def next(self):
        return self.generator.next()

    def __next__(self):
        return next(self.generator)

    def rewind(self):
        self.generator = self.create_fresh_generator()

    def update_source_id(self, gen):
        for event in gen:
            event.source_id = self.get_hash()
            yield event

    def create_fresh_generator(self):
        date_generator = date_gen(
            start=self.start,
            end=self.end,
            delta=self.delta,
            trading_calendar=self.trading_calendar,
        )
        return (
            create_trade(
                sid=sid,
                price=float(i % 10) + 1.0,
                amount=(i * 50) % 900 + 100,
                datetime=date,
            )
            for (i, date), sid in itertools.product(
                enumerate(date_generator), self.sids
            )
        )
