#
# Copyright 2012 Quantopian, Inc.
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

import pytz
import logbook
import datetime

from collections import defaultdict, OrderedDict
import bisect

import zipline.protocol as zp
from zipline.finance.slippage import (
    VolumeShareSlippage,
    transact_partial
)
from zipline.finance.commission import PerShare

from zipline.finance.orders import (
    Order,
    MarketOrder,
    LimitOrder,
    StopOrder,
    StopLimitOrder
)

log = logbook.Logger('Transaction Simulator')


class TransactionSimulator(object):

    def __init__(self):
        self.transact = transact_partial(VolumeShareSlippage(), PerShare())
        self.open_orders = defaultdict(list)

    def place_order(self, order):
        # initialized filled field.
        order.filled = 0
        self.open_orders[order.sid].append(order)

    def place_order_v2(self, dt, sid, amount, *args):
        
        # parse extra args to determine order type

        # something could be done with amount to further divide
        # between buy by share count OR buy shares up to a dollar amount
        # int == share count  AND  float == dollar amount
        
        # print ("order_v2:")
        # print ("dt: ", dt)
        # print ("sid: ", sid)
        # print ("amount: ", amount)
        # print ("len(args)", len(args))
        # i = 0
        # for arg in args:
        #     print( "*args{p}: ".format(p=i), arg)
        #     i += 1

        o_info = args[0]

        if len(o_info) == 0 or ( len(o_info) == 1 and o_info[0] == "market"):
            order = MarketOrder({
                'dt': dt,
                'sid': sid,
                'amount': int(amount),
                'filled': 0
            })
        elif len(o_info) == 2 and o_info[0] == "limit":
            # TODO validate o_info[1] "limit_price"
            limit = o_info[1]
            order = LimitOrder({
                'dt': dt,
                'sid': sid,
                'amount': int(amount),
                'filled': 0,
                'limit': limit
            })
        elif len(o_info) == 2 and o_info[0] == "stop":
            # TODO validate o_info[1] "stop_price"
            stop = o_info[1]
            order = StopOrder({
                'dt': dt,
                'sid': sid,
                'amount': int(amount),
                'filled': 0,
                'stop': stop
            })
        elif len(o_info) == 3 and o_info[0] == "stoplimit":
            # TODO validate o_info[1,2] 
            stop = o_info[1]
            limit = o_info[2]
            order = StopLimitOrder({
                'dt': dt,
                'sid': sid,
                'amount': int(amount),
                'filled': 0,
                'stop': stop,
                'limit': limit 
            })
        else:
            return "error with args to {place_}order_v2"
            #some error happened
            # decipher and return error string

        self.open_orders[order.sid].append(order)
        # return ""

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        for date, snapshot in stream_in:
            yield date, [self.update(event) for event in snapshot]

    def update(self, event):
        event.TRANSACTION = None
        # We only fill transactions on trade events.
        if event.type == zp.DATASOURCE_TYPE.TRADE:
            event.TRANSACTION = self.transact(event, self.open_orders)
        return event


class TradingEnvironment(object):

    def __init__(
        self,
        benchmark_returns,
        treasury_curves,
        period_start=None,
        period_end=None,
        capital_base=None
    ):

        self.trading_day_map = OrderedDict()
        self.treasury_curves = treasury_curves
        self.benchmark_returns = benchmark_returns
        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base
        self._period_trading_days = None

        assert self.period_start <= self.period_end, \
            "Period start falls after period end."

        for bm in benchmark_returns:
            self.trading_day_map[bm.date] = bm

        self.first_trading_day = next(self.trading_day_map.iterkeys())
        self.last_trading_day = next(reversed(self.trading_day_map))

        assert self.period_start <= self.last_trading_day, \
            "Period start falls after the last known trading day."
        assert self.period_end >= self.first_trading_day, \
            "Period end falls before the first known trading day."

        self.first_open = self.calculate_first_open()
        self.last_close = self.calculate_last_close()

        self.prior_day_open = self.calculate_prior_day_open()

    def __repr__(self):
        return "%s(%r)" % (
            self.__class__.__name__,
            {'first_open': self.first_open,
             'last_close': self.last_close
             })

    def calculate_first_open(self):
        """
        Finds the first trading day on or after self.period_start.
        """
        first_open = self.period_start
        one_day = datetime.timedelta(days=1)

        while not self.is_trading_day(first_open):
            first_open = first_open + one_day

        first_open = self.set_NYSE_time(first_open, 9, 30)
        return first_open

    def calculate_prior_day_open(self):
        """
        Finds the first trading day open that falls at least a day
        before period_start.
        """
        one_day = datetime.timedelta(days=1)
        first_open = self.period_start - one_day

        if first_open <= self.first_trading_day:
            log.warn("Cannot calculate prior day open.")
            return self.period_start

        while not self.is_trading_day(first_open):
            first_open = first_open - one_day

        first_open = self.set_NYSE_time(first_open, 9, 30)
        return first_open

    def calculate_last_close(self):
        """
        Finds the last trading day on or before self.period_end
        """
        last_close = self.period_end
        one_day = datetime.timedelta(days=1)

        while not self.is_trading_day(last_close):
            last_close = last_close - one_day

        last_close = self.set_NYSE_time(last_close, 16, 00)

        return last_close

    #TODO: add other exchanges and timezones...
    def set_NYSE_time(self, dt, hour, minute):
        naive = datetime.datetime(
            year=dt.year,
            month=dt.month,
            day=dt.day
        )
        local = pytz.timezone('US/Eastern')
        local_dt = naive.replace(tzinfo=local)
        # set the clock to the opening bell in NYC time.
        local_dt = local_dt.replace(hour=hour, minute=minute)
        # convert to UTC
        utc_dt = local_dt.astimezone(pytz.utc)
        return utc_dt

    def normalize_date(self, test_date):
        return datetime.datetime(
            year=test_date.year,
            month=test_date.month,
            day=test_date.day,
            tzinfo=pytz.utc
        )

    @property
    def period_trading_days(self):
        if self._period_trading_days is None:
            self._period_trading_days = []
            for date in self.trading_day_map.iterkeys():
                if date > self.period_end:
                    break
                if date >= self.period_start:
                    self.period_trading_days.append(date)
        return self._period_trading_days

    @property
    def days_in_period(self):
        """return the number of trading days within the period [start, end)"""
        return len(self.period_trading_days)

    def is_market_hours(self, test_date):
        if not self.is_trading_day(test_date):
            return False

        mkt_open = self.set_NYSE_time(test_date, 9, 30)
        #TODO: half days?
        mkt_close = self.set_NYSE_time(test_date, 16, 00)

        return test_date >= mkt_open and test_date <= mkt_close

    def is_trading_day(self, test_date):
        dt = self.normalize_date(test_date)
        return (dt in self.trading_day_map)

    def next_trading_day(self, test_date):
        dt = self.normalize_date(test_date)
        delta = datetime.timedelta(days=1)

        while dt <= self.last_trading_day:
            dt += delta
            if dt in self.trading_day_map:
                return dt

        return None

    def trading_day_distance(self, first_date, second_date):
        first_date = self.normalize_date(first_date)
        second_date = self.normalize_date(second_date)

        trading_days = self.trading_day_map.keys()
        # Find leftmost item greater than or equal to day
        i = bisect.bisect_left(trading_days, first_date)
        if i == len(trading_days):  # nothing found
            return None
        j = bisect.bisect_left(trading_days, second_date)
        if j == len(trading_days):
            return None

        return j - i

    def get_benchmark_daily_return(self, test_date):
        date = self.normalize_date(test_date)
        if date in self.trading_day_map:
            return self.trading_day_map[date].returns
        else:
            return 0.0
