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

from collections import defaultdict

import zipline.protocol as zp
from zipline.finance.slippage import (
    VolumeShareSlippage,
    transact_partial
)
from zipline.finance.commission import PerShare

log = logbook.Logger('Transaction Simulator')


class TransactionSimulator(object):

    def __init__(self):
        self.transact = transact_partial(VolumeShareSlippage(), PerShare())
        self.open_orders = defaultdict(list)

    def place_order(self, order):
        # initialized filled field.
        order.filled = 0
        self.open_orders[order.sid].append(order)

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        for event in stream_in:
            yield self.update(event)

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

        self.trading_days = []
        self.trading_day_map = {}
        self.treasury_curves = treasury_curves
        self.benchmark_returns = benchmark_returns
        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base
        self.period_trading_days = None

        assert self.period_start <= self.period_end, \
            "Period start falls after period end."

        for bm in benchmark_returns:
            self.trading_days.append(bm.date)
            self.trading_day_map[bm.date] = bm

        assert self.period_start <= self.trading_days[-1], \
            "Period start falls after the last known trading day."
        assert self.period_end >= self.trading_days[0], \
            "Period end falls before the first known trading day."

        self.first_open = self.calculate_first_open()
        self.last_close = self.calculate_last_close()

        self.prior_day_open = self.calculate_prior_day_open()

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

        if first_open <= self.trading_days[0]:
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
    def days_in_period(self):
        """return the number of trading days within the period [start, end)"""
        assert self.period_start is not None
        assert self.period_end is not None

        if self.period_trading_days is None:
            self.period_trading_days = []
            for date in self.trading_days:
                if date > self.period_end:
                    break
                if date >= self.period_start:
                    self.period_trading_days.append(date)

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

    def get_benchmark_daily_return(self, test_date):
        date = self.normalize_date(test_date)
        if date in self.trading_day_map:
            return self.trading_day_map[date].returns
        else:
            return 0.0
