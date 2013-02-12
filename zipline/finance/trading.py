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

import bisect
import pytz
import logbook
import datetime

from collections import defaultdict, OrderedDict
from delorean import Delorean
from pandas import DatetimeIndex

import zipline.protocol as zp
from zipline.finance.slippage import (
    VolumeShareSlippage,
    transact_partial
)

from zipline.finance.commission import PerShare

log = logbook.Logger('Transaction Simulator')

environment = None


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
        load=None,
        bm_symbol='^GSPC',
        exchange_tz="US/Eastern"
    ):

        self.trading_day_map = OrderedDict()
        self.bm_symbol = bm_symbol
        if not load:
            from zipline.data.loader import load_market_data
            load = load_market_data

        self.benchmark_returns, self.treasury_curves = \
            load(self.bm_symbol)

        self._period_trading_days = None
        self._trading_days_series = None
        self.full_trading_day = datetime.timedelta(hours=6, minutes=30)
        self.exchange_tz = exchange_tz

        for bm in self.benchmark_returns:
            self.trading_day_map[bm.date] = bm

        self.first_trading_day = next(self.trading_day_map.iterkeys())
        self.last_trading_day = next(reversed(self.trading_day_map))

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
    def trading_days(self):
        if self._trading_days_series is None:
            self._trading_days_series = \
                DatetimeIndex(self.trading_day_map.iterkeys())
        return self._trading_days_series

    def is_market_hours(self, test_date):
        if not self.is_trading_day(test_date):
            return False

        mkt_open, mkt_close = self.get_open_and_close(test_date)
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

    def next_open_and_close(self, start_date):
        """
        Given the start_date, returns the next open and close of
        the market.
        """
        next_open = self.next_trading_day(start_date)

        if next_open is None:
            raise Exception(
                "Attempt to backtest beyond available history. \
Last successful date: %s" % self.market_open)

        return self.get_open_and_close(next_open)

    def get_open_and_close(self, next_open):

        # creating a naive datetime with the correct hour,
        # minute, and date. this will allow us to use Delorean to
        # shift the time between EST and UTC.
        next_open = next_open.replace(
            hour=9,
            minute=30,
            second=0,
            tzinfo=None
        )
        # create a new Delorean with the next_open naive date and
        # the correct timezone for the exchange.
        open_delorean = Delorean(next_open, "US/Eastern")
        open_utc = open_delorean.shift("UTC").datetime

        market_open = open_utc
        market_close = market_open + self.get_trading_day_duration(open_utc)

        return market_open, market_close

    def get_trading_day_duration(self, trading_day):
        # TODO: make a list of half-days and modify the
        # calculation of market close to reflect them.
        return self.full_trading_day

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

    def get_index(self, dt):
        ndt = self.normalize_date(dt)
        return self.trading_days.searchsorted(ndt)


class SimulationParameters(object):
    def __init__(self, period_start, period_end,
                 capital_base=10e3):

        # raise and exception if the global environment is not
        # set.
        global environment
        if not environment:
            environment = TradingEnvironment()

        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base
        self.first_open = self.calculate_first_open()
        self.last_close = self.calculate_last_close()
        start_index = \
            environment.get_index(self.first_open)
        end_index = environment.get_index(self.last_close)

        # take an inclusive slice of the environment's
        # trading_days.
        self.trading_days = \
            environment.trading_days[start_index:end_index+1]

        self.prior_day_open = self.calculate_prior_day_open()

        assert self.period_start <= self.period_end, \
            "Period start falls after period end."

        assert self.period_start <= environment.last_trading_day, \
            "Period start falls after the last known trading day."
        assert self.period_end >= environment.first_trading_day, \
            "Period end falls before the first known trading day."

    def calculate_first_open(self):
        """
        Finds the first trading day on or after self.period_start.
        """
        first_open = self.period_start
        one_day = datetime.timedelta(days=1)

        while not environment.is_trading_day(first_open):
            first_open = first_open + one_day

        mkt_open, _ = environment.get_open_and_close(first_open)
        return mkt_open

    def calculate_prior_day_open(self):
        """
        Finds the first trading day open that falls at least a day
        before period_start.
        """
        one_day = datetime.timedelta(days=1)
        first_open = self.period_start - one_day

        if first_open <= environment.first_trading_day:
            log.warn("Cannot calculate prior day open.")
            return self.period_start

        while not environment.is_trading_day(first_open):
            first_open = first_open - one_day

        mkt_open, _ = environment.get_open_and_close(first_open)
        return mkt_open

    def calculate_last_close(self):
        """
        Finds the last trading day on or before self.period_end
        """
        last_close = self.period_end
        one_day = datetime.timedelta(days=1)

        while not environment.is_trading_day(last_close):
            last_close = last_close - one_day

        _, mkt_close = environment.get_open_and_close(last_close)
        return mkt_close

    @property
    def days_in_period(self):
        """return the number of trading days within the period [start, end)"""
        return len(self.trading_days)

    def __repr__(self):
        return "%s(%r)" % (
            self.__class__.__name__,
            {'first_open': self.first_open,
             'last_close': self.last_close
             })
