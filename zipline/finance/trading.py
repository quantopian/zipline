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

import bisect
import pytz
import logbook
import datetime

from delorean import Delorean
import pandas as pd

from zipline.data.loader import load_market_data
from zipline.utils.tradingcalendar import get_early_closes


log = logbook.Logger('Trading')


# The financial simulations in zipline depend on information
# about the benchmark index and the risk free rates of return.
# The benchmark index defines the benchmark returns used in
# the calculation of performance metrics such as alpha/beta. Many
# components, including risk, performance, transforms, and
# batch_transforms, need access to a calendar of trading days and
# market hours. The TradingEnvironment maintains two time keeping
# facilities:
#   - a DatetimeIndex of trading days for calendar calculations
#   - a timezone name, which should be local to the exchange
#   hosting the benchmark index. All dates are normalized to UTC
#   for serialization and storage, and the timezone is used to
#   ensure proper rollover through daylight savings and so on.
#
# This module maintains a global variable, environment, which is
# subsequently referenced directly by zipline financial
# components. To set the environment, you can set the property on
# the module directly:
#       import zipline.finance.trading as trading
#       trading.environment = TradingEnvironment()
#
# or if you want to switch the environment for a limited context
# you can use a TradingEnvironment in a with clause:
#       lse = TradingEnvironment(bm_index="^FTSE", exchange_tz="Europe/London")
#       with lse:
#           # the code here will have lse as the global trading.environment
#           algo.run(start, end)
#
# User code will not normally need to use TradingEnvironment
# directly. If you are extending zipline's core financial
# compponents and need to use the environment, you must import the module
# NOT the variable. If you import the module, you will get a
# reference to the environment at import time, which will prevent
# your code from responding to user code that changes the global
# state.

environment = None


class TradingEnvironment(object):

    def __init__(
        self,
        load=None,
        bm_symbol='^GSPC',
        exchange_tz="US/Eastern",
        max_date=None,
        extra_dates=None
    ):
        self.prev_environment = self
        self.bm_symbol = bm_symbol
        if not load:
            load = load_market_data

        self.benchmark_returns, treasury_curves_map = \
            load(self.bm_symbol)

        self.treasury_curves = pd.Series(treasury_curves_map)
        if max_date:
            self.treasury_curves = self.treasury_curves[:max_date]

        self.full_trading_day = datetime.timedelta(hours=6, minutes=30)
        self.early_close_trading_day = datetime.timedelta(hours=3, minutes=30)
        self.exchange_tz = exchange_tz

        bm = None

        trading_days_list = []
        for bm in self.benchmark_returns:
            if max_date and bm.date > max_date:
                break
            trading_days_list.append(bm.date)

        self.trading_days = pd.DatetimeIndex(trading_days_list)

        if bm and extra_dates:
            for extra_date in extra_dates:
                extra_date = extra_date.replace(hour=0, minute=0, second=0,
                                                microsecond=0)
                if extra_date not in self.trading_days:
                    self.trading_days = self.trading_days + \
                        pd.DatetimeIndex([extra_date])

        self.first_trading_day = self.trading_days[0]
        self.last_trading_day = self.trading_days[-1]

        self.early_closes = get_early_closes(self.first_trading_day,
                                             self.last_trading_day)

    def __enter__(self, *args, **kwargs):
        global environment
        self.prev_environment = environment
        environment = self
        # return value here is associated with "as such_and_such" on the
        # with clause.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global environment
        environment = self.prev_environment
        # signal that any exceptions need to be propagated up the
        # stack.
        return False

    def normalize_date(self, test_date):
        return datetime.datetime(
            year=test_date.year,
            month=test_date.month,
            day=test_date.day,
            tzinfo=pytz.utc
        )

    def utc_dt_in_exchange(self, dt):
        delorean = Delorean(dt, pytz.utc.zone)
        return delorean.shift(self.exchange_tz).datetime

    def exchange_dt_in_utc(self, dt):
        delorean = Delorean(dt, self.exchange_tz)
        return delorean.shift(pytz.utc.zone).datetime

    def is_market_hours(self, test_date):
        if not self.is_trading_day(test_date):
            return False

        mkt_open, mkt_close = self.get_open_and_close(test_date)
        return test_date >= mkt_open and test_date <= mkt_close

    def is_trading_day(self, test_date):
        dt = self.normalize_date(test_date)
        return (dt in self.trading_days)

    def next_trading_day(self, test_date):
        dt = self.normalize_date(test_date)
        delta = datetime.timedelta(days=1)

        while dt <= self.last_trading_day:
            dt += delta
            if dt in self.trading_days:
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
Last successful date: %s" % self.last_trading_day)

        return self.get_open_and_close(next_open)

    def get_open_and_close(self, next_open):

        # creating a naive datetime with the correct hour,
        # minute, and date. this will allow us to use Delorean to
        # shift the time between EST and UTC.
        next_open = next_open.replace(
            hour=9,
            minute=31,
            second=0,
            microsecond=0,
            tzinfo=None
        )
        # create a new Delorean with the next_open naive date and
        # the correct timezone for the exchange.
        open_utc = self.exchange_dt_in_utc(next_open)

        market_open = open_utc
        market_close = (market_open
                        + self.get_trading_day_duration(open_utc)
                        - datetime.timedelta(minutes=1))

        return market_open, market_close

    def get_trading_day_duration(self, trading_day):
        trading_day = self.normalize_date(trading_day)
        if trading_day in self.early_closes:
            return self.early_close_trading_day

        return self.full_trading_day

    def trading_day_distance(self, first_date, second_date):
        first_date = self.normalize_date(first_date)
        second_date = self.normalize_date(second_date)

        # TODO: May be able to replace the following with searchsorted.
        # Find leftmost item greater than or equal to day
        i = bisect.bisect_left(self.trading_days, first_date)
        if i == len(self.trading_days):  # nothing found
            return None
        j = bisect.bisect_left(self.trading_days, second_date)
        if j == len(self.trading_days):
            return None

        return j - i

    def get_index(self, dt):
        ndt = self.normalize_date(dt)
        return self.trading_days.searchsorted(ndt)


class SimulationParameters(object):
    def __init__(self, period_start, period_end,
                 capital_base=10e3,
                 emission_rate='daily',
                 data_frequency='daily'):
        global environment
        if not environment:
            # This is the global environment for trading simulation.
            environment = TradingEnvironment()

        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base

        self.emission_rate = emission_rate
        self.data_frequency = data_frequency

        assert self.period_start <= self.period_end, \
            "Period start falls after period end."

        assert self.period_start <= environment.last_trading_day, \
            "Period start falls after the last known trading day."
        assert self.period_end >= environment.first_trading_day, \
            "Period end falls before the first known trading day."

        self.first_open = self.calculate_first_open()
        self.last_close = self.calculate_last_close()
        start_index = \
            environment.get_index(self.first_open)
        end_index = environment.get_index(self.last_close)

        # take an inclusive slice of the environment's
        # trading_days.
        self.trading_days = \
            environment.trading_days[start_index:end_index + 1]

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
        return """
{class_name}(
    period_start={period_start},
    period_end={period_end},
    capital_base={capital_base},
    emission_rate={emission_rate},
    first_open={first_open},
    last_close={last_close})\
""".format(class_name=self.__class__.__name__,
           period_start=self.period_start,
           period_end=self.period_end,
           capital_base=self.capital_base,
           emission_rate=self.emission_rate,
           first_open=self.first_open,
           last_close=self.last_close)
