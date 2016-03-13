#
# Copyright 2015 Quantopian, Inc.
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

"""

Performance Tracking
====================

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | period_start    | The beginning of the period to be tracked. datetime|
    |                 | in pytz.utc timezone. Will always be 0:00 on the   |
    |                 | date in UTC. The fact that the time may be on the  |
    |                 | prior day in the exchange's local time is ignored  |
    +-----------------+----------------------------------------------------+
    | period_end      | The end of the period to be tracked. datetime      |
    |                 | in pytz.utc timezone. Will always be 23:59 on the  |
    |                 | date in UTC. The fact that the time may be on the  |
    |                 | next day in the exchange's local time is ignored   |
    +-----------------+----------------------------------------------------+
    | progress        | percentage of test completed                       |
    +-----------------+----------------------------------------------------+
    | capital_base    | The initial capital assumed for this tracker.      |
    +-----------------+----------------------------------------------------+
    | cumulative_perf | A dictionary representing the cumulative           |
    |                 | performance through all the events delivered to    |
    |                 | this tracker. For details see the comments on      |
    |                 | :py:meth:`PerformancePeriod.to_dict`               |
    +-----------------+----------------------------------------------------+
    | todays_perf     | A dictionary representing the cumulative           |
    |                 | performance through all the events delivered to    |
    |                 | this tracker with datetime stamps between last_open|
    |                 | and last_close. For details see the comments on    |
    |                 | :py:meth:`PerformancePeriod.to_dict`               |
    |                 | TODO: adding this because we calculate it. May be  |
    |                 | overkill.                                          |
    +-----------------+----------------------------------------------------+
    | cumulative_risk | A dictionary representing the risk metrics         |
    | _metrics        | calculated based on the positions aggregated       |
    |                 | through all the events delivered to this tracker.  |
    |                 | For details look at the comments for               |
    |                 | :py:meth:`zipline.finance.risk.RiskMetrics.to_dict`|
    +-----------------+----------------------------------------------------+

"""

from __future__ import division
import logbook
import pickle
from six import iteritems
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.tools import normalize_date

import zipline.finance.risk as risk
from . period import PerformancePeriod

from zipline.utils.pandas_utils import sort_values
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)
from . position_tracker import PositionTracker

log = logbook.Logger('Performance')


class PerformanceTracker(object):
    """
    Tracks the performance of the algorithm.
    """
    def __init__(self, sim_params, env):

        self.sim_params = sim_params
        self.env = env

        self.period_start = self.sim_params.period_start
        self.period_end = self.sim_params.period_end
        self.last_close = self.sim_params.last_close
        first_open = self.sim_params.first_open.tz_convert(
            self.env.exchange_tz
        )
        self.day = pd.Timestamp(datetime(first_open.year, first_open.month,
                                         first_open.day), tz='UTC')
        self.market_open, self.market_close = env.get_open_and_close(self.day)
        self.total_days = self.sim_params.days_in_period
        self.capital_base = self.sim_params.capital_base
        self.emission_rate = sim_params.emission_rate

        all_trading_days = env.trading_days
        mask = ((all_trading_days >= normalize_date(self.period_start)) &
                (all_trading_days <= normalize_date(self.period_end)))

        self.trading_days = all_trading_days[mask]

        self.dividend_frame = pd.DataFrame()
        self._dividend_count = 0

        self.position_tracker = PositionTracker(asset_finder=env.asset_finder)

        if self.emission_rate == 'daily':
            self.all_benchmark_returns = pd.Series(
                index=self.trading_days)
            self.cumulative_risk_metrics = \
                risk.RiskMetricsCumulative(self.sim_params, self.env)

        elif self.emission_rate == 'minute':
            self.all_benchmark_returns = pd.Series(index=pd.date_range(
                self.sim_params.first_open, self.sim_params.last_close,
                freq='Min'))

            self.cumulative_risk_metrics = \
                risk.RiskMetricsCumulative(self.sim_params, self.env,
                                           create_first_day_stats=True)

        # this performance period will span the entire simulation from
        # inception.
        self.cumulative_performance = PerformancePeriod(
            # initial cash is your capital base.
            starting_cash=self.capital_base,
            # the cumulative period will be calculated over the entire test.
            period_open=self.period_start,
            period_close=self.period_end,
            # don't save the transactions for the cumulative
            # period
            keep_transactions=False,
            keep_orders=False,
            # don't serialize positions for cumulative period
            serialize_positions=False,
            asset_finder=self.env.asset_finder,
        )
        self.cumulative_performance.position_tracker = self.position_tracker

        # this performance period will span just the current market day
        self.todays_performance = PerformancePeriod(
            # initial cash is your capital base.
            starting_cash=self.capital_base,
            # the daily period will be calculated for the market day
            period_open=self.market_open,
            period_close=self.market_close,
            keep_transactions=True,
            keep_orders=True,
            serialize_positions=True,
            asset_finder=self.env.asset_finder,
        )
        self.todays_performance.position_tracker = self.position_tracker

        self.saved_dt = self.period_start
        # one indexed so that we reach 100%
        self.day_count = 0.0
        self.txn_count = 0

        self.account_needs_update = True
        self._account = None

    def __repr__(self):
        return "%s(%r)" % (
            self.__class__.__name__,
            {'simulation parameters': self.sim_params})

    @property
    def progress(self):
        if self.emission_rate == 'minute':
            # Fake a value
            return 1.0
        elif self.emission_rate == 'daily':
            return self.day_count / self.total_days

    def set_date(self, date):
        if self.emission_rate == 'minute':
            self.saved_dt = date
            self.todays_performance.period_close = self.saved_dt

    def update_dividends(self, new_dividends):
        """
        Update our dividend frame with new dividends.  @new_dividends should be
        a DataFrame with columns containing at least the entries in
        zipline.protocol.DIVIDEND_FIELDS.
        """

        # Mark each new dividend with a unique integer id.  This ensures that
        # we can differentiate dividends whose date/sid fields are otherwise
        # identical.
        new_dividends['id'] = np.arange(
            self._dividend_count,
            self._dividend_count + len(new_dividends),
        )
        self._dividend_count += len(new_dividends)

        self.dividend_frame = sort_values(pd.concat(
            [self.dividend_frame, new_dividends]
        ), ['pay_date', 'ex_date']).set_index('id', drop=False)

    def initialize_dividends_from_other(self, other):
        """
        Helper for copying dividends to a new PerformanceTracker while
        preserving dividend count.  Useful if a simulation needs to create a
        new PerformanceTracker mid-stream and wants to preserve stored dividend
        info.

        Note that this does not copy unpaid dividends.
        """
        self.dividend_frame = other.dividend_frame
        self._dividend_count = other._dividend_count

    def handle_sid_removed_from_universe(self, sid):
        """
        This method handles any behaviors that must occur when a SID leaves the
        universe of the TradingAlgorithm.

        Parameters
        __________
        sid : int
            The sid of the Asset being removed from the universe.
        """

        # Drop any dividends for the sid from the dividends frame
        self.dividend_frame = self.dividend_frame[
            self.dividend_frame.sid != sid
        ]

    def update_performance(self):
        # calculate performance as of last trade
        self.cumulative_performance.calculate_performance()
        self.todays_performance.calculate_performance()

    def get_portfolio(self, performance_needs_update):
        if performance_needs_update:
            self.update_performance()
            self.account_needs_update = True
        return self.cumulative_performance.as_portfolio()

    def get_account(self, performance_needs_update):
        if performance_needs_update:
            self.update_performance()
            self.account_needs_update = True
        if self.account_needs_update:
            self._update_account()
        return self._account

    def _update_account(self):
        self._account = self.cumulative_performance.as_account()
        self.account_needs_update = False

    def to_dict(self, emission_type=None):
        """
        Creates a dictionary representing the state of this tracker.
        Returns a dict object of the form described in header comments.
        """

        # Default to the emission rate of this tracker if no type is provided
        if emission_type is None:
            emission_type = self.emission_rate

        _dict = {
            'period_start': self.period_start,
            'period_end': self.period_end,
            'capital_base': self.capital_base,
            'cumulative_perf': self.cumulative_performance.to_dict(),
            'progress': self.progress,
            'cumulative_risk_metrics': self.cumulative_risk_metrics.to_dict()
        }
        if emission_type == 'daily':
            _dict['daily_perf'] = self.todays_performance.to_dict()
        elif emission_type == 'minute':
            _dict['minute_perf'] = self.todays_performance.to_dict(
                self.saved_dt)
        else:
            raise ValueError("Invalid emission type: %s" % emission_type)

        return _dict

    def _handle_event_price(self, event):
        self.position_tracker.update_last_sale(event)

    def process_trade(self, event):
        self._handle_event_price(event)

    def process_transaction(self, event):
        self._handle_event_price(event)
        self.txn_count += 1
        self.cumulative_performance.handle_execution(event)
        self.todays_performance.handle_execution(event)
        self.position_tracker.execute_transaction(event)

    def process_dividend(self, dividend):

        log.info("Ignoring DIVIDEND event.")

    def process_split(self, event):
        leftover_cash = self.position_tracker.handle_split(event)
        if leftover_cash > 0:
            self.cumulative_performance.handle_cash_payment(leftover_cash)
            self.todays_performance.handle_cash_payment(leftover_cash)

    def process_order(self, event):
        self.cumulative_performance.record_order(event)
        self.todays_performance.record_order(event)

    def process_commission(self, commission):
        sid = commission.sid
        cost = commission.cost

        self.position_tracker.handle_commission(sid, cost)
        self.cumulative_performance.handle_commission(cost)
        self.todays_performance.handle_commission(cost)

    def process_benchmark(self, event):
        if self.sim_params.data_frequency == 'minute' and \
           self.sim_params.emission_rate == 'daily':
            # Minute data benchmarks should have a timestamp of market
            # close, so that calculations are triggered at the right time.
            # However, risk module uses midnight as the 'day'
            # marker for returns, so adjust back to midnight.
            midnight = pd.tseries.tools.normalize_date(event.dt)
        else:
            midnight = event.dt

        if midnight not in self.all_benchmark_returns.index:
            raise AssertionError(
                ("Date %s not allocated in all_benchmark_returns. "
                 "Calendar seems to mismatch with benchmark. "
                 "Benchmark container is=%s" %
                 (midnight,
                  self.all_benchmark_returns.index)))

        self.all_benchmark_returns[midnight] = event.returns

    def process_close_position(self, event):

        # CLOSE_POSITION events that contain prices that must be handled as
        # a final trade event
        if 'price' in event:
            self.process_trade(event)

        txn = self.position_tracker.\
            maybe_create_close_position_transaction(event)
        if txn:
            self.process_transaction(txn)

    def check_upcoming_dividends(self, next_trading_day):
        """
        Check if we currently own any stocks with dividends whose ex_date is
        the next trading day.  Track how much we should be payed on those
        dividends' pay dates.

        Then check if we are owed cash/stock for any dividends whose pay date
        is the next trading day.  Apply all such benefits, then recalculate
        performance.
        """
        if len(self.dividend_frame) == 0:
            # We don't currently know about any dividends for this simulation
            # period, so bail.
            return

        # Dividends whose ex_date is the next trading day.  We need to check if
        # we own any of these stocks so we know to pay them out when the pay
        # date comes.
        ex_date_mask = (self.dividend_frame['ex_date'] == next_trading_day)
        dividends_earnable = self.dividend_frame[ex_date_mask]

        # Dividends whose pay date is the next trading day.  If we held any of
        # these stocks on midnight before the ex_date, we need to pay these out
        # now.
        pay_date_mask = (self.dividend_frame['pay_date'] == next_trading_day)
        dividends_payable = self.dividend_frame[pay_date_mask]

        position_tracker = self.position_tracker
        if len(dividends_earnable):
            position_tracker.earn_dividends(dividends_earnable)

        if not len(dividends_payable):
            return

        net_cash_payment = position_tracker.pay_dividends(dividends_payable)

        self.cumulative_performance.handle_dividends_paid(net_cash_payment)
        self.todays_performance.handle_dividends_paid(net_cash_payment)

    def handle_minute_close(self, dt):
        """
        Handles the close of the given minute. This includes handling
        market-close functions if the given minute is the end of the market
        day.

        Parameters
        __________
        dt : Timestamp
            The minute that is ending

        Returns
        _______
        (dict, dict/None)
            A tuple of the minute perf packet and daily perf packet.
            If the market day has not ended, the daily perf packet is None.
        """
        self.update_performance()
        todays_date = normalize_date(dt)
        account = self.get_account(False)

        bench_returns = self.all_benchmark_returns.loc[todays_date:dt]
        # cumulative returns
        bench_since_open = (1. + bench_returns).prod() - 1

        self.cumulative_risk_metrics.update(todays_date,
                                            self.todays_performance.returns,
                                            bench_since_open,
                                            account.leverage)

        minute_packet = self.to_dict(emission_type='minute')

        # if this is the close, update dividends for the next day.
        # Return the performance tuple
        if dt == self.market_close:
            return (minute_packet, self._handle_market_close(todays_date))
        else:
            return (minute_packet, None)

    def handle_market_close_daily(self):
        """
        Function called after handle_data when running with daily emission
        rate.
        """
        self.update_performance()
        completed_date = self.day
        account = self.get_account(False)

        # update risk metrics for cumulative performance
        self.cumulative_risk_metrics.update(
            completed_date,
            self.todays_performance.returns,
            self.all_benchmark_returns[completed_date],
            account.leverage)

        return self._handle_market_close(completed_date)

    def _handle_market_close(self, completed_date):

        # increment the day counter before we move markers forward.
        self.day_count += 1.0

        # Get the next trading day and, if it is past the bounds of this
        # simulation, return the daily perf packet
        next_trading_day = self.env.next_trading_day(completed_date)

        # Take a snapshot of our current performance to return to the
        # browser.
        daily_update = self.to_dict(emission_type='daily')

        # On the last day of the test, don't create tomorrow's performance
        # period.  We may not be able to find the next trading day if we're at
        # the end of our historical data
        if self.market_close >= self.last_close:
            return daily_update

        # move the market day markers forward
        self.market_open, self.market_close = \
            self.env.next_open_and_close(self.day)
        self.day = self.env.next_trading_day(self.day)

        # Roll over positions to current day.
        self.todays_performance.rollover()
        self.todays_performance.period_open = self.market_open
        self.todays_performance.period_close = self.market_close

        # If the next trading day is irrelevant, then return the daily packet
        if (next_trading_day is None) or (next_trading_day >= self.last_close):
            return daily_update

        # Check for any dividends, then return the daily perf packet
        self.check_upcoming_dividends(next_trading_day=next_trading_day)
        return daily_update

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """

        log_msg = "Simulated {n} trading days out of {m}."
        log.info(log_msg.format(n=int(self.day_count), m=self.total_days))
        log.info("first open: {d}".format(
            d=self.sim_params.first_open))
        log.info("last close: {d}".format(
            d=self.sim_params.last_close))

        bms = pd.Series(
            index=self.cumulative_risk_metrics.cont_index,
            data=self.cumulative_risk_metrics.benchmark_returns_cont)
        ars = pd.Series(
            index=self.cumulative_risk_metrics.cont_index,
            data=self.cumulative_risk_metrics.algorithm_returns_cont)
        acl = self.cumulative_risk_metrics.algorithm_cumulative_leverages
        self.risk_report = risk.RiskReport(
            ars,
            self.sim_params,
            benchmark_returns=bms,
            algorithm_leverages=acl,
            env=self.env)

        risk_dict = self.risk_report.to_dict()
        return risk_dict

    def __getstate__(self):
        state_dict = \
            {k: v for k, v in iteritems(self.__dict__)
                if not k.startswith('_')}

        state_dict['dividend_frame'] = pickle.dumps(self.dividend_frame)

        state_dict['_dividend_count'] = self._dividend_count

        STATE_VERSION = 4
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 4
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("PerformanceTracker saved state is too old.")

        self.__dict__.update(state)

        # Handle the dividend frame specially
        self.dividend_frame = pickle.loads(state['dividend_frame'])

        # properly setup the perf periods
        p_types = ['cumulative', 'todays']
        for p_type in p_types:
            name = p_type + '_performance'
            period = getattr(self, name, None)
            if period is None:
                continue
            period._position_tracker = self.position_tracker
