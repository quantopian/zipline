#
# Copyright 2016 Quantopian, Inc.
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

import pandas as pd
from pandas.tseries.tools import normalize_date

from zipline.finance.performance.period import PerformancePeriod
from zipline.errors import NoFurtherDataError
import zipline.finance.risk as risk

from . position_tracker import PositionTracker

log = logbook.Logger('Performance')


class PerformanceTracker(object):
    """
    Tracks the performance of the algorithm.
    """
    def __init__(self, sim_params, trading_calendar, env):
        self.sim_params = sim_params
        self.trading_calendar = trading_calendar
        self.asset_finder = env.asset_finder
        self.treasury_curves = env.treasury_curves

        self.period_start = self.sim_params.start_session
        self.period_end = self.sim_params.end_session
        self.last_close = self.sim_params.last_close
        self._current_session = self.sim_params.start_session

        self.market_open, self.market_close = \
            self.trading_calendar.open_and_close_for_session(
                self._current_session
            )

        self.total_session_count = len(self.sim_params.sessions)
        self.capital_base = self.sim_params.capital_base
        self.emission_rate = sim_params.emission_rate

        self.position_tracker = PositionTracker(
            asset_finder=env.asset_finder,
            data_frequency=self.sim_params.data_frequency
        )

        if self.emission_rate == 'daily':
            self.all_benchmark_returns = pd.Series(
                index=self.sim_params.sessions
            )
            self.cumulative_risk_metrics = \
                risk.RiskMetricsCumulative(
                    self.sim_params,
                    self.treasury_curves,
                    self.trading_calendar
                )
        elif self.emission_rate == 'minute':
            self.all_benchmark_returns = pd.Series(index=pd.date_range(
                self.sim_params.first_open, self.sim_params.last_close,
                freq='Min')
            )

            self.cumulative_risk_metrics = \
                risk.RiskMetricsCumulative(
                    self.sim_params,
                    self.treasury_curves,
                    self.trading_calendar,
                    create_first_day_stats=True
                )

        # this performance period will span the entire simulation from
        # inception.
        self.cumulative_performance = PerformancePeriod(
            # initial cash is your capital base.
            starting_cash=self.capital_base,
            data_frequency=self.sim_params.data_frequency,
            # the cumulative period will be calculated over the entire test.
            period_open=self.period_start,
            period_close=self.period_end,
            # don't save the transactions for the cumulative
            # period
            keep_transactions=False,
            keep_orders=False,
            # don't serialize positions for cumulative period
            serialize_positions=False,
            asset_finder=self.asset_finder,
            name="Cumulative"
        )
        self.cumulative_performance.position_tracker = self.position_tracker

        # this performance period will span just the current market day
        self.todays_performance = PerformancePeriod(
            # initial cash is your capital base.
            starting_cash=self.capital_base,
            data_frequency=self.sim_params.data_frequency,
            # the daily period will be calculated for the market day
            period_open=self.market_open,
            period_close=self.market_close,
            keep_transactions=True,
            keep_orders=True,
            serialize_positions=True,
            asset_finder=self.asset_finder,
            name="Daily"
        )
        self.todays_performance.position_tracker = self.position_tracker

        self.saved_dt = self.period_start
        # one indexed so that we reach 100%
        self.session_count = 0.0
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
            return self.session_count / self.total_session_count

    def set_date(self, date):
        if self.emission_rate == 'minute':
            self.saved_dt = date
            self.todays_performance.period_close = self.saved_dt

    def get_portfolio(self, performance_needs_update):
        if performance_needs_update:
            self.update_performance()
            self.account_needs_update = True
        return self.cumulative_performance.as_portfolio()

    def update_performance(self):
        # calculate performance as of last trade
        self.cumulative_performance.calculate_performance()
        self.todays_performance.calculate_performance()

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

    def prepare_capital_change(self, is_interday):
        self.cumulative_performance.initialize_subperiod_divider()

        if not is_interday:
            # Change comes in the middle of day
            self.todays_performance.initialize_subperiod_divider()

    def process_capital_change(self, capital_change_amount, is_interday):
        self.cumulative_performance.set_current_subperiod_starting_values(
            capital_change_amount)

        if is_interday:
            # Change comes between days
            self.todays_performance.adjust_period_starting_capital(
                capital_change_amount)
        else:
            # Change comes in the middle of day
            self.todays_performance.set_current_subperiod_starting_values(
                capital_change_amount)

    def process_transaction(self, transaction):
        self.txn_count += 1
        self.cumulative_performance.handle_execution(transaction)
        self.todays_performance.handle_execution(transaction)
        self.position_tracker.execute_transaction(transaction)

    def handle_splits(self, splits):
        leftover_cash = self.position_tracker.handle_splits(splits)
        if leftover_cash > 0:
            self.cumulative_performance.handle_cash_payment(leftover_cash)
            self.todays_performance.handle_cash_payment(leftover_cash)

    def process_order(self, event):
        self.cumulative_performance.record_order(event)
        self.todays_performance.record_order(event)

    def process_commission(self, commission):
        sid = commission['sid']
        cost = commission['cost']

        self.position_tracker.handle_commission(sid, cost)
        self.cumulative_performance.handle_commission(cost)
        self.todays_performance.handle_commission(cost)

    def process_close_position(self, asset, dt, data_portal):
        txn = self.position_tracker.\
            maybe_create_close_position_transaction(asset, dt, data_portal)
        if txn:
            self.process_transaction(txn)

    def check_upcoming_dividends(self, next_session, adjustment_reader):
        """
        Check if we currently own any stocks with dividends whose ex_date is
        the next trading day.  Track how much we should be payed on those
        dividends' pay dates.

        Then check if we are owed cash/stock for any dividends whose pay date
        is the next trading day.  Apply all such benefits, then recalculate
        performance.
        """
        if adjustment_reader is None:
            return
        position_tracker = self.position_tracker
        held_sids = set(position_tracker.positions)
        # Dividends whose ex_date is the next trading day.  We need to check if
        # we own any of these stocks so we know to pay them out when the pay
        # date comes.

        if held_sids:
            cash_dividends = adjustment_reader.get_dividends_with_ex_date(
                held_sids,
                next_session,
                self.asset_finder
            )
            stock_dividends = adjustment_reader.\
                get_stock_dividends_with_ex_date(
                    held_sids,
                    next_session,
                    self.asset_finder
                )

            position_tracker.earn_dividends(
                cash_dividends,
                stock_dividends
            )

        net_cash_payment = position_tracker.pay_dividends(next_session)
        if not net_cash_payment:
            return

        self.cumulative_performance.handle_dividends_paid(net_cash_payment)
        self.todays_performance.handle_dividends_paid(net_cash_payment)

    def handle_minute_close(self, dt, data_portal):
        """
        Handles the close of the given minute in minute emission.

        Parameters
        __________
        dt : Timestamp
            The minute that is ending

        Returns
        _______
        A minute perf packet.
        """
        self.position_tracker.sync_last_sale_prices(dt, False, data_portal)
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
        return minute_packet

    def handle_market_close(self, dt, data_portal):
        """
        Handles the close of the given day, in both minute and daily emission.
        In daily emission, also updates performance, benchmark and risk metrics
        as it would in handle_minute_close if it were minute emission.

        Parameters
        __________
        dt : Timestamp
            The minute that is ending

        Returns
        _______
        A daily perf packet.
        """
        completed_session = self._current_session

        if self.emission_rate == 'daily':
            # this method is called for both minutely and daily emissions, but
            # this chunk of code here only applies for daily emissions. (since
            # it's done every minute, elsewhere, for minutely emission).
            self.position_tracker.sync_last_sale_prices(dt, False, data_portal)
            self.update_performance()
            account = self.get_account(False)

            benchmark_value = self.all_benchmark_returns[completed_session]

            self.cumulative_risk_metrics.update(
                completed_session,
                self.todays_performance.returns,
                benchmark_value,
                account.leverage)

        # increment the day counter before we move markers forward.
        self.session_count += 1.0

        # Get the next trading day and, if it is past the bounds of this
        # simulation, return the daily perf packet
        try:
            next_session = self.trading_calendar.next_session_label(
                completed_session
            )
        except NoFurtherDataError:
            next_session = None

        # Take a snapshot of our current performance to return to the
        # browser.
        daily_update = self.to_dict(emission_type='daily')

        # On the last day of the test, don't create tomorrow's performance
        # period.  We may not be able to find the next trading day if we're at
        # the end of our historical data
        if self.market_close >= self.last_close:
            return daily_update

        # If the next trading day is irrelevant, then return the daily packet
        if (next_session is None) or (next_session >= self.last_close):
            return daily_update

        # move the market day markers forward
        # TODO Is this redundant with next_trading_day above?
        self._current_session = next_session
        self.market_open, self.market_close = \
            self.trading_calendar.open_and_close_for_session(
                self._current_session
            )

        # Roll over positions to current day.
        self.todays_performance.rollover()
        self.todays_performance.period_open = self.market_open
        self.todays_performance.period_close = self.market_close

        # Check for any dividends, then return the daily perf packet
        self.check_upcoming_dividends(
            next_session=next_session,
            adjustment_reader=data_portal._adjustment_reader
        )

        return daily_update

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """

        log_msg = "Simulated {n} trading days out of {m}."
        log.info(log_msg.format(n=int(self.session_count),
                                m=self.total_session_count))
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

        risk_report = risk.RiskReport(
            ars,
            self.sim_params,
            benchmark_returns=bms,
            algorithm_leverages=acl,
            trading_calendar=self.trading_calendar,
            treasury_curves=self.treasury_curves,
        )

        return risk_report.to_dict()
