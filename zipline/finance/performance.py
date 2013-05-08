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

Position Tracking
=================

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | sid             | the identifier for the security held in this       |
    |                 | position.                                          |
    +-----------------+----------------------------------------------------+
    | amount          | whole number of shares in the position             |
    +-----------------+----------------------------------------------------+
    | last_sale_price | price at last sale of the security on the exchange |
    +-----------------+----------------------------------------------------+
    | cost_basis      | the volume weighted average price paid per share   |
    +-----------------+----------------------------------------------------+



Performance Period
==================

Performance Periods are updated with every trade. When calling
code needs a portfolio object that fulfills the algorithm
protocol, use the PerformancePeriod.as_portfolio method. See that
method for comments on the specific fields provided (and
omitted).

    +---------------+------------------------------------------------------+
    | key           | value                                                |
    +===============+======================================================+
    | ending_value  | the total market value of the positions held at the  |
    |               | end of the period                                    |
    +---------------+------------------------------------------------------+
    | cash_flow     | the cash flow in the period (negative means spent)   |
    |               | from buying and selling securities in the period.    |
    |               | Includes dividend payments in the period as well.    |
    +---------------+------------------------------------------------------+
    | starting_value| the total market value of the positions held at the  |
    |               | start of the period                                  |
    +---------------+------------------------------------------------------+
    | starting_cash | cash on hand at the beginning of the period          |
    +---------------+------------------------------------------------------+
    | ending_cash   | cash on hand at the end of the period                |
    +---------------+------------------------------------------------------+
    | positions     | a list of dicts representing positions, see          |
    |               | :py:meth:`Position.to_dict()`                        |
    |               | for details on the contents of the dict              |
    +---------------+------------------------------------------------------+
    | pnl           | Dollar value profit and loss, for both realized and  |
    |               | unrealized gains.                                    |
    +---------------+------------------------------------------------------+
    | returns       | percentage returns for the entire portfolio over the |
    |               | period                                               |
    +---------------+------------------------------------------------------+
    | cumulative\   | The net capital used (positive is spent) during      |
    | _capital_used | the period                                           |
    +---------------+------------------------------------------------------+
    | max_capital\  | The maximum amount of capital deployed during the    |
    | _used         | period.                                              |
    +---------------+------------------------------------------------------+
    | max_leverage  | The maximum leverage used during the period.         |
    +---------------+------------------------------------------------------+
    | period_close  | The last close of the market in period. datetime in  |
    |               | pytz.utc timezone.                                   |
    +---------------+------------------------------------------------------+
    | period_open   | The first open of the market in period. datetime in  |
    |               | pytz.utc timezone.                                   |
    +---------------+------------------------------------------------------+
    | transactions  | all the transactions that were acrued during this    |
    |               | period. Unset/missing for cumulative periods.        |
    +---------------+------------------------------------------------------+


"""

import logbook
import math

import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

import zipline.protocol as zp
import zipline.finance.risk as risk
import zipline.finance.trading as trading

log = logbook.Logger('Performance')


class PerformanceTracker(object):
    """
    Tracks the performance of the algorithm.
    """

    def __init__(self, sim_params):

        self.sim_params = sim_params

        self.period_start = self.sim_params.period_start
        self.period_end = self.sim_params.period_end
        self.last_close = self.sim_params.last_close
        first_day = self.sim_params.first_open
        self.market_open, self.market_close = \
            trading.environment.get_open_and_close(first_day)
        self.total_days = self.sim_params.days_in_period
        self.capital_base = self.sim_params.capital_base
        self.emission_rate = sim_params.emission_rate
        self.emission_rate = sim_params.emission_rate

        self.perf_periods = []

        if self.emission_rate == 'daily':
            self.all_benchmark_returns = pd.Series(
                index=trading.environment.trading_days)
            self.intraday_risk_metrics = None
            self.cumulative_risk_metrics = \
                risk.RiskMetricsIterative(self.sim_params)

        elif self.emission_rate == 'minute':
            self.all_benchmark_returns = pd.Series(index=pd.date_range(
                self.sim_params.first_open, self.sim_params.last_close,
                freq='Min'))
            self.intraday_risk_metrics = \
                risk.RiskMetricsIterative(self.sim_params)

            self.cumulative_risk_metrics = \
                risk.RiskMetricsIterative(self.sim_params)
            self.cumulative_risk_metrics.initialize_daily_indices()

            self.minute_performance = PerformancePeriod(
                # initial cash is your capital base.
                self.capital_base,
                # the cumulative period will be calculated over the
                # entire test.
                self.period_start,
                self.period_end,
                # don't save the transactions for the cumulative
                # period
                keep_transactions=False,
                keep_orders=False,
                # don't serialize positions for cumualtive period
                serialize_positions=False
            )
            self.perf_periods.append(self.minute_performance)

        # this performance period will span the entire simulation from
        # inception.
        self.cumulative_performance = PerformancePeriod(
            # initial cash is your capital base.
            self.capital_base,
            # the cumulative period will be calculated over the entire test.
            self.period_start,
            self.period_end,
            # don't save the transactions for the cumulative
            # period
            keep_transactions=False,
            keep_orders=False,
            # don't serialize positions for cumualtive period
            serialize_positions=False
        )
        self.perf_periods.append(self.cumulative_performance)

        # this performance period will span just the current market day
        self.todays_performance = PerformancePeriod(
            # initial cash is your capital base.
            self.capital_base,
            # the daily period will be calculated for the market day
            self.market_open,
            self.market_close,
            keep_transactions=True,
            keep_orders=True,
            serialize_positions=True
        )
        self.perf_periods.append(self.todays_performance)

        self.saved_dt = self.period_start
        self.returns = []
        # one indexed so that we reach 100%
        self.day_count = 0.0
        self.txn_count = 0
        self.event_count = 0

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

    def get_portfolio(self):
        return self.cumulative_performance.as_portfolio()

    def to_dict(self, emission_type=None):
        """
        Creates a dictionary representing the state of this tracker.
        Returns a dict object of the form described in header comments.
        """
        if not emission_type:
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
            _dict.update({'daily_perf': self.todays_performance.to_dict()})
        elif emission_type == 'minute':
            _dict.update({
                'intraday_risk_metrics': self.intraday_risk_metrics.to_dict(),
                'minute_perf': self.todays_performance.to_dict(self.saved_dt)
            })

        return _dict

    def process_event(self, event):

        self.event_count += 1

        if event.type == zp.DATASOURCE_TYPE.TRADE:
            #update last sale
            for perf_period in self.perf_periods:
                perf_period.update_last_sale(event)

        elif event.type == zp.DATASOURCE_TYPE.TRANSACTION:
            # Trade simulation always follows a transaction with the
            # TRADE event that was used to simulate it, so we don't
            # check for end of day rollover messages here.
            self.txn_count += 1
            for perf_period in self.perf_periods:
                perf_period.execute_transaction(event)

        elif event.type == zp.DATASOURCE_TYPE.DIVIDEND:
            for perf_period in self.perf_periods:
                perf_period.add_dividend(event)

        elif event.type == zp.DATASOURCE_TYPE.ORDER:
            for perf_period in self.perf_periods:
                perf_period.record_order(event)

        elif event.type == zp.DATASOURCE_TYPE.CUSTOM:
            pass
        elif event.type == zp.DATASOURCE_TYPE.BENCHMARK:
            if (
                self.sim_params.data_frequency == 'minute'
                and
                self.sim_params.emission_rate == 'daily'
            ):
                # Minute data benchmarks should have a timestamp of market
                # close, so that calculations are triggered at the right time.
                # However, risk module uses midnight as the 'day'
                # marker for returns, so adjust back to midgnight.
                midnight = event.dt.replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0)
            else:
                midnight = event.dt

            self.all_benchmark_returns[midnight] = event.returns

        #calculate performance as of last trade
        for perf_period in self.perf_periods:
            perf_period.calculate_performance()

    def handle_minute_close(self, dt):
        todays_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        minute_returns = self.minute_performance.returns
        self.minute_performance.rollover()
        algo_minute_returns = pd.Series({dt: minute_returns})
        bench_minute_returns = pd.Series({dt: self.all_benchmark_returns[dt]})
        # the intraday risk is calculated on top of minute performance
        # returns for the bench and the algo
        self.intraday_risk_metrics.update(dt,
                                          algo_minute_returns,
                                          bench_minute_returns)

        bench_since_open = \
            self.intraday_risk_metrics.benchmark_period_returns[-1]

        benchmark_returns = pd.Series({todays_date: bench_since_open})

        # if we've reached market close, check on dividends
        if dt == self.market_close:
            for perf_period in self.perf_periods:
                perf_period.update_dividends(todays_date)

        algorithm_returns = pd.Series({
            todays_date: self.todays_performance.returns
        })
        self.cumulative_risk_metrics.update(todays_date,
                                            algorithm_returns,
                                            benchmark_returns)

        # if this is the close, save the returns objects for cumulative
        # risk calculations
        if dt == self.market_close:
            todays_return_obj = zp.DailyReturn(
                todays_date,
                self.todays_performance.returns
            )
            self.returns.append(todays_return_obj)

    def handle_intraday_close(self):
        self.intraday_risk_metrics = \
            risk.RiskMetricsIterative(self.sim_params)
        # increment the day counter before we move markers forward.
        self.day_count += 1.0
        # move the market day markers forward
        if self.market_close < trading.environment.last_trading_day:
            _, self.market_close = \
                trading.environment.next_open_and_close(self.market_open)
        else:
            self.market_close = self.sim_params.last_close

    def handle_market_close(self):
        # add the return results from today to the list of DailyReturn objects.
        todays_date = self.market_close.replace(hour=0, minute=0, second=0,
                                                microsecond=0)
        self.cumulative_performance.update_dividends(todays_date)
        self.todays_performance.update_dividends(todays_date)

        todays_return_obj = zp.DailyReturn(
            todays_date,
            self.todays_performance.returns
        )
        self.returns.append(todays_return_obj)

        #update risk metrics for cumulative performance
        self.cumulative_risk_metrics.update(
            todays_return_obj.date,
            todays_return_obj.returns,
            self.all_benchmark_returns[todays_return_obj.date])

        # increment the day counter before we move markers forward.
        self.day_count += 1.0

        # Take a snapshot of our current peformance to return to the
        # browser.
        daily_update = self.to_dict()

        # On the last day of the test, don't create tomorrow's performance
        # period.  We may not be able to find the next trading day if we're
        # at the end of our historical data
        if self.market_close >= self.last_close:
            return daily_update

        #move the market day markers forward
        self.market_open, self.market_close = \
            trading.environment.next_open_and_close(self.market_open)

        # Roll over positions to current day.
        self.todays_performance.rollover()
        self.todays_performance.period_open = self.market_open
        self.todays_performance.period_close = self.market_close

        # The dividend calculation for the daily needs to be made
        # after the rollover. midnight_between is the last midnight
        # hour between the close of markets and the next open. To
        # make sure midnight_between matches identically with
        # dividend data dates, it is in UTC.
        midnight_between = self.market_open.replace(hour=0, minute=0, second=0,
                                                    microsecond=0)
        self.cumulative_performance.update_dividends(midnight_between)
        self.todays_performance.update_dividends(midnight_between)

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

        bms = self.cumulative_risk_metrics.benchmark_returns
        ars = self.cumulative_risk_metrics.algorithm_returns
        self.risk_report = risk.RiskReport(
            ars,
            self.sim_params,
            benchmark_returns=bms)

        risk_dict = self.risk_report.to_dict()
        return risk_dict


class Position(object):

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0
        self.last_sale_date = 0.0
        self.dividends = []

    def update_dividends(self, midnight_utc):
        """
        midnight_utc is the 0 hour for the current (not yet open) trading day.
        This method will be invoked at the end of the market
        close handling, before the next market open.
        """
        payment = 0.0
        unpaid_dividends = []
        for dividend in self.dividends:
            if midnight_utc == dividend.ex_date:
                # if we own shares at midnight of the div_ex date
                # we are entitled to the dividend.
                dividend.amount_on_ex_date = self.amount
                if dividend.net_amount:
                    dividend.payment = self.amount * dividend.net_amount
                else:
                    dividend.payment = self.amount * dividend.gross_amount

            if midnight_utc == dividend.pay_date:
                # if it is the payment date, include this
                # dividend's actual payment (calculated on
                # ex_date)
                payment += dividend.payment
            else:
                unpaid_dividends.append(dividend)

        self.dividends = unpaid_dividends
        return payment

    def add_dividend(self, dividend):
        self.dividends.append(dividend)

    def update(self, txn):
        if(self.sid != txn.sid):
            raise NameError('updating position with txn for a different sid')

         #we're covering a short or closing a position
        if(self.amount + txn.amount == 0):
            self.cost_basis = 0.0
            self.amount = 0
        else:
            prev_cost = self.cost_basis * self.amount
            txn_cost = txn.amount * txn.price
            total_cost = prev_cost + txn_cost
            total_shares = self.amount + txn.amount
            self.cost_basis = total_cost / total_shares
            self.amount = self.amount + txn.amount

    def __repr__(self):
        template = "sid: {sid}, amount: {amount}, cost_basis: {cost_basis}, \
        last_sale_price: {last_sale_price}"
        return template.format(
            sid=self.sid,
            amount=self.amount,
            cost_basis=self.cost_basis,
            last_sale_price=self.last_sale_price
        )

    def to_dict(self):
        """
        Creates a dictionary representing the state of this position.
        Returns a dict object of the form:
        """
        return {
            'sid': self.sid,
            'amount': self.amount,
            'cost_basis': self.cost_basis,
            'last_sale_price': self.last_sale_price
        }


class PerformancePeriod(object):

    def __init__(
            self,
            starting_cash,
            period_open=None,
            period_close=None,
            keep_transactions=True,
            keep_orders=False,
            serialize_positions=True):

        self.period_open = period_open
        self.period_close = period_close

        self.ending_value = 0.0
        self.period_cash_flow = 0.0
        self.pnl = 0.0
        # sid => position object
        self.positions = positiondict()
        self.starting_value = 0.0
        # cash balance at start of period
        self.starting_cash = starting_cash
        self.ending_cash = starting_cash
        self.keep_transactions = keep_transactions
        self.processed_transactions = defaultdict(list)
        self.keep_orders = keep_orders
        self.orders_by_modified = defaultdict(list)
        self.orders_by_id = OrderedDict()
        self.cumulative_capital_used = 0.0
        self.max_capital_used = 0.0
        self.max_leverage = 0.0

        # Maps position to following array indexes
        self._position_index_map = {}
        # Arrays for quick calculations of positions value
        self._position_amounts = np.array([])
        self._position_last_sale_prices = np.array([])

        self.calculate_performance()

        # An object to recycle via assigning new values
        # when returning portfolio information.
        # So as not to avoid creating a new object for each event
        self._portfolio_store = zp.Portfolio()
        self._positions_store = zp.Positions()
        self.serialize_positions = serialize_positions

    def rollover(self):
        self.starting_value = self.ending_value
        self.starting_cash = self.ending_cash
        self.period_cash_flow = 0.0
        self.pnl = 0.0
        self.processed_transactions = defaultdict(list)
        self.orders_by_modified = defaultdict(list)
        self.orders_by_id = OrderedDict()
        self.cumulative_capital_used = 0.0
        self.max_capital_used = 0.0
        self.max_leverage = 0.0

    def index_for_position(self, sid):
        try:
            index = self._position_index_map[sid]
        except KeyError:
            index = len(self._position_index_map)
            self._position_index_map[sid] = index
            self._position_amounts = np.append(self._position_amounts, [0])
            self._position_last_sale_prices = np.append(
                self._position_last_sale_prices, [0])
        return index

    def add_dividend(self, div):
        # The dividend is received on midnight of the dividend
        # declared date. We calculate the dividends based on the amount of
        # stock owned on midnight of the ex dividend date. However, the cash
        # is not dispersed until the payment date, which is
        # included in the event.
        self.positions[div.sid].add_dividend(div)

    def update_dividends(self, todays_date):
        """
        Check the payment date and ex date against today's date
        to detrmine if we are owed a dividend payment or if the
        payment has been disbursed.
        """
        cash_payments = 0.0
        for sid, pos in self.positions.iteritems():
            cash_payments += pos.update_dividends(todays_date)

        # credit our cash balance with the dividend payments, or
        # if we are short, debit our cash balance with the
        # payments.
        self.period_cash_flow += cash_payments
        # debit our cumulative cash spent with the dividend
        # payments, or credit our cumulative cash spent if we are
        # short the stock.
        self.cumulative_capital_used -= cash_payments

        # recalculate performance, including the dividend
        # paymtents
        self.calculate_performance()

    def calculate_performance(self):
        self.ending_value = self.calculate_positions_value()

        total_at_start = self.starting_cash + self.starting_value
        self.ending_cash = self.starting_cash + self.period_cash_flow
        total_at_end = self.ending_cash + self.ending_value

        self.pnl = total_at_end - total_at_start
        if total_at_start != 0:
            self.returns = self.pnl / total_at_start
        else:
            self.returns = 0.0

    def record_order(self, order):
        if self.keep_orders:
            self.orders_by_modified[order.dt].append(order)
            # to preserve the order of the orders by modified date
            # we delete and add back. (ordered dictionary is sorted by
            # first insertion date).
            if order.id in self.orders_by_id:
                del self.orders_by_id[order.id]
            self.orders_by_id[order.id] = order

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        position = self.positions[txn.sid]
        position.update(txn)
        index = self.index_for_position(txn.sid)
        self._position_amounts[index] = position.amount

        self.period_cash_flow += -1 * txn.price * txn.amount

        # Max Leverage
        # ---------------
        # Calculate the maximum capital used and maximum leverage
        transaction_cost = txn.price * txn.amount
        self.cumulative_capital_used += transaction_cost

        if math.fabs(self.cumulative_capital_used) > self.max_capital_used:
            self.max_capital_used = math.fabs(self.cumulative_capital_used)

            # We want to conveye a level, rather than a precise figure.
            # round to the nearest 5,000 to keep the number easy on the eyes
            self.max_capital_used = self.round_to_nearest(
                self.max_capital_used,
                base=5000
            )

            # we're adding a 10% cushion to the capital used.
            self.max_leverage = 1.1 * \
                self.max_capital_used / self.starting_cash

        # add transaction to the list of processed transactions
        if self.keep_transactions:
            self.processed_transactions[txn.dt].append(txn)

    def round_to_nearest(self, x, base=5):
        return int(base * round(float(x) / base))

    def calculate_positions_value(self):
        return np.dot(self._position_amounts, self._position_last_sale_prices)

    def update_last_sale(self, event):
        is_trade = event.type == zp.DATASOURCE_TYPE.TRADE
        has_price = not np.isnan(event.price)
        # isnan check will keep the last price if its not present
        if (event.sid in self.positions) and is_trade and has_price:
            self.positions[event.sid].last_sale_price = event.price
            index = self.index_for_position(event.sid)
            self._position_last_sale_prices[index] = event.price

            self.positions[event.sid].last_sale_date = event.dt

    def __core_dict(self):
        rval = {
            'ending_value': self.ending_value,
            # this field is renamed to capital_used for backward
            # compatibility.
            'capital_used': self.period_cash_flow,
            'starting_value': self.starting_value,
            'starting_cash': self.starting_cash,
            'ending_cash': self.ending_cash,
            'portfolio_value': self.ending_cash + self.ending_value,
            'cumulative_capital_used': self.cumulative_capital_used,
            'max_capital_used': self.max_capital_used,
            'max_leverage': self.max_leverage,
            'pnl': self.pnl,
            'returns': self.returns,
            'period_open': self.period_open,
            'period_close': self.period_close
        }

        return rval

    def to_dict(self, dt=None):
        """
        Creates a dictionary representing the state of this performance
        period. See header comments for a detailed description.

        Kwargs:
            dt (datetime): If present, only return transactions for the dt.
        """
        rval = self.__core_dict()

        if self.serialize_positions:
            positions = self.get_positions_list()
            rval['positions'] = positions

        # we want the key to be absent, not just empty
        if self.keep_transactions:
            if dt:
                # Only include transactions for given dt
                transactions = [x.to_dict()
                                for x in self.processed_transactions[dt]]
            else:
                transactions = \
                    [y.to_dict()
                     for x in self.processed_transactions.itervalues()
                     for y in x]
            rval['transactions'] = transactions

        if self.keep_orders:
            if dt:
                # only include orders modified as of the given dt.
                orders = [x.to_dict() for x in self.orders_by_modified[dt]]
            else:
                orders = [x.to_dict() for x in self.orders_by_id.itervalues()]
            rval['orders'] = orders

        return rval

    def as_portfolio(self):
        """
        The purpose of this method is to provide a portfolio
        object to algorithms running inside the same trading
        client. The data needed is captured raw in a
        PerformancePeriod, and in this method we rename some
        fields for usability and remove extraneous fields.
        """
        # Recycles containing objects' Portfolio object
        # which is used for returning values.
        # as_portfolio is called in an inner loop,
        # so repeated object creation becomes too expensive
        portfolio = self._portfolio_store
        # maintaining the old name for the portfolio field for
        # backward compatibility
        portfolio.capital_used = self.period_cash_flow
        portfolio.starting_cash = self.starting_cash
        portfolio.portfolio_value = self.ending_cash + self.ending_value
        portfolio.pnl = self.pnl
        portfolio.returns = self.returns
        portfolio.cash = self.ending_cash
        portfolio.start_date = self.period_open
        portfolio.positions = self.get_positions()
        portfolio.positions_value = self.ending_value
        return portfolio

    def get_positions(self):

        positions = self._positions_store

        for sid, pos in self.positions.iteritems():

            if sid not in positions:
                positions[sid] = zp.Position(sid)
            position = positions[sid]
            position.amount = pos.amount
            position.cost_basis = pos.cost_basis
            position.last_sale_price = pos.last_sale_price

        return positions

    def get_positions_list(self):
        positions = []
        for sid, pos in self.positions.iteritems():
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions


class positiondict(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos
