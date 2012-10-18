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
    | started_at      | datetime in utc marking the start of this test     |
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
    | capital_used  | the net capital consumed (positive means spent) by   |
    |               | buying and selling securities in the period          |
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
    | cumulative_   | The net capital used (positive is spent) during      |
    | capital_used  | the period                                           |
    +---------------+------------------------------------------------------+
    | max_capital_  | The maximum amount of capital deployed during the    |
    | used          | period.                                              |
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
import datetime
import pytz
import math

from zipline.utils.protocol_utils import ndict
import zipline.protocol as zp
import zipline.finance.risk as risk

log = logbook.Logger('Performance')


class PerformanceTracker(object):
    """
    Tracks the performance of the zipline as it is running in
    the simulator, relays this out to the Deluge broker and then
    to the client. Visually:

        +--------------------+   Result Stream   +--------+
        | PerformanceTracker | ----------------> | Deluge |
        +--------------------+                   +--------+

    """

    def __init__(self, trading_environment):

        self.trading_environment = trading_environment
        self.trading_day = datetime.timedelta(hours=6, minutes=30)
        self.calendar_day = datetime.timedelta(hours=24)
        self.started_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        self.period_start = self.trading_environment.period_start
        self.period_end = self.trading_environment.period_end
        self.market_open = self.trading_environment.first_open
        self.market_close = self.market_open + self.trading_day
        self.progress = 0.0
        self.total_days = self.trading_environment.days_in_period
        # one indexed so that we reach 100%
        self.day_count = 0.0
        self.capital_base = self.trading_environment.capital_base
        self.returns = []
        self.txn_count = 0
        self.event_count = 0
        self.last_dict = None
        self.cumulative_risk_metrics = risk.RiskMetricsIterative(
            self.period_start, self.trading_environment)

        # this performance period will span the entire simulation.
        self.cumulative_performance = PerformancePeriod(
            # initial positions are empty
            positiondict(),
            # initial portfolio positions have zero value
            0,
            # initial cash is your capital base.
            self.capital_base,
            # the cumulative period will be calculated over the entire test.
            self.period_start,
            self.period_end
        )

        # this performance period will span just the current market day
        self.todays_performance = PerformancePeriod(
            # initial positions are empty
            positiondict(),
            # initial portfolio positions have zero value
            0,
            # initial cash is your capital base.
            self.capital_base,
            # the daily period will be calculated for the market day
            self.market_open,
            self.market_close,
            # save the transactions for the daily periods
            keep_transactions=True
        )

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        for event in stream_in:
            if event.dt == "DONE":
                event.perf_message = self.handle_simulation_end()
                del event['TRANSACTION']
                yield event
            else:
                event.perf_message = self.process_event(event)
                event.portfolio = self.get_portfolio()
                del event['TRANSACTION']
                yield event

    def get_portfolio(self):
        return self.cumulative_performance.as_portfolio()

    def to_dict(self):
        """
        Creates a dictionary representing the state of this tracker.
        Returns a dict object of the form described in header comments.
        """
        return {
            'started_at': self.started_at,
            'period_start': self.period_start,
            'period_end': self.period_end,
            'progress': self.progress,
            'capital_base': self.capital_base,
            'cumulative_perf': self.cumulative_performance.to_dict(),
            'daily_perf': self.todays_performance.to_dict(),
            'cumulative_risk_metrics': self.cumulative_risk_metrics.to_dict()
        }

    def process_event(self, event):

        message = None

        assert isinstance(event, ndict)
        self.event_count += 1

        if(event.dt >= self.market_close):
            message = self.handle_market_close()

        if event.TRANSACTION:
            self.txn_count += 1
            self.cumulative_performance.execute_transaction(event.TRANSACTION)
            self.todays_performance.execute_transaction(event.TRANSACTION)

        #update last sale
        self.cumulative_performance.update_last_sale(event)
        self.todays_performance.update_last_sale(event)

        #calculate performance as of last trade
        self.cumulative_performance.calculate_performance()
        self.todays_performance.calculate_performance()

        return message

    def handle_market_close(self):

        # add the return results from today to the list of DailyReturn objects.
        todays_date = self.market_close.replace(hour=0, minute=0, second=0)
        todays_return_obj = risk.DailyReturn(
            todays_date,
            self.todays_performance.returns
        )
        self.returns.append(todays_return_obj)

        #update risk metrics for cumulative performance
        self.cumulative_risk_metrics.update(
            self.todays_performance.returns, datetime.timedelta(days=1))

        # increment the day counter before we move markers forward.
        self.day_count += 1.0
        # calculate progress of test
        self.progress = self.day_count / self.total_days

        # Take a snapshot of our current peformance to return to the
        # browser.
        daily_update = self.to_dict()

        #move the market day markers forward
        self.market_open = self.market_open + self.calendar_day

        while not self.trading_environment.is_trading_day(self.market_open):
            if self.market_open > self.trading_environment.trading_days[-1]:
                raise Exception(
                    "Attempt to backtest beyond available history.")
            self.market_open = self.market_open + self.calendar_day

        self.market_close = self.market_open + self.trading_day

        # Roll over positions to current day.
        self.todays_performance = PerformancePeriod(
            self.todays_performance.positions,
            self.todays_performance.ending_value,
            self.todays_performance.ending_cash,
            self.market_open,
            self.market_close,
            keep_transactions=True
        )

        return daily_update

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """

        log_msg = "Simulated {n} trading days out of {m}."
        log.info(log_msg.format(n=self.day_count, m=self.total_days))
        log.info("first open: {d}".format(
            d=self.trading_environment.first_open))

        # the stream will end on the last trading day, but will not trigger
        # an end of day, so we trigger the final market close here.
        self.handle_market_close()

        self.risk_report = risk.RiskReport(
            self.returns,
            self.trading_environment
        )

        risk_dict = self.risk_report.to_dict()
        return risk_dict


class Position(object):

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0
        self.last_sale_date = 0.0

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

    def currentValue(self):
        return self.amount * self.last_sale_price

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
        initial_positions,
        starting_value,
        starting_cash,
        period_open=None,
        period_close=None,
        keep_transactions=False):

        self.period_open = period_open
        self.period_close = period_close

        self.ending_value = 0.0
        self.period_capital_used = 0.0
        self.pnl = 0.0
        #sid => position object
        if not isinstance(initial_positions, positiondict):
            self.positions = positiondict()
            self.positions.update(initial_positions)
        else:
            self.positions = initial_positions
        self.starting_value = starting_value
        #cash balance at start of period
        self.starting_cash = starting_cash
        self.ending_cash = starting_cash
        self.keep_transactions = keep_transactions
        self.processed_transactions = []
        self.cumulative_capital_used = 0.0
        self.max_capital_used = 0.0
        self.max_leverage = 0.0

        self.calculate_performance()

    def calculate_performance(self):
        self.ending_value = self.calculate_positions_value()

        total_at_start = self.starting_cash + self.starting_value
        self.ending_cash = self.starting_cash + self.period_capital_used
        total_at_end = self.ending_cash + self.ending_value

        self.pnl = total_at_end - total_at_start
        if total_at_start != 0:
            self.returns = self.pnl / total_at_start
        else:
            self.returns = 0.0

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        self.positions[txn.sid].update(txn)
        self.period_capital_used += -1 * txn.price * txn.amount

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
            self.processed_transactions.append(txn)

    def round_to_nearest(self, x, base=5):
        return int(base * round(float(x) / base))

    def calculate_positions_value(self):
        mktValue = 0.0
        for key, pos in self.positions.iteritems():
            mktValue += pos.currentValue()
        return mktValue

    def update_last_sale(self, event):
        is_trade = event.type == zp.DATASOURCE_TYPE.TRADE
        if event.sid in self.positions and is_trade:
            self.positions[event.sid].last_sale_price = event.price
            self.positions[event.sid].last_sale_date = event.dt

    def __core_dict(self):
        rval = {
            'ending_value': self.ending_value,
            'capital_used': self.period_capital_used,
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

    def to_dict(self):
        """
        Creates a dictionary representing the state of this performance
        period. See header comments for a detailed description.
        """
        rval = self.__core_dict()
        positions = self.get_positions_list()
        rval['positions'] = positions

        # we want the key to be absent, not just empty
        if self.keep_transactions:
            transactions = [x.as_dict() for x in self.processed_transactions]
            rval['transactions'] = transactions

        return rval

    def as_portfolio(self):
        """
        The purpose of this method is to provide a portfolio
        object to algorithms running inside the same trading
        client. The data needed is captured raw in a
        PerformancePeriod, and in this method we rename some
        fields for usability and remove extraneous fields.
        """
        portfolio = self.__core_dict()
        # rename:
        # ending_cash -> cash
        # period_open -> backtest_start
        #
        # remove:
        # period_close, starting_value,
        # cumulative_capital_used, max_leverage, max_capital_used
        portfolio['cash'] = portfolio['ending_cash']
        portfolio['start_date'] = portfolio['period_open']
        portfolio['positions_value'] = portfolio['ending_value']

        del(portfolio['ending_cash'])
        del(portfolio['period_open'])
        del(portfolio['period_close'])
        del(portfolio['starting_value'])
        del(portfolio['ending_value'])
        del(portfolio['cumulative_capital_used'])
        del(portfolio['max_leverage'])
        del(portfolio['max_capital_used'])

        portfolio['positions'] = self.get_positions()
        return ndict(portfolio)

    def get_positions(self):

        positions = ndict(internal=position_ndict())

        for sid, pos in self.positions.iteritems():
            cur = pos.to_dict()
            positions[sid] = ndict(cur)

        return positions

    def get_positions_list(self):
        positions = []
        for sid, pos in self.positions.iteritems():
            cur = pos.to_dict()
            positions.append(cur)
        return positions


class positiondict(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos


class position_ndict(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = ndict(pos.to_dict())
        return pos
