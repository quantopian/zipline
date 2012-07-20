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
    | exceeded_max_   | True if the simulation was stopped because single  |
    | loss            | day losses exceeded the max_drawdown stipulated in |
    |                 | trading_environment.                               |
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

import zmq

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

        self.trading_environment     = trading_environment
        self.trading_day             = datetime.timedelta(hours = 6, minutes = 30)
        self.calendar_day            = datetime.timedelta(hours = 24)
        self.started_at              = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        self.period_start            = self.trading_environment.period_start
        self.period_end              = self.trading_environment.period_end
        self.market_open             = self.trading_environment.first_open
        self.market_close            = self.market_open + self.trading_day
        self.progress                = 0.0
        self.total_days              = self.trading_environment.days_in_period
        # one indexed so that we reach 100%
        self.day_count               = 0.0
        self.capital_base            = self.trading_environment.capital_base
        self.returns                 = []
        self.txn_count               = 0
        self.event_count             = 0
        self.last_dict               = None
        self.order_log               = []
        self.exceeded_max_loss       = False

        self.results_socket = None
        self.results_addr   = None

        # this performance period will span the entire simulation.
        self.cumulative_performance = PerformancePeriod(
            # initial positions are empty
            {},
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
            {},
            # initial portfolio positions have zero value
            0,
            # initial cash is your capital base.
            self.capital_base,
            # the daily period will be calculated for the market day
            self.market_open,
            self.market_close,
            # save the transactions for the daily periods
            keep_transactions = True
        )

    def set_sids(self, sid_list):
        for sid in sid_list:
            self.cumulative_performance.positions[sid] = Position(sid)

    def get_portfolio(self):
        return self.cumulative_performance.as_portfolio()

    def open(self, context):
        if self.results_addr:
            sock = context.socket(zmq.PUSH)
            sock.connect(self.results_addr)
            self.results_socket = sock
        else:
            log.warn("Not streaming results because no results socket given")

    def publish_to(self, results_addr):
        """
        Publish the performance results asynchronously to a
        socket.
        """
        #assert isinstance(results_addr, basestring), type(results_addr)
        #self.results_addr = results_addr
        self.results_socket = results_addr

    def to_dict(self):
        """
        Creates a dictionary representing the state of this tracker.
        Returns a dict object of the form described in header comments.
        """
        return {
            'started_at'              : self.started_at,
            'period_start'            : self.period_start,
            'period_end'              : self.period_end,
            'progress'                : self.progress,
            'capital_base'            : self.capital_base,
            'cumulative_perf'         : self.cumulative_performance.to_dict(),
            'daily_perf'              : self.todays_performance.to_dict(),
            'cumulative_risk_metrics' : self.cumulative_risk_metrics.to_dict()
        }

    def log_order(self, order):
        self.order_log.append(order)

    def process_event(self, event):

        if self.exceeded_max_loss:
            return

        assert isinstance(event, zp.ndict)
        self.event_count += 1

        if(event.dt >= self.market_close):
            self.handle_market_close()

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

    def handle_market_close(self):

        # add the return results from today to the list of DailyReturn objects.
        todays_date = self.market_close.replace(hour=0, minute=0, second=0)
        todays_return_obj = risk.DailyReturn(
            todays_date,
            self.todays_performance.returns
        )
        self.returns.append(todays_return_obj)

        #calculate risk metrics for cumulative performance
        self.cumulative_risk_metrics = risk.RiskMetrics(
            start_date=self.period_start,
            end_date=self.market_close.replace(hour=0, minute=0, second=0),
            returns=self.returns,
            trading_environment=self.trading_environment
        )

        # increment the day counter before we move markers forward.
        self.day_count += 1.0
        # calculate progress of test
        self.progress = self.day_count / self.total_days

        # Output results
        if self.results_socket:
            msg = zp.PERF_FRAME(self.to_dict())
            self.results_socket.send(msg)

        #
        if self.trading_environment.max_drawdown:
            returns = self.todays_performance.returns
            max_dd = -1 * self.trading_environment.max_drawdown
            if returns < max_dd:
                log.info(str(returns) + " broke through " + str(max_dd))
                log.info("Exceeded max drawdown.")
                # mark the perf period with max loss flag,
                # so it shows up in the update, but don't end the test
                # here. Let the update go out before stopping
                self.exceeded_max_loss = True
                return


        #move the market day markers forward
        self.market_open = self.market_open + self.calendar_day

        while not self.trading_environment.is_trading_day(self.market_open):
            if self.market_open > self.trading_environment.trading_days[-1]:
                raise Exception("Attempt to backtest beyond available history.")
            self.market_open = self.market_open + self.calendar_day

        self.market_close = self.market_open + self.trading_day

        # Roll over positions to current day.
        self.todays_performance = PerformancePeriod(
            self.todays_performance.positions,
            self.todays_performance.ending_value,
            self.todays_performance.ending_cash,
            self.market_open,
            self.market_close,
            keep_transactions = True
        )

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """

        log_msg = "Simulated {n} trading days out of {m}."
        log.info(log_msg.format(n=self.day_count, m=self.total_days))
        log.info("first open: {d}".format(d=self.trading_environment.first_open))

        # the stream will end on the last trading day, but will not trigger
        # an end of day, so we trigger the final market close here.
        # In the case of max drawdown, we needn't close again.
        if not self.exceeded_max_loss:
            self.handle_market_close()

        self.risk_report = risk.RiskReport(
            self.returns,
            self.trading_environment,
            exceeded_max_loss = self.exceeded_max_loss
        )

        if self.results_socket:
            log.info("about to stream the risk report...")
            risk_dict = self.risk_report.to_dict()

            msg = zp.RISK_FRAME(risk_dict)
            self.results_socket.send(msg)

class Position(object):

    def __init__(self, sid):
        self.sid             = sid
        self.amount          = 0
        self.cost_basis      = 0.0 ##per share
        self.last_sale_price = None
        self.last_sale_date  = None

    def update(self, txn):
        if(self.sid != txn.sid):
            raise NameError('updating position with txn for a different sid')

         #we're covering a short or closing a position
        if(self.amount + txn.amount == 0):
            self.cost_basis = 0.0
            self.amount = 0
        else:
            prev_cost       = self.cost_basis*self.amount
            txn_cost        = txn.amount*txn.price
            total_cost      = prev_cost + txn_cost
            total_shares    = self.amount + txn.amount
            self.cost_basis = total_cost/total_shares
            self.amount     = self.amount + txn.amount

    def currentValue(self):
        return self.amount * self.last_sale_price


    def __repr__(self):
        template = "sid: {sid}, amount: {amount}, cost_basis: {cost_basis}, \
        last_sale_price: {last_sale_price}"
        return template.format(
            sid             = self.sid,
            amount          = self.amount,
            cost_basis      = self.cost_basis,
            last_sale_price = self.last_sale_price
        )

    def to_dict(self):
        """
        Creates a dictionary representing the state of this position.
        Returns a dict object of the form:
        """
        return {
            'sid'             : self.sid,
            'amount'          : self.amount,
            'cost_basis'      : self.cost_basis,
            'last_sale_price' : self.last_sale_price
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

        self.ending_value            = 0.0
        self.period_capital_used     = 0.0
        self.pnl                     = 0.0
        #sid => position object
        self.positions               = initial_positions
        self.starting_value          = starting_value
        #cash balance at start of period
        self.starting_cash           = starting_cash
        self.ending_cash             = starting_cash
        self.keep_transactions       = keep_transactions
        self.processed_transactions  = []
        self.cumulative_capital_used = 0.0
        self.max_capital_used        = 0.0
        self.max_leverage            = 0.0

        self.calculate_performance()

    def calculate_performance(self):
        self.ending_value = self.calculate_positions_value()

        total_at_start      = self.starting_cash + self.starting_value
        self.ending_cash    = self.starting_cash + self.period_capital_used
        total_at_end        = self.ending_cash + self.ending_value

        self.pnl            = total_at_end - total_at_start
        if(total_at_start != 0):
            self.returns = self.pnl / total_at_start
        else:
            self.returns = 0.0

    def execute_transaction(self, txn):

        # Update Position
        # ----------------
        if(not self.positions.has_key(txn.sid)):
            self.positions[txn.sid] = Position(txn.sid)
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
            self.max_leverage = 1.1 * self.max_capital_used / self.starting_cash

        # add transaction to the list of processed transactions
        if self.keep_transactions:
            self.processed_transactions.append(txn)

    def round_to_nearest(self, x, base=5):
        return int(base * round(float(x)/base))

    def calculate_positions_value(self):
        mktValue = 0.0
        for key,pos in self.positions.iteritems():
            mktValue += pos.currentValue()
        return mktValue

    def update_last_sale(self, event):
        is_trade = event.type == zp.DATASOURCE_TYPE.TRADE
        if self.positions.has_key(event.sid) and is_trade:
            self.positions[event.sid].last_sale_price = event.price
            self.positions[event.sid].last_sale_date = event.dt

    def __core_dict(self):
        rval = {
            'ending_value'              : self.ending_value,
            'capital_used'              : self.period_capital_used,
            'starting_value'            : self.starting_value,
            'starting_cash'             : self.starting_cash,
            'ending_cash'               : self.ending_cash,
            'portfolio_value'           : self.ending_cash + self.ending_value,
            'cumulative_capital_used'   : self.cumulative_capital_used,
            'max_capital_used'          : self.max_capital_used,
            'max_leverage'              : self.max_leverage,
            'pnl'                       : self.pnl,
            'returns'                   : self.returns,
            'period_open'               : self.period_open,
            'period_close'              : self.period_close
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

        portfolio['positions'] = self.get_positions(ndicted=True)
        return zp.ndict(portfolio)

    def get_positions(self, ndicted=False):
        if ndicted:
            positions = zp.ndict({})
        else:
            positions = {}

        for sid, pos in self.positions.iteritems():
            cur = pos.to_dict()
            if ndicted:
                positions[sid] = zp.ndict(cur)
            else:
                positions[sid] = cur

        return positions

    #
    def get_positions_list(self):
        positions = []
        for sid, pos in self.positions.iteritems():
            cur = pos.to_dict()
            positions.append(cur)
        return positions
