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
    | cumulative_capti| The net capital used (positive is spent) through   |
    | al_used         | the course of all the events sent to this tracker  |
    +-----------------+----------------------------------------------------+
    | max_capital_used| The maximum amount of capital deployed through the |
    |                 | course of all the events sent to this tracker      |
    +-----------------+----------------------------------------------------+
    | last_close      | The most recent close of the market. datetime in   |
    |                 | pytz.utc timezone. Will always be 23:59 on the     |
    |                 | date in UTC. The fact that the time may be on the  |
    |                 | next day in the exchange's local time is ignored   |
    +-----------------+----------------------------------------------------+
    | last_open       | The most recent open of the market. datetime in    |
    |                 | pytz.utc timezone. Will always be 00:00 on the     |
    |                 | date in UTC. The fact that the time may be on the  |
    |                 | next day in the exchange's local time is ignored   |
    +-----------------+----------------------------------------------------+
    | capital_base    | The initial capital assumed for this tracker.      |
    +-----------------+----------------------------------------------------+
    | returns         | List of dicts representing daily returns. See the  |
    |                 | comments for                                       |
    |                 | :py:meth:`zipline.finance.risk.DailyReturn.to_dict`|
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
    | timestamp       | System time evevent occurs in zipilne              |
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
    | last_sale_date  | datetime of the last trade of the position's       |
    |                 | security on the exchange                           |
    +-----------------+----------------------------------------------------+
    | transactions    | all the transactions that were acrued into this    |
    |                 | position.                                          |
    +-----------------+----------------------------------------------------+
    | timestamp       | System time event occurs in zipilne                |
    +-----------------+----------------------------------------------------+

Performance Period
==================

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
    | timestamp     | System time evevent occurs in zipilne                |
    +---------------+------------------------------------------------------+

"""
import datetime
import pytz
import msgpack
import pandas
import math

import zmq

import zipline.util as qutil
import zipline.protocol as zp
import zipline.finance.risk as risk

class PerformanceTracker():
    """
    Tracks the performance of the zipline as it is running in
    the simulator, relays this out to the Deluge broker and then
    to the client. Visually:
    
        +--------------------+   Result Stream   +--------+
        | PerformanceTracker | ----------------> | Deluge |
        +--------------------+                   +--------+

    """

    def __init__(self, trading_environment):
        
        
        self.trading_environment    = trading_environment
        self.trading_day            = datetime.timedelta(hours = 6, minutes = 30)
        self.calendar_day           = datetime.timedelta(hours = 24)
        self.started_at             = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        self.period_start            = self.trading_environment.period_start
        self.period_end              = self.trading_environment.period_end
        self.market_open             = self.trading_environment.first_open
        self.market_close            = self.market_open + self.trading_day
        self.progress                = 0.0
        self.total_days              = self.trading_environment.days_in_period
        # one indexed so that we reach 100%
        self.day_count               = 0.0 
        self.cumulative_capital_used = 0.0
        self.max_capital_used        = 0.0
        self.capital_base            = self.trading_environment.capital_base
        self.returns                 = []
        self.txn_count               = 0
        self.event_count             = 0
        self.result_stream           = None
        self.last_dict               = None

        # this performance period will span the entire simulation.
        self.cumulative_performance = PerformancePeriod(
            # initial positions are empty
            {},
            # initial portfolio positions have zero value
            0,
            # initial cash is your capital base.
            starting_cash = self.capital_base
        )
        
        # this performance period will span just the current market day
        self.todays_performance = PerformancePeriod(
            # initial positions are empty
            {},
            # initial portfolio positions have zero value
            0,
            # initial cash is your capital base.
            starting_cash = self.capital_base
        )

    def get_portfolio(self):
        return self.cumulative_performance.to_namedict()

    def publish_to(self, zmq_socket, context=None):
        """
        Publish the performance results asynchronously to a
        socket.
        """
        if isinstance(zmq_socket, zmq.Socket):
            self.result_stream = zmq_socket
        else:
            ctx = context or zmq.Context.instance()
            sock = ctx.socket(zmq.PUSH)
            sock.connect(zmq_socket)

            self.result_stream = sock

    def to_dict(self):
        """
        Creates a dictionary representing the state of this tracker.
        Returns a dict object of the form described in header comments.
        """

        returns_list = [x.to_dict() for x in self.returns]

        return {
            'started_at'              : self.started_at,
            'period_start'            : self.period_start,
            'period_end'              : self.period_end,
            'progress'                : self.progress,
            'cumulative_captial_used' : self.cumulative_capital_used,
            'max_capital_used'        : self.max_capital_used,
            'last_close'              : self.market_close,
            'last_open'               : self.market_open,
            'capital_base'            : self.capital_base,
            'returns'                 : returns_list,
            'cumulative_perf'         : self.cumulative_performance.to_dict(),
            'todays_perf'             : self.todays_performance.to_dict(),
            'cumulative_risk_metrics' : self.cumulative_risk_metrics.to_dict(),
            'timestamp'               : datetime.datetime.now(),
        }
            
    def process_event(self, event):
        self.event_count += 1

        if(event.dt >= self.market_close):
            self.handle_market_close()

        if not pandas.isnull(event.TRANSACTION):
            self.txn_count += 1
            self.cumulative_performance.execute_transaction(event.TRANSACTION)
            self.todays_performance.execute_transaction(event.TRANSACTION)

            # we're adding a 10% cushion to the capital used,
            # and then rounding to the nearest 5k
            transaction_cost = event.TRANSACTION.price * event.TRANSACTION.amount
            self.cumulative_capital_used += transaction_cost

            if math.fabs(self.cumulative_capital_used) > self.max_capital_used:
                self.max_capital_used = math.fabs(self.cumulative_capital_used)

            cushioned_capital = 1.1 * self.max_capital_used
            self.max_capital_used = self.round_to_nearest(
                cushioned_capital,
                base=5000
            )
            self.max_leverage = self.max_capital_used / self.capital_base

        #update last sale
        self.cumulative_performance.update_last_sale(event)
        self.todays_performance.update_last_sale(event)

        

    def handle_market_close(self):
        #calculate performance as of last trade
        self.cumulative_performance.calculate_performance()
        self.todays_performance.calculate_performance()
        
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
        if self.result_stream:
            msg = zp.PERF_FRAME(self.to_dict())
            self.result_stream.send(msg)

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
            self.todays_performance.ending_cash
        )

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the result_stream.
        """
        
        # the stream will end on the last trading day, but will not trigger
        # an end of day, so we trigger the final market close here.
        self.handle_market_close()
        
        log_msg = "Simulated {n} trading days out of {m}."
        qutil.LOGGER.info(log_msg.format(n=self.day_count, m=self.total_days))
        qutil.LOGGER.info("first open: {d}".format(d=self.trading_environment.first_open))
        
        self.risk_report = risk.RiskReport(
            self.returns,
            self.trading_environment
        )

        if self.result_stream:
            qutil.LOGGER.info("about to stream the risk report...")
            report = self.risk_report.to_dict()
            msg = zp.RISK_FRAME(report)
            self.result_stream.send(msg)
            # this signals that the simulation is complete.
            self.result_stream.send("DONE")

    def round_to_nearest(self, x, base=5):
        return int(base * round(float(x)/base))


class Position():

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0 ##per share
        self.last_sale_price = None
        self.last_sale_date = None

    def update(self, txn):
        if(self.sid != txn.sid):
            raise NameError('updating position with txn for a different sid')

         #we're covering a short or closing a position
        if(self.amount + txn.amount == 0):
            self.cost_basis = 0.0
            self.amount = 0
        else:
            prev_cost = self.cost_basis*self.amount
            txn_cost = txn.amount*txn.price
            total_cost = prev_cost + txn_cost
            total_shares = self.amount + txn.amount
            self.cost_basis = total_cost/total_shares
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
            'sid'             : self.sid,
            'amount'          : self.amount,
            'cost_basis'      : self.cost_basis,
            'last_sale_price' : self.last_sale_price,
            'last_sale_date'  : self.last_sale_date,
            'timestamp'       : datetime.datetime.now()
        }


class PerformancePeriod():

    def __init__(self, initial_positions, starting_value, starting_cash):
        self.ending_value           = 0.0
        self.period_capital_used    = 0.0
        self.pnl                    = 0.0
        #sid => position object
        self.positions              = initial_positions
        self.starting_value         = starting_value
        #cash balance at start of period
        self.starting_cash          = starting_cash
        self.ending_cash            = starting_cash
        self.processed_transactions = []
        
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
        if(not self.positions.has_key(txn.sid)):
            self.positions[txn.sid] = Position(txn.sid)
        self.positions[txn.sid].update(txn)
        self.period_capital_used += -1 * txn.price * txn.amount
        self.processed_transactions.append(txn)

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

    def to_dict(self):
        """
        Creates a dictionary representing the state of this performance 
        period. See header comments for a detailed description.
        """
        positions = self.get_positions()

        return {
            'ending_value'   : self.ending_value,
            'capital_used'   : self.period_capital_used,
            'starting_value' : self.starting_value,
            'starting_cash'  : self.starting_cash,
            'ending_cash'    : self.ending_cash,
            'portfolio_value': self.ending_cash + self.ending_value,
            'positions'      : positions,
            'timestamp'      : datetime.datetime.now(),
            'pnl'            : self.pnl,
            'returns'        : self.returns,
            'transactions'   : self.processed_transactions,
        }
        
    def to_namedict(self):
        """
        Creates a namedict representing the state of this perfomance period.
        Properties are the same as the results of to_dict. See header comments
        for a detailed description.    
        
        """
        positions = self.get_positions(namedicted=True)
        
        positions = zp.namedict(positions)
        
        return zp.namedict({
            'ending_value'  : self.ending_value,
            'capital_used'   : self.period_capital_used,
            'starting_value' : self.starting_value,
            'starting_cash'  : self.starting_cash,
            'ending_cash'   : self.ending_cash,
            'positions'      : positions
        })
        
    def get_positions(self, namedicted=False):
        positions = {}
        for sid, pos in self.positions.iteritems():
            cur = pos.to_dict()
            if namedicted:
                positions[sid] = zp.namedict(cur)
            else:
                positions[sid] = cur
        
        return positions
        
        
            

