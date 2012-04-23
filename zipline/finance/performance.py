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
        self.result_stream           = None
        self.last_dict               = None
        self.order_log               = []
        self.exceeded_max_loss       = False

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
        assert isinstance(event, zp.namedict)
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

        if self.trading_environment.max_drawdown:
            max_dd = -1 * self.trading_environment.max_drawdown
            if self.todays_performance.returns < max_dd:
                qutil.LOGGER.info("Exceeded max drawdown.")
                # mark the perf period with max loss flag, 
                # so it shows up in the update, but don't end the test
                # here. Let the update go out before stopping
                self.exceeded_max_loss = True
                
        # Output results
        if self.result_stream:
            msg = zp.PERF_FRAME(self.to_dict())
            self.result_stream.send(msg)
            
        if self.exceeded_max_loss:
            # now that we've sent the day's update, kill this test
            self.handle_simulation_end(skip_close=True)
            return
            
        # check the day's returns versus the max drawdown
        # max_drawdown is optional:
        if self.trading_environment.max_drawdown:
            max_dd = -1 * self.trading_environment.max_drawdown
            if self.todays_performance.returns < max_dd:
                qutil.LOGGER.info("Exceeded max drawdown.")
                # TODO: any other information we need to relay on the 
                # result socket?
                self.exceeded_max_loss = True
                self.handle_simulation_end(skip_close=True)
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

    def handle_simulation_end(self, skip_close=False):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the result_stream.
        """
        
        log_msg = "Simulated {n} trading days out of {m}."
        qutil.LOGGER.info(log_msg.format(n=self.day_count, m=self.total_days))
        qutil.LOGGER.info("first open: {d}".format(d=self.trading_environment.first_open))
        
        # the stream will end on the last trading day, but will not trigger
        # an end of day, so we trigger the final market close here.
        # In the case of errors, we needn't close again.
        if not skip_close:
            self.handle_market_close()
        
        self.risk_report = risk.RiskReport(
            self.returns,
            self.trading_environment,
            exceeded_max_loss = self.exceeded_max_loss
        )
        
        if self.result_stream:
            qutil.LOGGER.info("about to stream the risk report...")
            risk_dict = self.risk_report.to_dict()
            
            msg = zp.RISK_FRAME(risk_dict)
            self.result_stream.send(msg)
            # this signals that the simulation is complete.
            self.result_stream.send("DONE")


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
            'last_sale_price' : self.last_sale_price
        }


class PerformancePeriod():

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

    def to_dict(self):
        """
        Creates a dictionary representing the state of this performance 
        period. See header comments for a detailed description.
        """
        positions = self.get_positions_list()
        transactions = [x.as_dict() for x in self.processed_transactions]

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
            'positions'                 : positions,
            'pnl'                       : self.pnl,
            'returns'                   : self.returns,
            'transactions'              : transactions,
            'period_open'               : self.period_open,
            'period_close'              : self.period_close
        }
        
        # we want the key to be absent, not just empty
        if not self.keep_transactions:
            del(rval['transactions'])
        
        return rval
        
    def to_namedict(self):
        """
        Creates a namedict representing the state of this perfomance period.
        Properties are the same as the results of to_dict. See header comments
        for a detailed description.    
        
        """
        positions = self.get_positions(namedicted=True)
        
        positions = zp.namedict(positions)
        
        return zp.namedict({
            'ending_value'              : self.ending_value,
            'capital_used'              : self.period_capital_used,
            'starting_value'            : self.starting_value,
            'starting_cash'             : self.starting_cash,
            'ending_cash'               : self.ending_cash,
            'cumulative_capital_used'   : self.cumulative_capital_used,
            'max_capital_used'          : self.max_capital_used,
            'max_leverage'              : self.max_leverage,    
            'positions'                 : positions,
            'transactions'              : self.processed_transactions
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
        
    #
    def get_positions_list(self):
        positions = []
        for sid, pos in self.positions.iteritems():
            cur = pos.to_dict()
            positions.append(cur)
        return positions
        
        
            

