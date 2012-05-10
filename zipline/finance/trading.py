import logging
import datetime
import pytz
import math
import pandas
import time

from collections import Counter

# from gevent.select import select

import zipline.messaging as qmsg
import zipline.protocol as zp
import zipline.finance.performance as perf

from zipline.utils.protocol_utils import Enum, namedict

# the simulation style enumerates the available transaction simulation
# strategies. 
SIMULATION_STYLE  = Enum(
    'PARTIAL_VOLUME',
    'BUY_ALL',
    'FIXED_SLIPPAGE',
    'NOOP'
)

LOGGER = logging.getLogger('ZiplineLogger')

class TradeSimulationClient(qmsg.Component):
    
    def __init__(self, trading_environment, sim_style):
        qmsg.Component.__init__(self)
        self.received_count         = 0
        self.prev_dt                = None
        self.event_queue            = None
        self.txn_count              = 0
        self.order_count            = 0
        self.trading_environment    = trading_environment
        self.current_dt             = trading_environment.period_start
        self.last_iteration_dur     = datetime.timedelta(seconds=0)
        self.algorithm              = None
        self.max_wait               = datetime.timedelta(seconds=60)
        self.last_msg_dt            = datetime.datetime.utcnow()
        self.txn_sim                = TransactionSimulator(sim_style)
        
        assert self.trading_environment.frame_index != None
        self.event_frame = pandas.DataFrame(
            index=self.trading_environment.frame_index
        )
        
        self.perf = perf.PerformanceTracker(self.trading_environment)
    
    @property
    def get_id(self):
        return str(zp.FINANCE_COMPONENT.TRADING_CLIENT)
        
    def set_algorithm(self, algorithm):
        """
        :param algorithm: must implement the algorithm protocol. See 
        :py:mod:`zipline.test.algorithm`
        """
        self.algorithm = algorithm 
        # register the trading_client's order method with the algorithm
        self.algorithm.set_order(self.order)
        # ask the algorithm to initialize
        self.algorithm.initialize()
    
    def open(self):
        self.result_feed = self.connect_result()
    
    def do_work(self):
        # poll all the sockets
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        # see if the poller has results for the result_feed
        if self.result_feed in socks and \
            socks[self.result_feed] == self.zmq.POLLIN:   
            
            self.last_msg_dt = datetime.datetime.utcnow()
            
            # get the next message from the result feed
            msg = self.result_feed.recv()
            
            # if the feed is done, shut 'er down
            if msg == str(zp.CONTROL_PROTOCOL.DONE):
                self.finish_simulation()
                return
            
            # result_feed is a merge component, so unframe accordingly
            event = zp.MERGE_UNFRAME(msg)
            self.received_count += 1
            # update performance and relay the event to the algorithm
            self.process_event(event)
            if self.perf.exceeded_max_loss:
                self.finish_simulation()
            
    def finish_simulation(self):
        LOGGER.info("Client is DONE!")
        # signal the performance tracker that the simulation has
        # ended. Perf will internally calculate the full risk report.
        self.perf.handle_simulation_end()

        # signal Simulator, our ComponentHost, that this component is
        # done and Simulator needn't block exit on this component.
        self.signal_done()

    def process_event(self, event):
        
        
        # generate transactions, if applicable
        txn = self.txn_sim.apply_trade_to_open_orders(event)
        if txn:
            event.TRANSACTION = txn
            # track the number of transactions, for testing purposes.
            self.txn_count += 1
        else:
            event.TRANSACTION = None
            
        # the performance class needs to process each event, without 
        # skipping. Algorithm should wait until the performance has been 
        # updated, so that down stream components can safely assume that
        # performance is up to date. Note that this is done before we
        # mark the time for the algorithm's processing, thereby not
        # running the algo's clock for performance book keeping.
        self.perf.process_event(event)
        
        # mark the start time for client's processing of this event.
        event_start = datetime.datetime.utcnow()
        
            
        # queue the event.
        self.queue_event(event)
        
        
        # if the event is later than our current time, run the algo
        # otherwise, the algorithm has fallen behind the feed 
        # and processing per event is longer than time between events.
        if event.dt >= self.current_dt:
            # compress time by moving the current_time up to the event
            # time.
            self.current_dt = event.dt
            self.run_algorithm()
        
        # tally the time spent on this iteration
        self.last_iteration_dur = datetime.datetime.utcnow() - event_start
        # move the algorithm's clock forward to include iteration time
        self.current_dt = self.current_dt  + self.last_iteration_dur
    
        
    def run_algorithm(self):
        """
        As per the algorithm protocol: 
        
        - Set the current portfolio for the algorithm as per protocol.
        - Construct frame based on backlog of events, send to algorithm.
        """
        current_portfolio = self.perf.get_portfolio()
        self.algorithm.set_portfolio(current_portfolio)
        frame = self.get_frame()
        if len(frame) > 0:
            self.algorithm.handle_frame(frame)
    
    def connect_order(self):
        return self.connect_push_socket(self.addresses['order_address'])
    
    def order(self, sid, amount):
        order = zp.namedict({
            'dt':self.current_dt,
            'sid':sid,
            'amount':amount
        })
        self.order_count += 1
        self.perf.log_order(order)
        self.txn_sim.add_open_order(order)
        
    def signal_order_done(self):
        self.order_socket.send(str(zp.ORDER_PROTOCOL.DONE))
        
    def queue_event(self, event):
        if self.event_queue == None:
            self.event_queue = []
        series = event.as_series()
        self.event_queue.append(series)
    
    def get_frame(self):
        for event in self.event_queue:
            self.event_frame[event['sid']] = event
        self.event_queue = []
        return self.event_frame
                     

class TransactionSimulator(object):
    
    def __init__(self, style=SIMULATION_STYLE.PARTIAL_VOLUME): 
        self.open_orders                = {}
        self.order_count                = 0
        self.txn_count                  = 0
        self.trade_window               = datetime.timedelta(seconds=30)
        self.orderTTL                   = datetime.timedelta(days=1)
        self.commission                 = 0.03
        
        if not style or style == SIMULATION_STYLE.PARTIAL_VOLUME:
            self.apply_trade_to_open_orders = self.simulate_with_partial_volume
        elif style == SIMULATION_STYLE.BUY_ALL:
            self.apply_trade_to_open_orders =  self.simulate_buy_all
        elif style == SIMULATION_STYLE.FIXED_SLIPPAGE:
            self.apply_trade_to_open_orders = self.simulate_with_fixed_cost
        elif style == SIMULATION_STYLE.NOOP:
            self.apply_trade_to_open_orders = self.simulate_noop
            
    def add_open_order(self, event):
        """Orders are captured in a buffer by sid. No calculations are done here.
            Amount is explicitly converted to an int.
            Orders of amount zero are ignored.
        """
        self.order_count += 1
        
        event.amount = int(event.amount)
        if event.amount == 0:
            log = "requested to trade zero shares of {sid}".format(
                sid=event.sid
            )
            LOGGER.debug(log)
            return
        
        if(not self.open_orders.has_key(event.sid)):
            self.open_orders[event.sid] = []
            
        # set the filled property to zero
        event.filled = 0
        self.open_orders[event.sid].append(event)
     
    def simulate_buy_all(self, event):
        txn = self.create_transaction(
                event.sid, 
                event.volume, 
                event.price, 
                event.dt, 
                1
            )
        return txn
        
    def simulate_noop(self, event):
        return None    
    
    def simulate_with_fixed_cost(self, event):
        if self.open_orders.has_key(event.sid):
            orders = self.open_orders[event.sid] 
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None
            
        amount = 0
        for order in orders:
            amount += order.amount
        
        if(amount == 0):
            return
            
        direction = amount / math.fabs(amount)
        
        
        txn = self.create_transaction(
                event.sid, 
                amount, 
                event.price + 0.10, 
                event.dt, 
                direction
            )
        
        self.open_orders[event.sid] = []
        
        return txn
        
    def simulate_with_partial_volume(self, event):
        if(event.volume == 0):
            #there are zero volume events bc some stocks trade 
            #less frequently than once per minute.
            return None
            
        if self.open_orders.has_key(event.sid):
            orders = self.open_orders[event.sid] 
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None
     
        dt = event.dt
        expired = []
        total_order = 0
        simulated_amount = 0
        simulated_impact = 0.0
        direction = 1.0
        for order in orders:
            
            if(order.dt < event.dt):
                
                # orders are only good on the day they are issued
                if order.dt.day < event.dt.day:
                    continue
    
                open_amount = order.amount - order.filled
                
                if(open_amount != 0):
                    direction = open_amount / math.fabs(open_amount)
                else:
                    direction = 1
                
                desired_order = total_order + open_amount
                
                volume_share = direction * (desired_order) / event.volume
                if volume_share > .25:
                    volume_share = .25
                simulated_amount = int(volume_share * event.volume * direction)
                simulated_impact = (volume_share)**2 * .1 * direction * event.price
                
                order.filled += (simulated_amount - total_order)
                total_order = simulated_amount
                
                # we cap the volume share at 25% of a trade
                if volume_share == .25:
                    break
                  
        orders = [ x for x in orders if abs(x.amount - x.filled) > 0 and x.dt.day >= event.dt.day]
       
        self.open_orders[event.sid] = orders
        
        
        if simulated_amount != 0:
            return self.create_transaction(
                event.sid, 
                simulated_amount, 
                event.price + simulated_impact, 
                dt.replace(tzinfo = pytz.utc), 
                direction
            )
        elif len(orders) > 0:
            warning = """
Calculated a zero volume transaction on trade: 
{event} 
for orders: 
{orders}
            """
            warning = warning.format(
                event=str(event), 
                orders=str(orders)
            )
            LOGGER.warn(warning)
            return None
    
        
    def create_transaction(self, sid, amount, price, dt, direction):   
        self.txn_count += 1             
        txn = {'sid'                : sid, 
                'amount'             : int(amount), 
                'dt'                 : dt, 
                'price'              : price, 
                'commission'          : self.commission * amount * direction,
                'source_id'          : zp.FINANCE_COMPONENT.TRANSACTION_SIM
                }
        return zp.namedict(txn) 
                

class TradingEnvironment(object):

    def __init__(
        self, 
        benchmark_returns, 
        treasury_curves, 
        period_start    = None, 
        period_end      = None, 
        capital_base    = None,
        max_drawdown    = None
    ):
    
        self.trading_days = []
        self.trading_day_map = {}
        self.treasury_curves = treasury_curves
        self.benchmark_returns = benchmark_returns
        self.frame_index = ['sid', 'volume', 'dt', 'price', 'changed']
        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base
        self.period_trading_days = None
        self.max_drawdown = max_drawdown
        
        for bm in benchmark_returns:
            self.trading_days.append(bm.date)
            self.trading_day_map[bm.date] = bm
        
        self.first_open = self.calculate_first_open()
        self.last_close = self.calculate_last_close()

    def calculate_first_open(self):
        """
        Finds the first trading day on or after self.period_start.
        """
        first_open  = self.period_start
        one_day      = datetime.timedelta(days=1)
        
        while not self.is_trading_day(first_open):
            first_open = first_open + one_day

        first_open = self.set_NYSE_time(first_open, 9, 30)
        return first_open
        
    def calculate_last_close(self):
        """
        Finds the last trading day on or before self.period_end
        """
        last_close  = self.period_end
        one_day     = datetime.timedelta(days=1)
        
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
        local = pytz.timezone ('US/Eastern')
        local_dt = naive.replace (tzinfo = local)
        # set the clock to the opening bell in NYC time.
        local_dt = local_dt.replace(hour=hour, minute=minute)
        # convert to UTC
        utc_dt = local_dt.astimezone (pytz.utc)
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
        assert(self.period_start != None)
        assert(self.period_end != None)
                            
        if self.period_trading_days == None:
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
        return self.trading_day_map.has_key(dt)

    def get_benchmark_daily_return(self, test_date):
        date = self.normalize_date(test_date)
        if self.trading_day_map.has_key(date):
            return self.trading_day_map[date].returns
        else:
            return 0.0
            
    def add_to_frame(self, name):
        """
        Add an entry to the frame index. 
        :param name: new index entry name. Used by TradingSimulationClient
        to 
        """
        self.frame_index.append(name)

                

