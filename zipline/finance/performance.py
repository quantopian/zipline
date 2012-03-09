import datetime
import pytz
import math

from zmq.core.poll import select

import zipline.messaging as qmsg
import zipline.util as qutil
import zipline.protocol as zp
import zipline.finance.risk as risk

class PerformanceTracker():
    
    def __init__(self, period_start, period_end, capital_base, trading_environment):
        self.trading_day            = datetime.timedelta(hours=6, minutes=30)
        self.calendar_day           = datetime.timedelta(hours=24)
        self.period_start           = period_start
        self.period_end             = period_end
        self.market_open            = self.period_start 
        self.market_close           = self.market_open + self.trading_day
        self.progress               = 0.0
        self.total_days             = (self.period_end - self.period_start).days
        self.day_count              = 0
        self.cumulative_capital_used= 0.0
        self.max_capital_used       = 0.0
        self.capital_base           = capital_base
        self.trading_environment    = trading_environment
        self.returns                = []
        self.txn_count              = 0 
        self.event_count            = 0
        self.cumulative_performance = PerformancePeriod(
            self.period_start, 
            self.period_end, 
            {}, 
            capital_base, 
            capital_base = capital_base
        )
            
        self.todays_performance     = PerformancePeriod(
            self.market_open, 
            self.market_close, 
            {}, 
            capital_base, 
            capital_base = capital_base
        )
        
        
    
    def update(self, event):
            self.event_count += 1
            if(event.dt >= self.market_close):
                self.handle_market_close()
            
            if event.TRANSACTION != None:                
                self.txn_count += 1
                self.cumulative_performance.execute_transaction(event.TRANSACTION)
                self.todays_performance.execute_transaction(event.TRANSACTION)
                
                #we're adding a 10% cushion to the capital used, and then rounding to the nearest 5k
                self.cumulative_capital_used += event.TRANSACTION.price * event.TRANSACTION.amount
                if(math.fabs(self.cumulative_capital_used) > self.max_capital_used):
                    self.max_capital_used = math.fabs(self.cumulative_capital_used)
                self.max_capital_used = self.round_to_nearest(1.1 * self.max_capital_used, base=5000)
                self.max_leverage = self.max_capital_used/self.capital_base
            
            #update last sale    
            self.cumulative_performance.update_last_sale(event)
            self.todays_performance.update_last_sale(event)
            
            #calculate performance as of last trade
            self.cumulative_performance.calculate_performance()
            self.todays_performance.calculate_performance()
               
    def handle_market_close(self):
        self.market_open = self.market_open + self.calendar_day
        while not self.trading_environment.is_trading_day(self.market_open):
            if self.market_open > self.trading_environment.trading_days[-1]:
                raise Exception("Attempting to backtest beyond available history.")
            self.market_open = self.market_open + self.calendar_day
        self.market_close = self.market_open + self.trading_day   
        self.day_count += 1.0
        self.progress = self.day_count / self.total_days
        #add the return results from today to the list of daily return objects.
        todays_date = self.todays_performance.period_end.replace(hour=0, minute=0, second=0)
        todays_return_obj = risk.daily_return(todays_date, self.todays_performance.returns)
        self.returns.append(todays_return_obj)
        
        #calculate risk metrics for cumulative performance
        self.cur_period_metrics = risk.RiskMetrics(start_date=self.cumulative_performance.period_start, 
                                                    end_date=self.cumulative_performance.period_end.replace(hour=0, minute=0, second=0), 
                                                    returns=self.returns,
                                                    trading_environment=self.trading_environment)
                                                    
        ######################################################################################################
        #######TODO: report/relay metrics out to qexec -- values come from self.cur_period_metrics ###########
        #######TODO: report/relay position data out to qexec -- values come from self.cumulative_performance #
        ######################################################################################################
        
        #roll over positions to current day.
        self.todays_performance = PerformancePeriod(self.market_open, 
                                                    self.market_close, 
                                                    self.todays_performance.positions, 
                                                    self.todays_performance.ending_value, 
                                                    self.capital_base)
        
    def handle_simulation_end(self):
        self.risk_report = risk.RiskReport(self.returns, self.trading_environment)
        ######################################################################################################
        #######TODO: report/relay metrics out to qexec -- values come from self.risk_report        ###########
        ######################################################################################################
    
    def round_to_nearest(self, x, base=5):
        return int(base * round(float(x)/base))

class Position():
    sid         = None
    amount      = None
    cost_basis   = None
    last_sale    = None
    last_date    = None
    
    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0 ##per share
    
    def update(self, txn):
        if(self.sid != txn.sid):
            raise NameError('attempt to update position with transaction in different sid')
            #throw exception
        
        if(self.amount + txn.amount == 0): #we're covering a short or closing a position
            self.cost_basis = 0.0
            self.amount = 0
        else:
            self.cost_basis = (self.cost_basis*self.amount + (txn.amount*txn.price))/(self.amount + txn.amount)
            self.amount = self.amount + txn.amount
            
    def currentValue(self):
        return self.amount * self.last_sale
        
        
    def __repr__(self):
        return "sid: {sid}, amount: {amount}, cost_basis: {cost_basis}, last_sale: {last_sale}".format(
        sid=self.sid, amount=self.amount, cost_basis=self.cost_basis, last_sale=self.last_sale)
        
class PerformancePeriod():
    
    def __init__(self, period_start, period_end, initial_positions, initial_value, capital_base = None):
        self.ending_value        = 0.0
        self.period_capital_used  = 0.0
        self.period_start       = period_start
        self.period_end         = period_end
        self.positions          = initial_positions #sid => position object
        self.starting_value      = initial_value
        if(capital_base != None):
            self.capital_base   = capital_base
        else:
            self.capital_base   = 0
            
    def calculate_performance(self):
        self.ending_value = self.calculate_positions_value()
        self.pnl = (self.ending_value - self.starting_value) - self.period_capital_used
        if(self.capital_base != 0):
            self.returns = self.pnl / self.starting_value
        else:
            self.returns = 0.0
            
    def execute_transaction(self, txn):
        if(txn.dt > self.period_end):
            raise Exception("transaction dated {dt} attempted for period ending {ending}".
                            format(dt=txn.dt, ending=self.period_end))
        if(not self.positions.has_key(txn.sid)):
            self.positions[txn.sid] = Position(txn.sid)
        self.positions[txn.sid].update(txn)
        self.period_capital_used += -1 * txn.price * txn.amount
        

    def calculate_positions_value(self):
        mktValue = 0.0
        for key,pos in self.positions.iteritems():
            mktValue += pos.currentValue()
        return mktValue
                
    def update_last_sale(self, event):
        if self.positions.has_key(event.sid):
            self.positions[event.sid].last_sale = event.price 
            self.positions[event.sid].last_date = event.dt
        
        

    