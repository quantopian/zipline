"""
Algorithm Protocol
===================

For a class to be passed as a trading algorithm to the 
:py:class:`zipline.lines.SimulatedTrading` zipline
it must follow an implementation protocol. Examples of this algorithm protocol
are provided below.

The algorithm must expose methods:

  - initialize: method that takes no args, no returns. Simply called to 
  enable the algorithm to set any internal state needed.

  - get_sid_filter: method that takes no args, and returns a list
  of valid sids. List must have a length between 1 and 10. If None is returned
  the filter will block all events.
  
  - handle_data: method that accepts a :py:class:`zipline.protocol_utils.ndict` 
  of the current state of the simulation universe. An example data ndict::
  
    +-----------------+--------------+----------------+--------------------+
    |                 | SID(133)     |  SID(134)      | SID(135)           |
    +=================+==============+================+====================+
    | price           | $10.10       | $22.50         | $13.37             |
    +-----------------+--------------+----------------+--------------------+
    | volume          | 10,000       | 5,000          | 50,000             |
    +-----------------+--------------+----------------+--------------------+
    | mvg_avg_30      | $9.97        | $22.61         | $13.37             |
    +-----------------+--------------+----------------+--------------------+
    | dt              | 6/30/2012    | 6/30/2011      | 6/29/2012          |
    +-----------------+--------------+----------------+--------------------+
            
  - set_order: method that accepts a callable. Will be set as the value of the 
    order method of trading_client. An algorithm can then place orders with a 
    valid SID and a number of shares::
    
        self.order(SID(133), share_count)
        
  - set_performance: property which can be set equal to the 
    cumulative_trading_performance property of the trading_client. An 
    algorithm can then check position information with the
    Portfolio object::
    
        self.Portfolio[SID(133)]['cost_basis']

"""

import zipline.protocol as zp

class TestAlgorithm():
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """
    
    def __init__(self, sid, amount, order_count):
        self.count = order_count
        self.sid = sid
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
    
    def initialize(self):
        pass
        
    def set_order(self, order_callable):
        self.order = order_callable
        
    def set_portfolio(self, portfolio):
        self.portfolio = portfolio
        
    def handle_data(self, data):
        self.frame_count += 1
        #place an order for 100 shares of sid
        if self.incr < self.count:
            self.order(self.sid, self.amount)
            self.incr += 1
                
    def get_sid_filter(self):
        return [self.sid] 
        
#
class HeavyBuyAlgorithm():
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """
    
    def __init__(self, sid, amount):
        self.sid = sid
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
    
    def initialize(self):
        pass    
    
    def set_order(self, order_callable):
        self.order = order_callable
        
    def set_portfolio(self, portfolio):
        self.portfolio = portfolio
        
    def handle_data(self, data):
        self.frame_count += 1
        #place an order for 100 shares of sid
        self.order(self.sid, self.amount)
        self.incr += 1
                
    def get_sid_filter(self):
        return [self.sid]         
    
class NoopAlgorithm(object):
    """
    Dolce fa niente.
    """
    
    def initialize(self):
        pass
        
    def set_order(self, order_callable):
        pass
    
    def set_portfolio(self, portfolio):
        pass
        
    def handle_data(self, data):
        pass
    
    def get_sid_filter(self):
        return None