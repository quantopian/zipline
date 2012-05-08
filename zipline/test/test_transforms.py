from datetime import timedelta
from collections import defaultdict
from unittest2 import TestCase

import zipline.test.factory as factory
import zipline.util as qutil
from zipline.finance.vwap import DailyVWAP, VWAPTransform, DailyVWAP_df
from zipline.finance.returns import ReturnsFromPriorClose
from zipline.finance.movingaverage import MovingAverage
from zipline.lines import SimulatedTrading
from zipline.simulator import AddressAllocator, Simulator


allocator = AddressAllocator(1000)

class ZiplineWithTransformsTestCase(TestCase):
    leased_sockets = defaultdict(list)
    
    def setUp(self):
        qutil.configure_logging()
        self.trading_environment = factory.create_trading_environment()
        self.zipline_test_config = {
            'allocator':allocator,
            'sid':133
        }
        
    def test_vwap_tnfm(self):
        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )
        
        vwap = VWAPTransform("vwap_10", daycount=10)
        zipline.add_transform(vwap)
        
        zipline.simulate(blocking=True)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)
        
class FinanceTransformsTestCase(TestCase):
    def setUp(self):
        self.trading_environment = factory.create_trading_environment()
        
    def test_vwap(self):
        
        trade_history = factory.create_trade_history(
            133, 
            [10.0, 10.0, 10.0, 11.0], 
            [100, 100, 100, 300], 
            timedelta(days=1), 
            self.trading_environment
        )
        
        vwap = DailyVWAP(daycount=2)
        for trade in trade_history:
            vwap.update(trade)
        
        self.assertEqual(vwap.vwap, 10.75)
        
    
    def test_returns(self):
        trade_history = factory.create_trade_history(
            133, 
            [10.0, 10.0, 10.0, 11.0], 
            [100, 100, 100, 300], 
            timedelta(days=1), 
            self.trading_environment
        )
        
        returns = ReturnsFromPriorClose()
        for trade in trade_history:
            returns.update(trade)
            
        
        self.assertEqual(returns.returns, .1)
        
    
    def test_moving_average(self):
        trade_history = factory.create_trade_history(
            133, 
            [10.0, 10.0, 10.0, 11.0], 
            [100, 100, 100, 300], 
            timedelta(days=1), 
            self.trading_environment
        )
        
        ma = MovingAverage(daycount=2)
        for trade in trade_history:
            ma.update(trade)
            
        
        self.assertEqual(ma.average, 10.5)
        
        
    