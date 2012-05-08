from datetime import timedelta

from unittest2 import TestCase
import zipline.test.factory as factory
from zipline.finance.transforms.vwap import DailyVWAP, DailyVWAP_df
from zipline.finance.transforms.returns import ReturnsFromPriorClose
from zipline.finance.transforms.moving_average import MovingAverage


class FinanceTestCase(TestCase):
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