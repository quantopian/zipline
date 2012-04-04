"""Tests for the zipline.finance package"""
import mock
import pytz

from unittest2 import TestCase
from datetime import datetime, timedelta
from collections import defaultdict

from nose.tools import timed

import zipline.test.factory as factory
import zipline.util as qutil
import zipline.finance.risk as risk
import zipline.protocol as zp
import zipline.finance.performance as perf

from zipline.test.algorithms import TestAlgorithm
from zipline.sources import SpecificEquityTrades
from zipline.finance.trading import TransactionSimulator, OrderDataSource, \
TradeSimulationClient, TradingEnvironment
from zipline.simulator import AddressAllocator, Simulator
from zipline.monitor import Controller
from zipline.lines import SimulatedTrading

DEFAULT_TIMEOUT = 15 # seconds

allocator = AddressAllocator(1000)

class FinanceTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        qutil.configure_logging()
        self.zipline_test_config = {
            'allocator':allocator,
            'sid':133
        }

    @timed(DEFAULT_TIMEOUT)
    def test_factory(self):
        trading_environment = factory.create_trading_environment()
        trade_source = factory.create_daily_trade_source(
            [133],
            200,
            trading_environment
        )
        prev = None
        for trade in trade_source.event_list:
            if prev:
                self.assertTrue(trade.dt > prev.dt)
            prev = trade
        
    @timed(DEFAULT_TIMEOUT)
    def test_orders(self):
        
        # Simulation
        # ----------
        zipline = SimulatedTrading.create_test_zipline(**self.zipline_test_config)
        zipline.simulate(blocking=True)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)

        # TODO: Make more assertions about the final state of the components.
        self.assertEqual(zipline.sim.feed.pending_messages(), 0, \
            "The feed should be drained of all messages, found {n} remaining." \
            .format(n=zipline.sim.feed.pending_messages()))


    @timed(DEFAULT_TIMEOUT)
    def test_performance(self): 
        #provide enough trades to ensure all orders are filled.
        self.zipline_test_config['order_count'] = 100
        self.zipline_test_config['trade_count'] = 200
        zipline = SimulatedTrading.create_test_zipline(**self.zipline_test_config)
        zipline.simulate(blocking=True)

        self.assertEqual(
            zipline.sim.feed.pending_messages(), 
            0, 
            "The feed should be drained of all messages, found {n} remaining." \
            .format(n=zipline.sim.feed.pending_messages())
        )
        
        self.assertEqual(
            zipline.sim.merge.pending_messages(), 
            0, 
            "The merge should be drained of all messages, found {n} remaining." \
            .format(n=zipline.sim.merge.pending_messages())
        )

        self.assertEqual(
            zipline.algorithm.count,
            zipline.algorithm.incr,
            "The test algorithm should send as many orders as specified.")
            
        order_source = zipline.sources[zp.FINANCE_COMPONENT.ORDER_SOURCE]
        self.assertEqual(
            order_source.sent_count, 
            zipline.algorithm.count, 
            "The order source should have sent as many orders as the algo."
        )
        
        transaction_sim = zipline.transforms[zp.TRANSFORM_TYPE.TRANSACTION]
        self.assertEqual(
            transaction_sim.txn_count,
            zipline.trading_client.perf.txn_count,
            "The perf tracker should handle the same number of transactions \
            as the simulator emits."
        ) 
        
        self.assertEqual(
            len(zipline.get_positions()), 
            1, 
            "Portfolio should have one position."
        )
        
        SID = self.zipline_test_config['sid']
        self.assertEqual(
            zipline.get_positions()[SID]['sid'], 
            SID, 
            "Portfolio should have one position in " + str(SID)
        )
        
        self.assertEqual(
            zipline.sources['flat'].count,
            self.zipline_test_config['trade_count'],
            "The simulated trade source should send all trades."
        )
        
        self.assertEqual(
            zipline.algorithm.frame_count,
            self.zipline_test_config['trade_count'],
            "The algorithm should receive all trades."
            )
    
    @timed(DEFAULT_TIMEOUT)  
    def test_sid_filter(self):
        """Ensure the algorithm's filter prevents events from arriving."""
        # create a test algorithm whose filter will not match any of the
        # trade events sourced inside the zipline.
        order_amount = 100
        order_count = 100
        no_match_sid = 222
        test_algo = TestAlgorithm(
            no_match_sid,
            order_amount,
            order_count
        )
        
        self.zipline_test_config['trade_count'] = 200
        self.zipline_test_config['algorithm'] = test_algo
        
        zipline = SimulatedTrading.create_test_zipline(**self.zipline_test_config)
       
        zipline.simulate(blocking=True)
        #check that the algorithm received no events
        self.assertEqual(
            0,
            test_algo.frame_count,
            "The algorithm should not receive any events due to filtering."
        )






