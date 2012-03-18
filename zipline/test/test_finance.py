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

from zipline.test.client import TestAlgorithm
from zipline.sources import SpecificEquityTrades
from zipline.finance.trading import TransactionSimulator, OrderDataSource, \
TradeSimulationClient
from zipline.simulator import AddressAllocator, Simulator
from zipline.monitor import Controller
from zipline.lines import SimulatedTrading

DEFAULT_TIMEOUT = 5 # seconds

allocator = AddressAllocator(1000)

class FinanceTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        qutil.configure_logging()
        self.benchmark_returns, self.treasury_curves = \
        factory.load_market_data()

        start = datetime.strptime("01/1/2006","%m/%d/%Y")
        start = start.replace(tzinfo=pytz.utc)
        self.trading_environment = risk.TradingEnvironment(
            self.benchmark_returns,
            self.treasury_curves,
            period_start = start,
            capital_base = 100000.0
        )

        self.allocator = allocator

    @timed(DEFAULT_TIMEOUT)
    def test_trade_feed_protocol(self):

        sid = 133
        price = [10.0] * 4
        volume = [100] * 4

        start_date = datetime.strptime("02/15/2012","%m/%d/%Y")
        one_day_td = timedelta(days=1)

        trades = factory.create_trade_history(
            sid, 
            price, 
            volume, 
            start_date, 
            one_day_td, 
            self.trading_environment
        )

        for trade in trades:
            #simulate data source sending frame
            msg = zp.DATASOURCE_FRAME(zp.namedict(trade))
            #feed unpacking frame
            recovered_trade = zp.DATASOURCE_UNFRAME(msg)
            #feed sending frame
            feed_msg = zp.FEED_FRAME(recovered_trade)
            #transform unframing
            recovered_feed = zp.FEED_UNFRAME(feed_msg)
            #do a transform
            trans_msg = zp.TRANSFORM_FRAME('helloworld', 2345.6)
            #simulate passthrough transform -- passthrough shouldn't even
            # unpack the msg, just resend.

            passthrough_msg = zp.TRANSFORM_FRAME(zp.TRANSFORM_TYPE.PASSTHROUGH,\
                    feed_msg)

            #merge unframes transform and passthrough
            trans_recovered = zp.TRANSFORM_UNFRAME(trans_msg)
            pt_recovered = zp.TRANSFORM_UNFRAME(passthrough_msg)
            #simulated merge
            pt_recovered.PASSTHROUGH.merge(trans_recovered)
            #frame the merged event
            merged_msg = zp.MERGE_FRAME(pt_recovered.PASSTHROUGH)
            #unframe the merge and validate values
            event = zp.MERGE_UNFRAME(merged_msg)

            #check the transformed value, should only be in event, not trade.
            self.assertTrue(event.helloworld == 2345.6)
            event.delete('helloworld')

            self.assertEqual(zp.namedict(trade), event)

    @timed(DEFAULT_TIMEOUT)
    def test_order_protocol(self):
        #client places an order
        now = datetime.utcnow().replace(tzinfo=pytz.utc)
        order = zp.namedict({
            'dt':now,
            'sid':133,
            'amount':100
        })
        order_msg = zp.ORDER_FRAME(order)

        #order datasource receives
        order = zp.ORDER_UNFRAME(order_msg)
        self.assertEqual(order.sid, 133)
        self.assertEqual(order.amount, 100)
        self.assertEqual(order.dt, now)
        
        #order datasource datasource frames the order
        order_event = zp.namedict({
            "sid"        : order.sid,
            "amount"     : order.amount,
            "dt"         : order.dt,
            "source_id"  : zp.FINANCE_COMPONENT.ORDER_SOURCE,
            "type"       : zp.DATASOURCE_TYPE.ORDER
        })


        order_ds_msg = zp.DATASOURCE_FRAME(order_event)

        #transaction transform unframes
        recovered_order = zp.DATASOURCE_UNFRAME(order_ds_msg)

        self.assertEqual(now, recovered_order.dt)

        #create a transaction from the order
        txn = zp.namedict({
            'sid'        : recovered_order.sid,
            'amount'     : recovered_order.amount,
            'dt'         : recovered_order.dt,
            'price'      : 10.0,
            'commission' : 0.50
        })

        #frame that transaction
        txn_msg = zp.TRANSFORM_FRAME(zp.TRANSFORM_TYPE.TRANSACTION, txn)

        #unframe
        recovered_tx = zp.TRANSFORM_UNFRAME(txn_msg).TRANSACTION
        self.assertEqual(recovered_tx.sid, 133)
        self.assertEqual(recovered_tx.amount, 100)

    @timed(DEFAULT_TIMEOUT)
    def test_orders(self):

        # Just verify sending and receiving orders.
        # --------------
        #
        SID=133
        sids = [133]
        trade_count = 100
        trade_source = factory.create_daily_trade_source(
            sids,
            trade_count,
            self.trading_environment
        )

        # Simulation
        # ----------
        zipline = SimulatedTrading(
                self.trading_environment,
                self.allocator
        )
        zipline.add_source(trade_source)
        
        order_amount = 100
        order_count = 10
        test_algo = TestAlgorithm(
            SID, 
            order_amount, 
            order_count, 
            zipline.trading_client
        )
        
        zipline.simulate(blocking=True)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)

        # TODO: Make more assertions about the final state of the components.
        self.assertEqual(zipline.sim.feed.pending_messages(), 0, \
            "The feed should be drained of all messages, found {n} remaining." \
            .format(n=zipline.sim.feed.pending_messages()))


    @timed(DEFAULT_TIMEOUT)
    def test_performance(self): 

        # verify order -> transaction -> portfolio position.
        # -------------- 
        SID=133
        sids = [133]
        trade_count = 100
        trade_source = factory.create_daily_trade_source(
            sids,
            trade_count,
            self.trading_environment
        )
       
        # Simulation
        # ----------
        zipline = SimulatedTrading(
                self.trading_environment,
                self.allocator
        )
        zipline.add_source(trade_source)
        
        order_amount = 100
        order_count = 25
        test_algo = TestAlgorithm(
            SID, 
            order_amount, 
            order_count, 
            zipline.trading_client
        )
        
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
            test_algo.count,
            test_algo.incr,
            "The test algorithm should send as many orders as specified.")
            
        order_source = zipline.sources[zp.FINANCE_COMPONENT.ORDER_SOURCE]
        self.assertEqual(
            order_source.sent_count, 
            test_algo.count, 
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
        
        self.assertEqual(
            zipline.get_positions()[SID]['sid'], 
            SID, 
            "Portfolio should have one position in " + str(SID)
        )
