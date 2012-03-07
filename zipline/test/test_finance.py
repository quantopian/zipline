"""Tests for the zipline.finance package"""
import mock
import pytz
from unittest2 import TestCase
from datetime import datetime, timedelta

import zipline.test.factory as factory
import zipline.util as qutil
import zipline.finance.risk as risk
import zipline.protocol as zp

from zipline.test.client import TestTradingClient
from zipline.sources import SpecificEquityTrades
from zipline.finance.trading import TransactionSimulator, OrderDataSource
from zipline.simulator import AddressAllocator, Simulator
from zipline.monitor import Controller


class FinanceTestCase(TestCase):

    def setUp(self):
        qutil.configure_logging()
        self.benchmark_returns, self.treasury_curves = factory.load_market_data()
        self.trading_calendar = risk.TradingCalendar(self.benchmark_returns, self.treasury_curves)
        

    def test_trade_feed_protocol(self):

        # TODO: Perhaps something more self-documenting for variables names?
        sid = 133
        price = [10.0] * 4
        volume = [100] * 4

        start_date = datetime.strptime("02/15/2012","%m/%d/%Y")
        one_day_td = timedelta(days=1)

        trades = factory.create_trade_history(sid, price, volume, start_date, one_day_td, self.trading_calendar)

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

    def test_order_protocol(self):
        #client places an order
        order_msg = zp.ORDER_FRAME(133, 100)

        #order datasource receives
        sid, amount = zp.ORDER_UNFRAME(order_msg)
        self.assertEqual(sid, 133)
        self.assertEqual(amount, 100)

        #order datasource datasource frames the order
        order_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
        order_event = zp.namedict({
            "sid"        : sid,
            "amount"     : amount,
            "dt"         : order_dt,
            "source_id"  : zp.FINANCE_COMPONENT.ORDER_SOURCE,
            "type"       : zp.DATASOURCE_TYPE.ORDER
        })


        order_ds_msg = zp.DATASOURCE_FRAME(order_event)

        #transaction transform unframes
        recovered_order = zp.DATASOURCE_UNFRAME(order_ds_msg)

        self.assertEqual(order_dt, recovered_order.dt)

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

    def test_orders(self):

        # Just verify sending and receiving orders.
        # --------------

        # Allocate sockets for the simulator components
        allocator = AddressAllocator(8)
        sockets = allocator.lease(8)

        addresses = {
            'sync_address'   : sockets[0],
            'data_address'   : sockets[1],
            'feed_address'   : sockets[2],
            'merge_address'  : sockets[3],
            'result_address' : sockets[4],
            'order_address'  : sockets[5]
        }

        con = Controller(
            sockets[6],
            sockets[7],
            logging = qutil.LOGGER
        )

        sim = Simulator(addresses)

        # Simulation Components
        # ---------------------

        # TODO: Perhaps something more self-documenting for variables names?
        sid = 133
        price = [10.1] * 16
        volume = [100] * 16
        start_date = datetime.strptime("02/1/2012","%m/%d/%Y")
        trade_time_increment = timedelta(days=1)

        trade_history = factory.create_trade_history( sid, price, volume, start_date, trade_time_increment, self.trading_calendar )

        set1 = SpecificEquityTrades("flat-133", trade_history)

        #client sill send 10 orders for 100 shares of 133
        client = TestTradingClient(133, 100, 10)
        ts = datetime.strptime("02/1/2012","%m/%d/%Y").replace(tzinfo=pytz.utc)

        order_source = OrderDataSource(ts)
        transaction_sim = TransactionSimulator()

        sim.register_components([client, order_source, transaction_sim, set1])
        sim.register_controller( con )

        # Simulation
        # ----------
        sim_context = sim.simulate()
        sim_context.join()


        # TODO: Make more assertions about the final state of the components.
        self.assertEqual(sim.feed.pending_messages(), 0, \
            "The feed should be drained of all messages, found {n} remaining." \
            .format(n=sim.feed.pending_messages()))
