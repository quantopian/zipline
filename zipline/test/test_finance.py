"""Tests for the zipline.finance package"""
import mock
import zipline.host_settings
from unittest2 import TestCase
from zipline.test.test_devsimulator import ThreadSimulator, DummyAllocator
from zipline.test.test_messaging import SimulatorTestCase
import zipline.test.factory as factory
from zipline.monitor import Controller
from zipline.messaging import DataSource
import zipline.util as qutil
import zipline.db as db
import zipline.host_settings

class FinanceTestCase(SimulatorTestCase, TestCase):

    allocator = DummyAllocator(100)

    def setup_logging(self):
        qutil.configure_logging()

        # lazy import by design
        self.logger = mock.Mock()

    def setup_allocator(self):
        pass

    def get_simulator(self, addresses):
        return ThreadSimulator(addresses)

    def get_controller(self):
        # Allocate two more sockets
        controller_sockets = self.allocate_sockets(2)

        return Controller(
            controller_sockets[0],
            controller_sockets[1],
            logging = self.logger,
        )

    #
    def test_orders(self):

        # Base Simuation
        # --------------

        # Allocate sockets for the simulator components
        sockets = self.allocate_sockets(6)

        addresses = {
            'sync_address'   : sockets[0],
            'data_address'   : sockets[1],
            'feed_address'   : sockets[2],
            'merge_address'  : sockets[3],
            'result_address' : sockets[4],
            'order_address'  : sockets[5]
        }

        sim = self.get_simulator(addresses)
        con = self.get_controller()

        # Simulation Components
        # ---------------------

        set1 = SpecificEquityTrades("flat-133",factory.create_trade_history(133,    
                                                                            [10.0,10.0,10.0,10.0], 
                                                                            [100,100,100,100], 
                                                                            datetime.datetime.utcnow(), 
                                                                            datetime.timedelta(days=1)))
        client = TestTradingClient(self, expected_msg_count=4)

        sim.register_components([set1, client])
        sim.register_controller( con )

        # Simulation
        # ----------
        sim.simulate()

        # Stop Running
        # ------------

        # TODO: less abrupt later, just shove a StopIteration
        # down the pipe to make it stop spinning
        sim.cuc._Thread__stop()

        self.assertEqual(sim.feed.pending_messages(), 0,
            "The feed should be drained of all messages, found {n} remaining."
            .format(n=sim.feed.pending_messages())
        )
        