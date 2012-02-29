"""
Test suite for the messaging infrastructure of Zipline.
"""
#don't worry about excessive public methods pylint: disable=R0904

import zipline.messaging as qmsg

from zipline.transforms.technical import MovingAverage
from zipline.sources import RandomEquityTrades
from zipline.test.dummy import ThreadPoolExecutorMixin
from zipline.test.client import TestClient

# Should not inherit form TestCase since test runners will pick
# it up as a test. Its a Mixin of sorts at this point.
class SimulatorTestCase(ThreadPoolExecutorMixin):
    
    # -------
    #  Cases
    # -------

    def test_simple(self):

        # Simple test just to make sure that the archiecture is
        # responding.

        # Base Simuation
        # --------------

        # Allocate sockets for the simulator components
        sockets = self.allocate_sockets(5)

        addresses = {
            'sync_address'   : sockets[0],
            'data_address'   : sockets[1],
            'feed_address'   : sockets[2],
            'merge_address'  : sockets[3],
            'result_address' : sockets[4]
        }

        sim = self.get_simulator(addresses)
        con = self.get_controller()

        # Simulation Components
        # ---------------------

        ret1 = RandomEquityTrades(133, "ret1", 1)
        ret2 = RandomEquityTrades(134, "ret2", 1)
        client = TestClient(expected_msg_count=(ret1.count + ret2.count))

        sim.register_controller( con )
        sim.register_components([ret1, ret2, client])

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


    def test_sources_only(self):

        # Base Simuation
        # --------------

        # Allocate sockets for the simulator components
        sockets = self.allocate_sockets(5)

        addresses = {
            'sync_address'   : sockets[0],
            'data_address'   : sockets[1],
            'feed_address'   : sockets[2],
            'merge_address'  : sockets[3],
            'result_address' : sockets[4]
        }

        sim = self.get_simulator(addresses)
        con = self.get_controller()

        # Simulation Components
        # ---------------------

        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        client = TestClient(expected_msg_count=ret1.count + ret2.count)

        sim.register_controller( con )
        sim.register_components([ret1, ret2, client])

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

    def test_transforms(self):

        # Base Simuation
        # --------------

        # Allocate sockets for the simulator components
        sockets = self.allocate_sockets(5)

        addresses = {
            'sync_address'   : sockets[0],
            'data_address'   : sockets[1],
            'feed_address'   : sockets[2],
            'merge_address'  : sockets[3],
            'result_address' : sockets[4]
        }

        sim = self.get_simulator(addresses)
        con = self.get_controller()

        # Simulation Components
        # ---------------------

        ret1 = RandomEquityTrades(133, "ret1", 5000)
        ret2 = RandomEquityTrades(134, "ret2", 5000)
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        client = TestClient(expected_msg_count=10000)

        sim.register_components([ret1, ret2, mavg1, mavg2, client])
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

    # TODO used?
    def dtest_error_in_feed(self):

        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        sources = {"ret1":ret1, "ret2":ret2}
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        transforms = {"mavg1":mavg1, "mavg2":mavg2}
        client = TestClient(expected_msg_count=0)
        sim = self.get_simulator(sources, transforms, client)

        # TODO: way too long
        sim.feed = DataFeedErr(sources.keys(), sim.data_address, sim.feed_address, sim.performance_address, qmsg.Sync(sim, "DataFeedErrorGenerator"))
        sim.simulate()


