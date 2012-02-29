"""
Test suite for the messaging infrastructure of QSim.
"""
#don't worry about excessive public methods pylint: disable=R0904

from collections import defaultdict

from zipline.transforms.technical import MovingAverage
from zipline.sources import RandomEquityTrades

from zipline.test.client import TestClient
from zipline.test.transform import DivideByZeroTransform


# Should not inherit form TestCase since test runners will pick
# it up as a test. Its a Mixin of sorts at this point.
class SimulatorTestCase(object):

    leased_sockets = defaultdict(list)

    def setUp(self):
        self.setup_logging()

        # TODO: how to make Nose use this cross-process????
        self.setup_allocator()

    def tearDown(self):
        pass
        #self.unallocate_sockets()

        # Assert the sockets were properly cleaned up
        #self.assertEmpty(self.leased_sockets[self.id()].values())

        # Assert they were returned to the heap
        #self.allocator.socketheap.assert

    def get_simulator(self):
        """
        Return a new simulator instance to be tested.
        """
        raise NotImplementedError

    def get_controller(self):
        """
        Return a new controler for simulator instance to be tested.
        """
        raise NotImplementedError

    def setup_allocator(self):
        """
        Setup the socket allocator for this test case.
        """
        raise NotImplementedError

    def allocate_sockets(self, n):
        """
        Allocate sockets local to this test case, track them so
        we can gc after test run.
        """

        assert isinstance(n, int)
        assert n > 0

        leased = self.allocator.lease(n)

        self.leased_sockets[self.id()].extend(leased)
        return leased

    def unallocate_sockets(self):
        self.allocator.reaquire(*self.leased_sockets[self.id()])

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
        client = TestClient(self, expected_msg_count=ret1.count + ret2.count)

        sim.register_controller( con )
        sim.register_components([ret1, ret2, client])

        # Simulation
        # ----------
        sim_context = sim.simulate()
        sim_context.join()

        # Stop Running
        # ------------

        self.assertTrue(sim.ready())
        self.assertFalse(sim.exception)

        self.assertEqual(sim.feed.pending_messages(), 0,
            "The feed should be drained of all messages, found {n} remaining."
            .format(n=sim.feed.pending_messages())
        )

    def test_simplefail(self):

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
        fail_transform = DivideByZeroTransform("fail")
        client = TestClient(self, expected_msg_count=ret1.count + ret2.count)

        sim.register_controller( con )
        sim.register_components([ret1, ret2, fail_transform, client])

        # Simulation
        # ----------
        sim.simulate()

        # Stop Running
        # ------------

        self.assertTrue(fail_transform.exception)
        self.assertFalse(fail_transform.successful())

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
        client = TestClient(self, expected_msg_count=ret1.count + ret2.count)

        sim.register_controller( con )
        sim.register_components([ret1, ret2, client])

        # Simulation
        # ----------
        sim.simulate()

        # Stop Running
        # ------------
        self.assertTrue(sim.ready())
        self.assertFalse(sim.exception)

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
        client = TestClient(self, expected_msg_count=10000)

        sim.register_components([ret1, ret2, mavg1, mavg2, client])
        sim.register_controller( con )

        # Simulation
        # ----------
        sim.simulate()

        # Stop Running
        # ------------
        self.assertTrue(sim.ready())
        self.assertFalse(sim.exception)

        self.assertEqual(sim.feed.pending_messages(), 0,
            "The feed should be drained of all messages, found {n} remaining."
            .format(n=sim.feed.pending_messages())
        )
