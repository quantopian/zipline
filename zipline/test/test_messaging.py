"""
Test suite for the messaging infrastructure of QSim.
"""
#don't worry about excessive public methods pylint: disable=R0904

# TODO: make sure this can run in parallel... right now this is
# forbiddeen because we hardcode the ports but it should totally
# be possible and then we can the suite much faster.
#
# nosetests --processes=5

import zipline.messaging as qmsg

from zipline.transforms.technical import MovingAverage
from zipline.sources import RandomEquityTrades

from zipline.test.client import TestClient


# Should not inherit form TestCase since test runners will pick
# it up as a test.
class SimulatorTestCase(object):

    def get_simulator(self):
        """
        Return the simulator instance to be tested.
        """
        raise NotImplementedError

    def test_sources_only(self):

        sim = self.get_simulator()
        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        client = TestClient(self, expected_msg_count=800)
        sim.register_components([ret1, ret2, client])
        sim.simulate()

        self.assertEqual(sim.feed.pending_messages(), 0,
            "The feed should be drained of all messages, found {n} remaining."
                .format(n=sim.feed.pending_messages()))


    def test_transforms(self):
        sim = self.get_simulator()
        ret1 = RandomEquityTrades(133, "ret1", 5000)
        ret2 = RandomEquityTrades(134, "ret2", 5000)
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        client = TestClient(self, expected_msg_count=10000)
        sim.register_components([ret1, ret2, mavg1, mavg2, client])
        sim.simulate()

        self.assertEqual(sim.feed.pending_messages(), 0, \
                "The feed should be drained of all messages.")

    def dtest_error_in_feed(self):

        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        sources = {"ret1":ret1, "ret2":ret2}
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        transforms = {"mavg1":mavg1, "mavg2":mavg2}
        client = TestClient(self, expected_msg_count=0)
        sim = self.get_simulator(sources, transforms, client)

        # TODO: way too long
        sim.feed = DataFeedErr(sources.keys(), sim.data_address, sim.feed_address, sim.performance_address, qmsg.Sync(sim, "DataFeedErrorGenerator"))
        sim.simulate()


