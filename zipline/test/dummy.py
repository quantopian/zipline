"""
Dummy simulator for test/development on Zipline.
"""

import threading
import mock
from collections import defaultdict
from zipline.monitor import Controller
from zipline.messaging import SimulatorBase
import zipline.util as qutil

class DummyAllocator(object):

    def __init__(self, ns):
        self.idx = 0
        self.sockets = [
            'tcp://127.0.0.1:%s' % (10000 + n)
            for n in xrange(ns)
        ]

    def lease(self, n):
        sockets = self.sockets[self.idx:self.idx+n]
        self.idx += n
        return sockets

    def reaquire(self, *conn):
        pass


class ThreadSimulator(SimulatorBase):

    def __init__(self, addresses):
        SimulatorBase.__init__(self, addresses)

    def launch_controller(self):
        thread = threading.Thread(target=self.controller.run)
        thread.start()
        self.cuc = thread
        return thread

    def launch_component(self, component):
        thread = threading.Thread(target=component.run)
        thread.start()
        return thread
        
class ExecutorMixinBase(object):
    """Abstract base to allow mixin for tests that need a dummy simulator."""
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

class ThreadPoolExecutorMixin(ExecutorMixinBase):
    """Dummy server using threads."""
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



