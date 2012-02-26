"""
Dummy simulator backported from Qexec for development on Zipline.
"""

import threading
import mock
from unittest2 import TestCase

from zipline.test.test_messaging import SimulatorTestCase
from zipline.monitor import Controller
from zipline.messaging import ComponentHost
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

class SimulatorBase(ComponentHost):
    """
    Simulator coordinates the launch and communication of source, feed, transform, and merge components.
    """

    def __init__(self, addresses, gevent_needed=False):
        """
        """
        ComponentHost.__init__(self, addresses, gevent_needed)

    def simulate(self):
        self.run()

    def get_id(self):
        return "Simulator"

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

class ThreadPoolExecutor(SimulatorTestCase, TestCase):

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
