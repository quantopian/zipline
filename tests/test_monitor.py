from logbook.compat import LoggingHandler
from unittest2 import TestCase, skip

from zipline.core.monitor import Controller

class TestMonitor(TestCase):
    def setUp(self):
        self.log_handler = LoggingHandler()
        self.log_handler.push_application()

    def tearDown(self):
        self.log_handler.pop_application()

    def test_init(self):
        pub_socket   = 'tcp://127.0.0.1:5000'
        route_socket = 'tcp://127.0.0.1:5001'

        con = Controller(pub_socket, route_socket)
        con.manage([])

    def test_init_topology(self):
        pub_socket   = 'tcp://127.0.0.1:5000'
        route_socket = 'tcp://127.0.0.1:5001'

        con = Controller(pub_socket, route_socket, )
        con.manage([ 'a', 'b', 'c', 'd' ])
