from zipline.utils.test_utils import setup_logger, teardown_logger
from unittest2 import TestCase, skip

from zipline.core.monitor import Monitor


class TestMonitor(TestCase):
    def setUp(self):
        setup_logger(self, '/var/log/qexec/qexec.log')

    def tearDown(self):
        teardown_logger(self)

    def test_init(self):
        pub_socket = 'tcp://127.0.0.1:5000'
        route_socket = 'tcp://127.0.0.1:5001'
        exception_socket = 'tcp://127.0.0.1:5002'

        mon = Monitor(pub_socket, route_socket, exception_socket)
        mon.manage([])

    def test_init_topology(self):
        pub_socket = 'tcp://127.0.0.1:5000'
        route_socket = 'tcp://127.0.0.1:5001'
        exception_socket = 'tcp://127.0.0.1:5002'

        mon = Monitor(pub_socket, route_socket, exception_socket)
        mon.manage(['a', 'b', 'c', 'd'])
