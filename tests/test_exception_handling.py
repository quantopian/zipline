import zmq

from unittest2 import TestCase
from collections import defaultdict
from logbook.compat import LoggingHandler

from zipline.test_algorithms import ExceptionAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading

from zipline.utils.test_utils import drain_zipline

DEFAULT_TIMEOUT = 15 # seconds
EXTENDED_TIMEOUT = 90

allocator = AddressAllocator(1000)


class FinanceTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        self.zipline_test_config = {
            'allocator' : allocator,
            'sid'       : 133,
            'devel'     : False,
            'results_socket'    : allocator.lease(1)[0]
        }
        self.ctx = zmq.Context()

        self.log_handler = LoggingHandler()
        self.log_handler.push_application()

    def tearDown(self):
        self.log_handler.pop_application()

    def test_exception_in_init(self):

        # Simulation
        # ----------

        self.zipline_test_config['simulation_style'] = \
            SIMULATION_STYLE.FIXED_SLIPPAGE
        self.zipline_test_config['algorithm'] = ExceptionAlgorithm('initialize')

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )

        output, _ = drain_zipline(self, zipline)
        self.assertEqual(output, ['EXCEPTION'])
        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)

        # TODO:
        #   - exception protocol to use prefix/payload as EXCEPT,
        #   and the stack trace
        #   - test exception in handle_data
