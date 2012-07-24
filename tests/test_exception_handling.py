from unittest2 import TestCase
from collections import defaultdict
from logbook.compat import LoggingHandler

from zipline.test_algorithms import ExceptionAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading

DEFAULT_TIMEOUT = 15 # seconds
EXTENDED_TIMEOUT = 90

allocator = AddressAllocator(1000)


class FinanceTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        self.zipline_test_config = {
            'allocator' : allocator,
            'sid'       : 133,
            'devel'     : True
        }
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
        self.zipline_test_config['devel'] = False

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )
        zipline.simulate(blocking=True)

        self.assertTrue(zipline.sim.ready())
        self.assertTrue(zipline.sim.exception)
