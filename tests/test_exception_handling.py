import zmq

from unittest2 import TestCase
from collections import defaultdict
from logbook.compat import LoggingHandler

from zipline.test_algorithms import ExceptionAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading

from zipline.utils.test_utils import drain_zipline, check

DEFAULT_TIMEOUT = 15 # seconds
EXTENDED_TIMEOUT = 90

allocator = AddressAllocator(1000)


class ExceptionTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        self.zipline_test_config = {
            'allocator' : allocator,
            'sid'       : 133,
            'devel'     : False,
            'results_socket'    : allocator.lease(1)[0],
            'simulation_style'  : SIMULATION_STYLE.FIXED_SLIPPAGE
        }
        self.ctx = zmq.Context()

        self.log_handler = LoggingHandler()
        self.log_handler.push_application()

    def tearDown(self):
        self.log_handler.pop_application()

    def test_exception_in_init(self):

        # Simulation
        # ----------

        self.zipline_test_config['algorithm'] = \
                ExceptionAlgorithm('initialize')

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )

        output, _ = drain_zipline(self, zipline)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        check(self, payload, INITIALIZE_STACK_TB)

        import nose.tools; nose.tools.set_trace()
        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)

        # TODO:
        #   - exception protocol to use prefix/payload as EXCEPT,
        #   and the stack trace
        #   - test exception in handle_data
        #   - define more zipline failure modes: exception in other
        #   components, exception in Monitor, etc. write tests
        #   for those scenarios.



INITIALIZE_STACK_TB =\
[{'file': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self._run()',
  'lineno': 229,
  'method': 'run'},
 {'file': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self.open()',
  'lineno': 208,
  'method': '_run'},
 {'file': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.initialize_algo()',
  'lineno': 73,
  'method': 'open'},
 {'file': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.do_op(self.algorithm.initialize)',
  'lineno': 83,
  'method': 'initialize_algo'},
 {'file': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'callable_op(*args, **kwargs)',
  'lineno': 205,
  'method': 'do_op'},
 {'file': '/Users/fawce/projects/qexec/zipline_repo/zipline/test_algorithms.py',
  'line': 'raise Exception("Algo exception in initialize")',
  'lineno': 161,
  'method': 'initialize'}]
