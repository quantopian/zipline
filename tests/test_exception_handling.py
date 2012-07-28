import zmq

from unittest2 import TestCase
from collections import defaultdict

from zipline.test_algorithms import ExceptionAlgorithm, NoopAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading

from zipline.utils.test_utils import \
        drain_zipline, \
        check, \
        setup_logger, \
        teardown_logger

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
        setup_logger(self)

    def tearDown(self):
        self.ctx.term()
        teardown_logger(self)

    def test_exception_in_init(self):
        # Simulation
        # ----------
        self.zipline_test_config['algorithm'] = \
                ExceptionAlgorithm(
                    'initialize',
                    self.zipline_test_config['sid']
                )

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )
        output, _ = drain_zipline(self, zipline)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']['stack']
        check(self, payload, INITIALIZE_TB)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)


    def test_exception_in_handle_data(self):

        # Simulation
        # ----------
        self.zipline_test_config['algorithm'] = \
                ExceptionAlgorithm(
                    'handle_data',
                    self.zipline_test_config['sid']
                )

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )

        output, _ = drain_zipline(self, zipline)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']['stack']
        check(self, payload, HANDLE_DATA_TB)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)


        # TODO:
        #   - define more zipline failure modes: exception in other
        #   components, exception in Monitor, etc. write tests
        #   for those scenarios.



INITIALIZE_TB =\
[{'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self._run()',
  'lineno': 204,
  'method': 'run'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self.loop()',
  'lineno': 195,
  'method': '_run'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self.do_work()',
  'lineno': 235,
  'method': 'loop'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.initialize_algo()',
  'lineno': 97,
  'method': 'do_work'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.do_op(self.algorithm.initialize)',
  'lineno': 80,
  'method': 'initialize_algo'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'callable_op(*args, **kwargs)',
  'lineno': 206,
  'method': 'do_op'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/test_algorithms.py',
  'line': 'raise Exception("Algo exception in initialize")',
  'lineno': 166,
  'method': 'initialize'}]


HANDLE_DATA_TB =\
[{'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self._run()',
  'lineno': 204,
  'method': 'run'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self.loop()',
  'lineno': 195,
  'method': '_run'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/core/component.py',
  'line': 'self.do_work()',
  'lineno': 235,
  'method': 'loop'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.process_event(event)',
  'lineno': 116,
  'method': 'do_work'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.run_algorithm()',
  'lineno': 164,
  'method': 'process_event'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'self.do_op(self.algorithm.handle_data, data)',
  'lineno': 186,
  'method': 'run_algorithm'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/components/tradesimulation.py',
  'line': 'callable_op(*args, **kwargs)',
  'lineno': 206,
  'method': 'do_op'},
 {'filename': '/Users/fawce/projects/qexec/zipline_repo/zipline/test_algorithms.py',
  'line': 'raise Exception("Algo exception in handle_data")',
  'lineno': 187,
  'method': 'handle_data'}]
