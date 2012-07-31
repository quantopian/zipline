import zmq

from unittest2 import TestCase
from collections import defaultdict

from zipline.test_algorithms import ExceptionAlgorithm, DivByZeroAlgorithm
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
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
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
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        check(self, payload, HANDLE_DATA_TB)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)

    def test_zerodivision_exception_in_handle_data(self):

        # Simulation
        # ----------
        self.zipline_test_config['algorithm'] = \
                DivByZeroAlgorithm(
                    self.zipline_test_config['sid']
                )

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )

        output, _ = drain_zipline(self, zipline)
        self.assertEqual(len(output), 5)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        check(self, payload, ZERO_DIV_TB)

        self.assertTrue(zipline.sim.ready())
        self.assertFalse(zipline.sim.exception)

        # TODO:
        #   - define more zipline failure modes: exception in other
        #   components, exception in Monitor, etc. write tests
        #   for those scenarios.



INITIALIZE_TB =\
{'message': 'Algo exception in initialize',
 'name': 'Exception',
 'stack': [{'filename': '/zipline/core/component.py', 'line': 'self._run()', 'lineno': 204, 'method': 'run'},
           {'filename': '/zipline/core/component.py', 'line': 'self.loop()', 'lineno': 195, 'method': '_run'},
           {'filename': '/zipline/core/component.py', 'line': 'self.do_work()', 'lineno': 235, 'method': 'loop'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.initialize_algo()',
            'lineno': 97,
            'method': 'do_work'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.do_op(self.algorithm.initialize)',
            'lineno': 80,
            'method': 'initialize_algo'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'callable_op(*args, **kwargs)',
            'lineno': 210,
            'method': 'do_op'},
           {'filename': '/zipline/test_algorithms.py',
            'line': 'raise Exception("Algo exception in initialize")',
            'lineno': 166,
            'method': 'initialize'}]}

HANDLE_DATA_TB =\
{
 'message': 'Algo exception in handle_data',
 'name': 'Exception',
 'stack': [{'filename': '/zipline/core/component.py', 'line': 'self._run()', 'lineno': 204, 'method': 'run'},
           {'filename': '/zipline/core/component.py', 'line': 'self.loop()', 'lineno': 195, 'method': '_run'},
           {'filename': '/zipline/core/component.py', 'line': 'self.do_work()', 'lineno': 235, 'method': 'loop'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.process_event(event)',
            'lineno': 116,
            'method': 'do_work'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.run_algorithm()',
            'lineno': 164,
            'method': 'process_event'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.do_op(self.algorithm.handle_data, data)',
            'lineno': 186,
            'method': 'run_algorithm'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'callable_op(*args, **kwargs)',
            'lineno': 210,
            'method': 'do_op'},
           {'filename': '/zipline/test_algorithms.py',
            'line': 'raise Exception("Algo exception in handle_data")',
            'lineno': 187,
            'method': 'handle_data'}]}


ZERO_DIV_TB= \
{'message': 'integer division or modulo by zero',
 'name': 'ZeroDivisionError',
 'stack': [{'filename': '/zipline/core/component.py', 'line': 'self._run()', 'lineno': 204, 'method': 'run'},
           {'filename': '/zipline/core/component.py', 'line': 'self.loop()', 'lineno': 195, 'method': '_run'},
           {'filename': '/zipline/core/component.py', 'line': 'self.do_work()', 'lineno': 235, 'method': 'loop'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.process_event(event)',
            'lineno': 116,
            'method': 'do_work'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.run_algorithm()',
            'lineno': 164,
            'method': 'process_event'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'self.do_op(self.algorithm.handle_data, data)',
            'lineno': 186,
            'method': 'run_algorithm'},
           {'filename': '/zipline/components/tradesimulation.py',
            'line': 'callable_op(*args, **kwargs)',
            'lineno': 210,
            'method': 'do_op'},
           {'filename': '/zipline/test_algorithms.py', 'line': '5/0', 'lineno': 218, 'method': 'handle_data'}]}
