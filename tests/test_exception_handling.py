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
            'allocator'             : allocator,
            'sid'                   : 133,
            'devel'                 : False,
            'results_socket_uri'    : allocator.lease(1)[0],
            'simulation_style'      : SIMULATION_STYLE.FIXED_SLIPPAGE
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
        self.assertEqual(len(output), 2)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        check(self, payload, INITIALIZE_TB)

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
        self.assertEqual(len(output), 3)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        check(self, payload, HANDLE_DATA_TB)

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
        self.assertEqual(len(output), 6)
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        check(self, payload, ZERO_DIV_TB)

        # TODO:
        #   - define more zipline failure modes: exception in other
        #   components, exception in Monitor, etc. write tests
        #   for those scenarios.



INITIALIZE_TB =\
{'message': 'Algo exception in initialize',
 'name': 'Exception',
 'stack': [{'filename': '/zipline/lines.py', 'line': 'for event in self.gen:', 'lineno': 152, 'method': 'stream_results'},
           {'filename': '/zipline/gens/tradesimulation.py', 'line': 'self.algo,', 'lineno': 93, 'method': 'simulate'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.algo.initialize()',
            'lineno': 123,
            'method': '__init__'},
           {'filename': '/zipline/test_algorithms.py',
            'line': 'raise Exception("Algo exception in initialize")',
            'lineno': 166,
            'method': 'initialize'}]}

HANDLE_DATA_TB =\
{'message': 'Algo exception in handle_data',
 'name': 'Exception',
 'stack': [{'filename': '/zipline/lines.py', 'line': 'for event in self.gen:', 'lineno': 152, 'method': 'stream_results'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'for message in algo_results:',
            'lineno': 100,
            'method': 'simulate'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'return self.__generator.next()',
            'lineno': 144,
            'method': 'next'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.update_current_snapshot(event)',
            'lineno': 199,
            'method': '_gen'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.simulate_current_snapshot()',
            'lineno': 221,
            'method': 'update_current_snapshot'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.algo.handle_data(self.universe)',
            'lineno': 246,
            'method': 'simulate_current_snapshot'},
           {'filename': '/zipline/test_algorithms.py',
            'line': 'raise Exception("Algo exception in handle_data")',
            'lineno': 187,
            'method': 'handle_data'}]}

ZERO_DIV_TB= \
{'message': 'integer division or modulo by zero',
 'name': 'ZeroDivisionError',
 'stack': [{'filename': '/zipline/lines.py', 'line': 'for event in self.gen:', 'lineno': 152, 'method': 'stream_results'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'for message in algo_results:',
            'lineno': 100,
            'method': 'simulate'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'return self.__generator.next()',
            'lineno': 144,
            'method': 'next'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.update_current_snapshot(event)',
            'lineno': 199,
            'method': '_gen'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.simulate_current_snapshot()',
            'lineno': 221,
            'method': 'update_current_snapshot'},
           {'filename': '/zipline/gens/tradesimulation.py',
            'line': 'self.algo.handle_data(self.universe)',
            'lineno': 246,
            'method': 'simulate_current_snapshot'},
           {'filename': '/zipline/test_algorithms.py', 'line': '5/0', 'lineno': 218, 'method': 'handle_data'}]}
