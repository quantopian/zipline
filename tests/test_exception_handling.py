import zmq

from unittest2 import TestCase
from collections import defaultdict

from zipline.test_algorithms import ExceptionAlgorithm, DivByZeroAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading
from zipline.gens.transform import StatefulTransform

from zipline.utils.test_utils import \
        drain_zipline, \
        check, \
        setup_logger, \
        teardown_logger, \
        ExceptionSource, \
        ExceptionTransform

DEFAULT_TIMEOUT = 15 # seconds
EXTENDED_TIMEOUT = 90

allocator = AddressAllocator(1000)

class ExceptionTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        self.zipline_test_config = {
            'sid'                   : 133,
            'results_socket_uri'    : allocator.lease(1)[0],
            'simulation_style'      : SIMULATION_STYLE.FIXED_SLIPPAGE
        }
        self.ctx = zmq.Context()
        setup_logger(self)

    def tearDown(self):
        self.ctx.term()
        teardown_logger(self)

    def test_datasource_exception(self):
        self.zipline_test_config['trade_source'] = ExceptionSource()
        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )
        output, _ = drain_zipline(self, zipline)
        assert len(output) == 1
        assert output[0]['prefix'] == 'EXCEPTION'
        message = output[0]['payload']
        for field in ['date', 'message', 'name', 'stack']:
            assert field in message.keys()

        assert message['message'] == 'integer division or modulo by zero'
        assert message['name'] == 'ZeroDivisionError'

    def test_tranform_exception(self):
        exc_tnfm = StatefulTransform(ExceptionTransform)
        self.zipline_test_config['transforms'] = [exc_tnfm]

        zipline = SimulatedTrading.create_test_zipline(
            **self.zipline_test_config
        )
        output, _ = drain_zipline(self, zipline)
        assert len(output) == 1
        assert output[0]['prefix'] == 'EXCEPTION'
        message = output[0]['payload']
        for field in ['date', 'message', 'name', 'stack']:
            assert field in message.keys()

        assert message['message'] == 'An assertion message'
        assert message['name'] == 'AssertionError'


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

        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        self.assertEqual(payload['message'],'Algo exception in initialize')
        self.assertEqual(payload['name'],'Exception')
        # make sure our path shortening is working
        self.assertEqual(payload['stack'][0]['filename'], '/zipline/lines.py')
        self.assertEqual(payload['stack'][-1]['filename'], '/zipline/test_algorithms.py')

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
        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        self.assertEqual(payload['message'],'Algo exception in handle_data')
        self.assertEqual(payload['name'],'Exception')
        # make sure our path shortening is working
        self.assertEqual(payload['stack'][0]['filename'], '/zipline/lines.py')
        self.assertEqual(payload['stack'][-1]['filename'], '/zipline/test_algorithms.py')

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

        self.assertEqual(output[-1]['prefix'], 'EXCEPTION')
        payload = output[-1]['payload']
        self.assertTrue(payload['date'])
        del payload['date']
        self.assertEqual(payload['message'],'integer division or modulo by zero')
        self.assertEqual(payload['name'],'ZeroDivisionError')
        # make sure our path shortening is working
        self.assertEqual(payload['stack'][0]['filename'], '/zipline/lines.py')
        self.assertEqual(payload['stack'][-1]['filename'], '/zipline/test_algorithms.py')
