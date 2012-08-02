import zmq
from datetime import datetime, timedelta

from unittest2 import TestCase
from collections import defaultdict

from zipline.test_algorithms import ExceptionAlgorithm, DivByZeroAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading

from zipline.utils.test_utils import (
        drain_zipline,
        check,
        setup_logger,
        teardown_logger,
        launch_component,
        gen_from_socket
)


from zipline.core import Component
from zipline.protocol import (
    DATASOURCE_FRAME
)

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.utils import hash_args


import logbook
log = logbook.Logger('ComponentTestCase')

allocator = AddressAllocator(1000)


class ComponentTestCase(TestCase):

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

    def test_specific_equity_source(self):
        #Set up source a. One minute between events.
        args_a = tuple()
        kwargs_a = {
            'sids'   : [1,2],
            'start'  : datetime(2012,6,6,0),
            'delta'  : timedelta(minutes = 1),
            'filter' : filter
        }

        c_id = SpecificEquityTrades.__name__ + hash_args(args_a, kwargs_a)

        c = Component(
                SpecificEquityTrades,
                args_a,
                kwargs_a,
                out_uri=self.out_uri,
                frame=DATASOURCE_FRAME,
                monitor_uri=None
            )
        # launch in a process
        proc = launch_component(c)

        for msg in gen_from_socket(self.out_uri):
            # assert things about the messages.
            log.info(msg)
