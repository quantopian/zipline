import zmq
import pytz
from datetime import datetime, timedelta

from unittest2 import TestCase
from collections import defaultdict

from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.lines import SimulatedTrading

from zipline.utils.test_utils import (
        drain_zipline,
        check,
        setup_logger,
        teardown_logger,
        launch_component,
        create_monitor,
        launch_monitor
)


from zipline.core import Component
from zipline.core.component import ComponentSocketArgs
from zipline.protocol import (
    DATASOURCE_FRAME,
    DATASOURCE_UNFRAME
)

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.utils import hash_args
from zipline.gens.zmqgen import gen_from_poller

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
        filter = [1,2,3,4]
        #Set up source a. One minute between events.
        args_a = tuple()
        kwargs_a = {
            'sids'   : [1,2],
            'start'  : datetime(2012,6,6,0,tzinfo=pytz.utc),
            'delta'  : timedelta(minutes = 1),
            'filter' : filter,
            'count'  : 100
        }

        c_id = SpecificEquityTrades.__name__ + hash_args(args_a, kwargs_a)
        mon = create_monitor(allocator)

        out_socket_args = ComponentSocketArgs(
                style=zmq.PUSH,
                uri=allocator.lease(1)[0],
                bind=True
        )

        c = Component(
                SpecificEquityTrades,
                args_a,
                kwargs_a,
                c_id,
                out_socket_args,
                DATASOURCE_FRAME,
                mon
            )

        mon.manage(set([c.get_id]))
        mon_proc = launch_monitor(mon)

        # launch in a process
        proc = launch_component(c)

        pull_socket = self.ctx.socket(zmq.PULL)
        pull_socket.connect(out_socket_args.uri)
        poller = zmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)
        unframe = DATASOURCE_UNFRAME
        for msg in gen_from_poller(poller, pull_socket, unframe):
            # assert things about the messages.
            log.info(msg)

        pull_socket.close()
        log.info("DONE!")
