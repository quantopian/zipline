import zmq
import pytz
from datetime import datetime, timedelta

from unittest2 import TestCase
from collections import defaultdict
from zipline.gens.composites import date_sorted_sources

from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator

from zipline.utils.test_utils import (
        setup_logger,
        teardown_logger,
        create_monitor,
        launch_monitor
)

from zipline.core import Component
from zipline.protocol import (
    DATASOURCE_FRAME,
    DATASOURCE_UNFRAME
)

from zipline.gens.tradegens import SpecificEquityTrades

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
        teardown_logger(self)

    def test_source(self):
        monitor = create_monitor(allocator)
        socket_uri = allocator.lease(1)[0]
        count = 100

        filter = [1,2,3,4]
        #Set up source a. One minute between events.
        args_a = tuple()
        kwargs_a = {
            'sids'   : [1,2],
            'start'  : datetime(2012,6,6,0,tzinfo=pytz.utc),
            'delta'  : timedelta(minutes = 1),
            'filter' : filter,
            'count'  : count
        }

        trade_gen = SpecificEquityTrades(*args_a, **kwargs_a)
        monitor.add_to_topology(trade_gen.get_hash())

        launch_monitor(monitor)

        comp_a = Component(
            trade_gen,
            monitor,
            socket_uri,
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME
        )


        for event in comp_a:
            log.info(event)


    def test_sort(self):
        monitor     = create_monitor(allocator)
        socket_uris    = allocator.lease(3)
        count       = 100

        filter = [1,2,3,4]
        #Set up source a. One minute between events.
        args_a = tuple()
        kwargs_a = {
            'sids'   : [1,2],
            'start'  : datetime(2012,6,6,0,tzinfo=pytz.utc),
            'delta'  : timedelta(minutes = 1),
            'filter' : filter,
            'count'  : count
        }
        trade_gen_a = SpecificEquityTrades(*args_a, **kwargs_a)
        monitor.add_to_topology(trade_gen_a.get_hash())

        #Set up source b. Two minutes between events.
        args_b = tuple()
        kwargs_b = {
            'sids'   : [2],
            'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
            'delta'  : timedelta(minutes = 1),
            'filter' : filter,
            'count'  : count
        }
        trade_gen_b = SpecificEquityTrades(*args_b, **kwargs_b)
        monitor.add_to_topology(trade_gen_b.get_hash())

        #Set up source c. Three minutes between events.
        args_c = tuple()
        kwargs_c = {
            'sids'   : [3],
            'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
            'delta'  : timedelta(minutes = 1),
            'filter' : filter,
            'count'  : count
        }

        trade_gen_c = SpecificEquityTrades(*args_c, **kwargs_c)
        monitor.add_to_topology(trade_gen_c.get_hash())

        launch_monitor(monitor)

        comp_a = Component(
            trade_gen_a,
            monitor,
            socket_uris[0],
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME
        )

        comp_b = Component(
            trade_gen_b,
            monitor,
            socket_uris[1],
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME
        )

        comp_c = Component(
            trade_gen_c,
            monitor,
            socket_uris[2],
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME
        )

        sources = [comp_a, comp_b, comp_c]

        sorted_out = date_sorted_sources(*sources)

        prev = None
        sort_count = 0
        for msg in sorted_out:
            if prev:
                self.assertTrue(msg.dt >= prev.dt, \
                        "Messages should be in date ascending order")
            prev = msg
            sort_count += 1

        self.assertEqual(count*3, sort_count)
