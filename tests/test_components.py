import zmq
import pytz
from pprint import pformat as pf
from datetime import datetime, timedelta

from unittest2 import TestCase
from collections import defaultdict
from zipline.gens.composites import date_sorted_sources, merged_transforms

from zipline.finance.trading import SIMULATION_STYLE
from zipline.core.devsimulator import AddressAllocator
from zipline.gens.transform import MovingAverage, Passthrough, StatefulTransform
from zipline.gens.tradesimulation import TradeSimulationClient as tsc

from zipline.utils.factory import create_trading_environment
from zipline.test_algorithms import TestAlgorithm


from zipline.utils.test_utils import (
        setup_logger,
        teardown_logger,
        create_monitor,
        launch_monitor
)

from zipline.core import Component
from zipline.protocol import (
    DATASOURCE_FRAME,
    DATASOURCE_UNFRAME,
    FEED_FRAME,
    FEED_UNFRAME,
    MERGE_FRAME,
    MERGE_UNFRAME,
    SIMULATION_STYLE
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


        comp_a = Component(
            trade_gen,
            monitor,
            socket_uri,
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME
        )

        launch_monitor(monitor)

        for event in comp_a:
            log.info(event)

        # wait for the sending process to exit
        comp_a.proc.join()


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

        launch_monitor(monitor)

        prev = None
        sort_count = 0
        for msg in sorted_out:
            if prev:
                self.assertTrue(msg.dt >= prev.dt, \
                        "Messages should be in date ascending order")
            prev = msg
            sort_count += 1

        self.assertEqual(count*3, sort_count)

        # wait for processes to finish
        comp_a.proc.join()
        comp_b.proc.join()
        comp_c.proc.join()


    def test_full(self):
        monitor     = create_monitor(allocator)

        filter = [2,3]
        #Set up source a. One minute between events.
        args_a = tuple()
        kwargs_a = {
            'count'  : 325,
            'sids'   : [1,2,3],
            'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
            'delta'  : timedelta(hours = 6),
            'filter' : filter
        }
        source_a = SpecificEquityTrades(*args_a, **kwargs_a)

        #Set up source b. Two minutes between events.
        args_b = tuple()
        kwargs_b = {
            'count'  : 7500,
            'sids'   : [2,3,4],
            'start'  : datetime(2012,1,3,14, tzinfo = pytz.utc),
            'delta'  : timedelta(minutes = 5),
            'filter' : filter
        }
        source_b = SpecificEquityTrades(*args_b, **kwargs_b)

        # ------------------------
        # Run sources in dedicated processes
        comp_a = Component(
            source_a,
            monitor,
            allocator.lease(1)[0],
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME,
            source_a.get_hash()
        )

        comp_b = Component(
            source_b,
            monitor,
            allocator.lease(1)[0],
            DATASOURCE_FRAME,
            DATASOURCE_UNFRAME,
            source_b.get_hash()
        )

        # Date sort the sources, and run the sort in a dedicated
        # process
        sources = [comp_a, comp_b]

        sorted_out = date_sorted_sources(*sources)

        #launch_monitor(monitor)
        #import nose.tools; nose.tools.set_trace()
        #for feed_msg in sorted_out:
        #    log.info(pf(feed_msg))

        #return

        sorted = Component(
                sorted_out,
                monitor,
                allocator.lease(1)[0],
                FEED_FRAME,
                FEED_UNFRAME,
                "sort"
                )


        passthrough = StatefulTransform(Passthrough)
        mavg_price = StatefulTransform(
                MovingAverage,
                timedelta(minutes = 20),
                ['price']
        )

        merged_gen = merged_transforms(sorted, passthrough, mavg_price)

        merged = Component(
                    merged_gen,
                    monitor,
                    allocator.lease(1)[0],
                    MERGE_FRAME,
                    MERGE_UNFRAME,
                    "merge"
                 )

        algo = TestAlgorithm(2, 10, 100, sid_filter = [2,3])
        environment = create_trading_environment(year = 2012)
        style = SIMULATION_STYLE.FIXED_SLIPPAGE

        trading_client = tsc(algo, environment, style)

        launch_monitor(monitor)
        for message in trading_client.simulate(merged):
            log.info(pf(message))


        # wait for processes to finish
        comp_a.proc.join()
        comp_b.proc.join()
        sorted.proc.join()
        merged.proc.join()
        return



    def test_compound(self):
        monitor     = create_monitor(allocator)

        filter = [2,3]
        #Set up source a. One minute between events.
        args_a = tuple()
        kwargs_a = {
            'count'  : 325,
            'sids'   : [1,2,3],
            'start'  : datetime(2012,1,3,15, tzinfo = pytz.utc),
            'delta'  : timedelta(hours = 6),
            'filter' : filter
        }
        source_a = SpecificEquityTrades(*args_a, **kwargs_a)

        #Set up source b. Two minutes between events.
        args_b = tuple()
        kwargs_b = {
            'count'  : 7500,
            'sids'   : [2,3,4],
            'start'  : datetime(2012,1,3,14, tzinfo = pytz.utc),
            'delta'  : timedelta(minutes = 5),
            'filter' : filter
        }
        source_b = SpecificEquityTrades(*args_b, **kwargs_b)

        sorted_out = date_sorted_sources(source_a, source_b)

        sorted = Component(
                sorted_out,
                monitor,
                allocator.lease(1)[0],
                FEED_FRAME,
                FEED_UNFRAME
        )

        launch_monitor(monitor)

        for event in sorted:
            log.info(event)


        sorted.proc.join()
