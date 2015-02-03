import pytz
import os.path
import shutil
from datetime import datetime, timedelta
from unittest import TestCase


from zipline.algorithm import TradingAlgorithm
from zipline.errors import TradingControlViolation
from zipline.sources import SpecificEquityTrades
from zipline.utils.test_utils import setup_logger
from zipline.utils import factory
from zipline.utils.security_list import (
    SecurityListSet, load_from_directory, SECURITY_LISTS_DIR)

LEVERAGED_ETFS = load_from_directory('leveraged_etf_list')


class RestrictedAlgoWithCheck(TradingAlgorithm):
    def initialize(self, sid):
            self.rl = SecurityListSet(self.get_datetime)
            self.set_do_not_order_list(self.rl.LEVERAGED_ETF_LIST)
            self.order_count = 0
            self.sid = sid

    def handle_data(self, data):
        if not self.order_count:
            if self.sid not in \
                    self.rl.LEVERAGED_ETF_LIST:
                self.order(self.sid, 100)
                self.order_count += 1


class RestrictedAlgoWithoutCheck(TradingAlgorithm):
    def initialize(self, sid):
        self.rl = SecurityListSet(self.get_datetime)
        self.set_do_not_order_list(self.rl.LEVERAGED_ETF_LIST)
        self.order_count = 0
        self.sid = sid

    def handle_data(self, data):
        self.order(self.sid, 100)
        self.order_count += 1


class IterateRLAlgo(TradingAlgorithm):
    def initialize(self, sid):
            self.rl = SecurityListSet(self.get_datetime)
            self.set_do_not_order_list(self.rl.LEVERAGED_ETF_LIST)
            self.order_count = 0
            self.sid = sid
            self.found = False

    def handle_data(self, data):
        for stock in self.rl.LEVERAGED_ETF_LIST:
            if stock == self.sid:
                self.found = True


class SecurityListTestCase(TestCase):

    def setUp(self):
        self.extra_knowledge_date = \
            datetime(2015, 1, 27, 0, 0, tzinfo=pytz.utc)
        self.trading_day_before_first_kd = datetime(
            2015, 1, 23, 0, 0, tzinfo=pytz.utc)

        setup_logger(self)

    def test_iterate_over_rl(self):
        sim_params = factory.create_simulation_parameters(
            start=LEVERAGED_ETFS.keys()[0], num_days=4)

        trade_history = factory.create_trade_history(
            'BZQ',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)
        algo = IterateRLAlgo(sid='BZQ', sim_params=sim_params)
        algo.run(self.source)
        self.assertTrue(algo.found)

    def test_security_list(self):

        # set the knowledge date to the first day of the
        # leveraged etf knowledge date.
        def get_datetime():
            return LEVERAGED_ETFS.keys()[0]

        rl = SecurityListSet(get_datetime)
        # assert that a sample from the leveraged list are in restricted

        self.assertIn("BZQ", rl.LEVERAGED_ETF_LIST)
        self.assertIn("URTY", rl.LEVERAGED_ETF_LIST)

        # assert that a sample of allowed stocks are not in restricted
        # AAPL
        self.assertNotIn("AAPL", rl.LEVERAGED_ETF_LIST)
        # GOOG
        self.assertNotIn("GOOG", rl.LEVERAGED_ETF_LIST)

    def test_security_add(self):
        def get_datetime():
            return datetime(2015, 1, 27, tzinfo=pytz.utc)
        try:
            add_data(['AAPL', 'GOOG'], [])
            rl = SecurityListSet(get_datetime)
            self.assertIn("AAPL", rl.LEVERAGED_ETF_LIST)
            self.assertIn("GOOG", rl.LEVERAGED_ETF_LIST)
            self.assertIn("BZQ", rl.LEVERAGED_ETF_LIST)
            self.assertIn("URTY", rl.LEVERAGED_ETF_LIST)
        finally:
            remove_data_directory()

    def test_security_add_delete(self):
        try:
            def get_datetime():
                return datetime(2015, 1, 27, tzinfo=pytz.utc)
            add_data([], ['BZQ', 'URTY'])
            rl = SecurityListSet(get_datetime)
            self.assertNotIn("BZQ", rl.LEVERAGED_ETF_LIST)
            self.assertNotIn("URTY", rl.LEVERAGED_ETF_LIST)
        finally:
            remove_data_directory()

    def test_algo_without_rl_violation_via_check(self):
        sim_params = factory.create_simulation_parameters(
            start=LEVERAGED_ETFS.keys()[0], num_days=4)

        trade_history = factory.create_trade_history(
            'BZQ',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        algo = RestrictedAlgoWithCheck(sid='BZQ', sim_params=sim_params)
        algo.run(self.source)

    def test_algo_without_rl_violation(self):
        sim_params = factory.create_simulation_parameters(
            start=LEVERAGED_ETFS.keys()[0], num_days=4)

        trade_history = factory.create_trade_history(
            'AAPL',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)
        algo = RestrictedAlgoWithoutCheck(sid='AAPL', sim_params=sim_params)
        algo.run(self.source)

    def test_algo_with_rl_violation(self):
        sim_params = factory.create_simulation_parameters(
            start=LEVERAGED_ETFS.keys()[0], num_days=4)

        trade_history = factory.create_trade_history(
            'BZQ',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        self.df_source, self.df = \
            factory.create_test_df_source(sim_params)

        algo = RestrictedAlgoWithoutCheck(sid='BZQ', sim_params=sim_params)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(self.source)

        self.check_algo_exception(algo, ctx, 0)

    def test_algo_with_rl_violation_on_knowledge_date(self):

        sim_params = factory.create_simulation_parameters(
            start=self.trading_day_before_first_kd, num_days=4)
        trade_history = factory.create_trade_history(
            'BZQ',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        algo = RestrictedAlgoWithoutCheck(sid='BZQ', sim_params=sim_params)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(self.source)

        self.check_algo_exception(algo, ctx, 1)

    def test_algo_with_rl_violation_after_knowledge_date(self):
        sim_params = factory.create_simulation_parameters(
            start=LEVERAGED_ETFS.keys()[0] + timedelta(days=7), num_days=5)

        trade_history = factory.create_trade_history(
            'BZQ',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)
        algo = RestrictedAlgoWithoutCheck(sid='BZQ', sim_params=sim_params)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(self.source)

        self.check_algo_exception(algo, ctx, 0)

    def test_algo_with_rl_violation_cumulative(self):
        """
        Add a new restriction, run a test long after both
        knowledge dates, make sure stock from original restriction
        set is still disallowed.
        """
        sim_params = factory.create_simulation_parameters(
            start=LEVERAGED_ETFS.keys()[0] + timedelta(days=7), num_days=4)

        try:
            add_data(['AAPL'], [])
            trade_history = factory.create_trade_history(
                'BZQ',
                [10.0, 10.0, 11.0, 11.0],
                [100, 100, 100, 300],
                timedelta(days=1),
                sim_params
            )
            self.source = SpecificEquityTrades(event_list=trade_history)
            algo = RestrictedAlgoWithoutCheck(
                sid='BZQ', sim_params=sim_params)
            with self.assertRaises(TradingControlViolation) as ctx:
                algo.run(self.source)

            self.check_algo_exception(algo, ctx, 0)
        finally:
            remove_data_directory()

    def test_algo_without_rl_violation_after_delete(self):
        try:
            # add a delete statement removing bzq
            # write a new delete statement file to disk
            add_data([], ['BZQ'])

            sim_params = factory.create_simulation_parameters(
                start=self.extra_knowledge_date, num_days=3)
            trade_history = factory.create_trade_history(
                'BZQ',
                [10.0, 10.0, 11.0, 11.0],
                [100, 100, 100, 300],
                timedelta(days=1),
                sim_params
            )
            self.source = SpecificEquityTrades(event_list=trade_history)
            algo = RestrictedAlgoWithoutCheck(
                sid='BZQ', sim_params=sim_params)
            algo.run(self.source)
        finally:
            remove_data_directory()

    def test_algo_with_rl_violation_after_add(self):
        try:
            add_data(['AAPL'], [])
            sim_params = factory.create_simulation_parameters(
                start=self.trading_day_before_first_kd, num_days=4)
            trade_history = factory.create_trade_history(
                'AAPL',
                [10.0, 10.0, 11.0, 11.0],
                [100, 100, 100, 300],
                timedelta(days=1),
                sim_params
            )
            self.source = SpecificEquityTrades(event_list=trade_history)
            algo = RestrictedAlgoWithoutCheck(
                sid='AAPL', sim_params=sim_params)
            with self.assertRaises(TradingControlViolation) as ctx:
                algo.run(self.source)

            self.check_algo_exception(algo, ctx, 2)
        finally:
            remove_data_directory()

    def check_algo_exception(self, algo, ctx, expected_order_count):
        self.assertEqual(algo.order_count, expected_order_count)
        exc = ctx.exception
        self.assertEqual(TradingControlViolation, type(exc))
        exc_msg = str(ctx.exception)
        self.assertTrue("RestrictedListOrder" in exc_msg)


def add_data(adds, deletes):
    directory = os.path.join(
        SECURITY_LISTS_DIR,
        "leveraged_etf_list/20150127/20150125"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    del_path = os.path.join(directory, "delete.txt")
    with open(del_path, 'w') as f:
        for sym in deletes:
            f.write(sym)
            f.write('\n')
    add_path = os.path.join(directory, "add.txt")
    with open(add_path, 'w') as f:
        for sym in adds:
            f.write(sym)
            f.write('\n')


def remove_data_directory():
    directory = os.path.join(
        SECURITY_LISTS_DIR,
        "leveraged_etf_list/20150127/"
    )
    shutil.rmtree(directory)
