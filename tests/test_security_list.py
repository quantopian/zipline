import pytz
from datetime import datetime, timedelta
from unittest import TestCase


from zipline.algorithm import TradingAlgorithm
from zipline.errors import TradingControlViolation
from zipline.sources import SpecificEquityTrades
from zipline.utils.test_utils import (
    setup_logger, add_security_data, remove_security_data_directory)
from zipline.utils import factory
from zipline.utils.security_list import (
    SecurityListSet, load_from_directory)

LEVERAGED_ETFS = load_from_directory('leveraged_etf_list')


class RestrictedAlgoWithCheck(TradingAlgorithm):
    def initialize(self, sid):
            self.rl = SecurityListSet(self.get_datetime)
            self.set_do_not_order_list(self.rl.leveraged_etf_list)
            self.order_count = 0
            self.sid = sid

    def handle_data(self, data):
        if not self.order_count:
            if self.sid not in \
                    self.rl.leveraged_etf_list:
                self.order(self.sid, 100)
                self.order_count += 1


class RestrictedAlgoWithoutCheck(TradingAlgorithm):
    def initialize(self, sid):
        self.rl = SecurityListSet(self.get_datetime)
        self.set_do_not_order_list(self.rl.leveraged_etf_list)
        self.order_count = 0
        self.sid = sid

    def handle_data(self, data):
        self.order(self.sid, 100)
        self.order_count += 1


class IterateRLAlgo(TradingAlgorithm):
    def initialize(self, sid):
            self.rl = SecurityListSet(self.get_datetime)
            self.set_do_not_order_list(self.rl.leveraged_etf_list)
            self.order_count = 0
            self.sid = sid
            self.found = False

    def handle_data(self, data):
        for stock in self.rl.leveraged_etf_list:
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
            start=list(LEVERAGED_ETFS.keys())[0], num_days=4)

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
            return list(LEVERAGED_ETFS.keys())[0]

        rl = SecurityListSet(get_datetime)
        # assert that a sample from the leveraged list are in restricted

        self.assertIn("BZQ", rl.leveraged_etf_list)
        self.assertIn("URTY", rl.leveraged_etf_list)
        self.assertIn("JFT", rl.leveraged_etf_list)

        # assert that a sample of allowed stocks are not in restricted
        # AAPL
        self.assertNotIn("AAPL", rl.leveraged_etf_list)
        # GOOG
        self.assertNotIn("GOOG", rl.leveraged_etf_list)

    def test_security_add(self):
        def get_datetime():
            return datetime(2015, 1, 27, tzinfo=pytz.utc)
        try:
            add_security_data(['AAPL', 'GOOG'], [])
            rl = SecurityListSet(get_datetime)
            self.assertIn("AAPL", rl.leveraged_etf_list)
            self.assertIn("GOOG", rl.leveraged_etf_list)
            self.assertIn("BZQ", rl.leveraged_etf_list)
            self.assertIn("URTY", rl.leveraged_etf_list)
        finally:
            remove_security_data_directory()

    def test_security_add_delete(self):
        try:
            def get_datetime():
                return datetime(2015, 1, 27, tzinfo=pytz.utc)
            add_security_data([], ['BZQ', 'URTY'])
            rl = SecurityListSet(get_datetime)
            self.assertNotIn("BZQ", rl.leveraged_etf_list)
            self.assertNotIn("URTY", rl.leveraged_etf_list)
        finally:
            remove_security_data_directory()

    def test_algo_without_rl_violation_via_check(self):
        sim_params = factory.create_simulation_parameters(
            start=list(LEVERAGED_ETFS.keys())[0], num_days=4)

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
            start=list(LEVERAGED_ETFS.keys())[0], num_days=4)

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
            start=list(LEVERAGED_ETFS.keys())[0], num_days=4)

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

        # repeat with a symbol from a different lookup date

        trade_history = factory.create_trade_history(
            'JFT',
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        self.df_source, self.df = \
            factory.create_test_df_source(sim_params)

        algo = RestrictedAlgoWithoutCheck(sid='JFT', sim_params=sim_params)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(self.source)

        self.check_algo_exception(algo, ctx, 0)

    def test_algo_with_rl_violation_after_knowledge_date(self):
        sim_params = factory.create_simulation_parameters(
            start=list(
                LEVERAGED_ETFS.keys())[0] + timedelta(days=7), num_days=5)

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
            start=list(
                LEVERAGED_ETFS.keys())[0] + timedelta(days=7), num_days=4)

        try:
            add_security_data(['AAPL'], [])
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
            remove_security_data_directory()

    def test_algo_without_rl_violation_after_delete(self):
        try:
            # add a delete statement removing bzq
            # write a new delete statement file to disk
            add_security_data([], ['BZQ'])

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
            remove_security_data_directory()

    def test_algo_with_rl_violation_after_add(self):
        try:
            add_security_data(['AAPL'], [])
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
            remove_security_data_directory()

    def check_algo_exception(self, algo, ctx, expected_order_count):
        self.assertEqual(algo.order_count, expected_order_count)
        exc = ctx.exception
        self.assertEqual(TradingControlViolation, type(exc))
        exc_msg = str(ctx.exception)
        self.assertTrue("RestrictedListOrder" in exc_msg)
