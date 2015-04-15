import pytz
from datetime import datetime, timedelta
from unittest import TestCase


from zipline.algorithm import TradingAlgorithm
from zipline.errors import TradingControlViolation
from zipline.sources import SpecificEquityTrades
from zipline.finance import trading
from zipline.utils.test_utils import (
    setup_logger, teardown_logger, security_list_copy, add_security_data)
from zipline.utils import factory
from zipline.utils.security_list import (
    SecurityListSet, load_from_directory)

LEVERAGED_ETFS = load_from_directory('leveraged_etf_list')


class RestrictedAlgoWithCheck(TradingAlgorithm):
    def initialize(self, symbol):
            self.rl = SecurityListSet(self.get_datetime)
            self.set_do_not_order_list(self.rl.leveraged_etf_list)
            self.order_count = 0
            self.sid = self.symbol(symbol)

    def handle_data(self, data):
        if not self.order_count:
            if self.sid not in \
                    self.rl.leveraged_etf_list:
                self.order(self.sid, 100)
                self.order_count += 1


class RestrictedAlgoWithoutCheck(TradingAlgorithm):
    def initialize(self, symbol):
        self.rl = SecurityListSet(self.get_datetime)
        self.set_do_not_order_list(self.rl.leveraged_etf_list)
        self.order_count = 0
        self.sid = self.symbol(symbol)

    def handle_data(self, data):
        self.order(self.sid, 100)
        self.order_count += 1


class IterateRLAlgo(TradingAlgorithm):
    def initialize(self, symbol):
            self.rl = SecurityListSet(self.get_datetime)
            self.set_do_not_order_list(self.rl.leveraged_etf_list)
            self.order_count = 0
            self.sid = self.symbol(symbol)
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

        trading.TradingEnvironment().update_asset_finder(
            erase_existing=True,
            identifiers=["BZQ", "URTY", "JFT", "AAPL", "GOOG"])

        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

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
        algo = IterateRLAlgo(symbol='BZQ', sim_params=sim_params)
        algo.run(self.source)
        self.assertTrue(algo.found)

    def test_security_list(self):

        # set the knowledge date to the first day of the
        # leveraged etf knowledge date.
        def get_datetime():
            return list(LEVERAGED_ETFS.keys())[0]

        rl = SecurityListSet(get_datetime)
        # assert that a sample from the leveraged list are in restricted
        should_exist = [
            asset.sid for asset in
            trading.environment.asset_finder.lookup_generic(
                ["BZQ", "URTY", "JFT"],
                as_of_date=self.extra_knowledge_date)[0]
        ]
        for sid in should_exist:
            self.assertIn(sid, rl.leveraged_etf_list)

        # assert that a sample of allowed stocks are not in restricted
        shouldnt_exist = [
            asset.sid for asset in
            trading.environment.asset_finder.lookup_generic(
                ["AAPL", "GOOG"],
                as_of_date=self.extra_knowledge_date)[0]
        ]
        for sid in shouldnt_exist:
            self.assertNotIn(sid, rl.leveraged_etf_list)

    def test_security_add(self):
        def get_datetime():
            return datetime(2015, 1, 27, tzinfo=pytz.utc)
        with security_list_copy():
            add_security_data(['AAPL', 'GOOG'], [])
            rl = SecurityListSet(get_datetime)
            should_exist = [
                asset.sid for asset in
                trading.environment.asset_finder.lookup_generic(
                    ["AAPL", "GOOG", "BZQ", "URTY"],
                    as_of_date=self.extra_knowledge_date)[0]
            ]
            for sid in should_exist:
                self.assertIn(sid, rl.leveraged_etf_list)

    def test_security_add_delete(self):
        with security_list_copy():
            def get_datetime():
                return datetime(2015, 1, 27, tzinfo=pytz.utc)
            add_security_data([], ['BZQ', 'URTY'])
            rl = SecurityListSet(get_datetime)
            self.assertNotIn("BZQ", rl.leveraged_etf_list)
            self.assertNotIn("URTY", rl.leveraged_etf_list)

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

        algo = RestrictedAlgoWithCheck(symbol='BZQ', sim_params=sim_params)
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
        algo = RestrictedAlgoWithoutCheck(symbol='AAPL', sim_params=sim_params)
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

        algo = RestrictedAlgoWithoutCheck(symbol='BZQ', sim_params=sim_params)
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

        algo = RestrictedAlgoWithoutCheck(symbol='JFT', sim_params=sim_params)
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
        algo = RestrictedAlgoWithoutCheck(symbol='BZQ', sim_params=sim_params)
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

        with security_list_copy():
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
                symbol='BZQ', sim_params=sim_params)
            with self.assertRaises(TradingControlViolation) as ctx:
                algo.run(self.source)

            self.check_algo_exception(algo, ctx, 0)

    def test_algo_without_rl_violation_after_delete(self):
        with security_list_copy():
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
                symbol='BZQ', sim_params=sim_params
            )
            algo.run(self.source)

    def test_algo_with_rl_violation_after_add(self):
        with security_list_copy():
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
                symbol='AAPL', sim_params=sim_params)
            with self.assertRaises(TradingControlViolation) as ctx:
                algo.run(self.source)

            self.check_algo_exception(algo, ctx, 2)

    def check_algo_exception(self, algo, ctx, expected_order_count):
        self.assertEqual(algo.order_count, expected_order_count)
        exc = ctx.exception
        self.assertEqual(TradingControlViolation, type(exc))
        exc_msg = str(ctx.exception)
        self.assertTrue("RestrictedListOrder" in exc_msg)
