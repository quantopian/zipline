import pandas as pd

from datetime import timedelta
from unittest import TestCase
from testfixtures import TempDirectory

from zipline.algorithm import TradingAlgorithm
from zipline.errors import TradingControlViolation
from zipline.finance.trading import TradingEnvironment
from zipline.testing import (
    add_security_data,
    security_list_copy,
    setup_logger,
    teardown_logger,
)
from zipline.testing.core import create_data_portal
from zipline.utils import factory
from zipline.utils.security_list import (
    SecurityListSet,
    load_from_directory,
)

LEVERAGED_ETFS = load_from_directory('leveraged_etf_list')


class RestrictedAlgoWithCheck(TradingAlgorithm):
    def initialize(self, symbol):
        self.rl = SecurityListSet(self.get_datetime, self.asset_finder)
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
        self.rl = SecurityListSet(self.get_datetime, self.asset_finder)
        self.set_do_not_order_list(self.rl.leveraged_etf_list)
        self.order_count = 0
        self.sid = self.symbol(symbol)

    def handle_data(self, data):
        self.order(self.sid, 100)
        self.order_count += 1


class IterateRLAlgo(TradingAlgorithm):
    def initialize(self, symbol):
        self.rl = SecurityListSet(self.get_datetime, self.asset_finder)
        self.set_do_not_order_list(self.rl.leveraged_etf_list)
        self.order_count = 0
        self.sid = self.symbol(symbol)
        self.found = False

    def handle_data(self, data):
        for stock in self.rl.leveraged_etf_list:
            if stock == self.sid:
                self.found = True


class SecurityListTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        # this is ugly, but we need to create two different
        # TradingEnvironment/DataPortal pairs

        cls.env = TradingEnvironment()
        cls.env2 = TradingEnvironment()

        cls.extra_knowledge_date = pd.Timestamp("2015-01-27", tz='UTC')
        cls.trading_day_before_first_kd = pd.Timestamp("2015-01-23", tz='UTC')

        symbols = ['AAPL', 'GOOG', 'BZQ', 'URTY', 'JFT']

        days = cls.env.days_in_range(
            list(LEVERAGED_ETFS.keys())[0],
            pd.Timestamp("2015-02-17", tz='UTC')
        )

        cls.sim_params = factory.create_simulation_parameters(
            start=list(LEVERAGED_ETFS.keys())[0],
            num_days=4,
            env=cls.env
        )

        cls.sim_params2 = factory.create_simulation_parameters(
            start=cls.trading_day_before_first_kd, num_days=4
        )

        equities_metadata = {}

        for i, symbol in enumerate(symbols):
            equities_metadata[i] = {
                'start_date': days[0],
                'end_date': days[-1],
                'symbol': symbol
            }

        equities_metadata2 = {}
        for i, symbol in enumerate(symbols):
            equities_metadata2[i] = {
                'start_date': cls.sim_params2.period_start,
                'end_date': cls.sim_params2.period_end,
                'symbol': symbol
            }

        cls.env.write_data(equities_data=equities_metadata)
        cls.env2.write_data(equities_data=equities_metadata2)

        cls.tempdir = TempDirectory()
        cls.tempdir2 = TempDirectory()

        cls.data_portal = create_data_portal(
            env=cls.env,
            tempdir=cls.tempdir,
            sim_params=cls.sim_params,
            sids=range(0, 5),
        )

        cls.data_portal2 = create_data_portal(
            env=cls.env2,
            tempdir=cls.tempdir2,
            sim_params=cls.sim_params2,
            sids=range(0, 5)
        )

        setup_logger(cls)

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()
        cls.tempdir2.cleanup()
        teardown_logger(cls)

    def test_iterate_over_restricted_list(self):
        algo = IterateRLAlgo(symbol='BZQ', sim_params=self.sim_params,
                             env=self.env)

        algo.run(self.data_portal)
        self.assertTrue(algo.found)

    def test_security_list(self):
        # set the knowledge date to the first day of the
        # leveraged etf knowledge date.
        def get_datetime():
            return list(LEVERAGED_ETFS.keys())[0]

        rl = SecurityListSet(get_datetime, self.env.asset_finder)
        # assert that a sample from the leveraged list are in restricted
        should_exist = [
            asset.sid for asset in
            [self.env.asset_finder.lookup_symbol(
                symbol,
                as_of_date=self.extra_knowledge_date)
             for symbol in ["BZQ", "URTY", "JFT"]]
        ]
        for sid in should_exist:
            self.assertIn(sid, rl.leveraged_etf_list)

        # assert that a sample of allowed stocks are not in restricted
        shouldnt_exist = [
            asset.sid for asset in
            [self.env.asset_finder.lookup_symbol(
                symbol,
                as_of_date=self.extra_knowledge_date)
             for symbol in ["AAPL", "GOOG"]]
        ]
        for sid in shouldnt_exist:
            self.assertNotIn(sid, rl.leveraged_etf_list)

    def test_security_add(self):
        def get_datetime():
            return pd.Timestamp("2015-01-27", tz='UTC')
        with security_list_copy():
            add_security_data(['AAPL', 'GOOG'], [])
            rl = SecurityListSet(get_datetime, self.env.asset_finder)
            should_exist = [
                asset.sid for asset in
                [self.env.asset_finder.lookup_symbol(
                    symbol,
                    as_of_date=self.extra_knowledge_date
                ) for symbol in ["AAPL", "GOOG", "BZQ", "URTY"]]
            ]
            for sid in should_exist:
                self.assertIn(sid, rl.leveraged_etf_list)

    def test_security_add_delete(self):
        with security_list_copy():
            def get_datetime():
                return pd.Timestamp("2015-01-27", tz='UTC')
            rl = SecurityListSet(get_datetime, self.env.asset_finder)
            self.assertNotIn("BZQ", rl.leveraged_etf_list)
            self.assertNotIn("URTY", rl.leveraged_etf_list)

    def test_algo_without_rl_violation_via_check(self):
        algo = RestrictedAlgoWithCheck(symbol='BZQ',
                                       sim_params=self.sim_params,
                                       env=self.env)
        algo.run(self.data_portal)

    def test_algo_without_rl_violation(self):
        algo = RestrictedAlgoWithoutCheck(symbol='AAPL',
                                          sim_params=self.sim_params,
                                          env=self.env)
        algo.run(self.data_portal)

    def test_algo_with_rl_violation(self):
        algo = RestrictedAlgoWithoutCheck(symbol='BZQ',
                                          sim_params=self.sim_params,
                                          env=self.env)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(self.data_portal)

        self.check_algo_exception(algo, ctx, 0)

        # repeat with a symbol from a different lookup date
        algo = RestrictedAlgoWithoutCheck(symbol='JFT',
                                          sim_params=self.sim_params,
                                          env=self.env)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(self.data_portal)

        self.check_algo_exception(algo, ctx, 0)

    def test_algo_with_rl_violation_after_knowledge_date(self):
        sim_params = factory.create_simulation_parameters(
            start=list(
                LEVERAGED_ETFS.keys())[0] + timedelta(days=7), num_days=5,
            env=self.env)

        data_portal = create_data_portal(
            self.env,
            self.tempdir,
            sim_params=sim_params,
            sids=range(0, 5)
        )

        algo = RestrictedAlgoWithoutCheck(symbol='BZQ',
                                          sim_params=sim_params,
                                          env=self.env)
        with self.assertRaises(TradingControlViolation) as ctx:
            algo.run(data_portal)

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
            algo = RestrictedAlgoWithoutCheck(
                symbol='BZQ', sim_params=sim_params, env=self.env)
            with self.assertRaises(TradingControlViolation) as ctx:
                algo.run(self.data_portal)

            self.check_algo_exception(algo, ctx, 0)

    def test_algo_without_rl_violation_after_delete(self):
        new_tempdir = TempDirectory()
        try:
            with security_list_copy():
                # add a delete statement removing bzq
                # write a new delete statement file to disk
                add_security_data([], ['BZQ'])

                # now fast-forward to self.extra_knowledge_date.  requires
                # a new env, simparams, and dataportal
                env = TradingEnvironment()
                sim_params = factory.create_simulation_parameters(
                    start=self.extra_knowledge_date, num_days=4, env=env)

                env.write_data(equities_data={
                    "0": {
                        'symbol': 'BZQ',
                        'start_date': sim_params.period_start,
                        'end_date': sim_params.period_end,
                    }
                })

                data_portal = create_data_portal(
                    env,
                    new_tempdir,
                    sim_params,
                    range(0, 5)
                )

                algo = RestrictedAlgoWithoutCheck(
                    symbol='BZQ', sim_params=sim_params, env=env
                )
                algo.run(data_portal)

        finally:
            new_tempdir.cleanup()

    def test_algo_with_rl_violation_after_add(self):
        with security_list_copy():
            add_security_data(['AAPL'], [])

            algo = RestrictedAlgoWithoutCheck(symbol='AAPL',
                                              sim_params=self.sim_params2,
                                              env=self.env2)
            with self.assertRaises(TradingControlViolation) as ctx:
                algo.run(self.data_portal2)

            self.check_algo_exception(algo, ctx, 2)

    def check_algo_exception(self, algo, ctx, expected_order_count):
        self.assertEqual(algo.order_count, expected_order_count)
        exc = ctx.exception
        self.assertEqual(TradingControlViolation, type(exc))
        exc_msg = str(ctx.exception)
        self.assertTrue("RestrictedListOrder" in exc_msg)
