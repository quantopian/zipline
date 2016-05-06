from datetime import timedelta

import pandas as pd
from testfixtures import TempDirectory

from zipline.algorithm import TradingAlgorithm
from zipline.errors import TradingControlViolation
from zipline.testing import (
    add_security_data,
    create_data_portal,
    security_list_copy,
    tmp_trading_env,
    tmp_dir,
)
from zipline.testing.fixtures import (
    WithLogger,
    WithTradingSchedule,
    ZiplineTestCase,
)
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


class SecurityListTestCase(WithLogger, WithTradingSchedule, ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(SecurityListTestCase, cls).init_class_fixtures()
        # this is ugly, but we need to create two different
        # TradingEnvironment/DataPortal pairs

        start = list(LEVERAGED_ETFS.keys())[0]
        end = pd.Timestamp('2015-02-17', tz='utc')
        cls.extra_knowledge_date = pd.Timestamp('2015-01-27', tz='utc')
        cls.trading_day_before_first_kd = pd.Timestamp('2015-01-23', tz='utc')
        symbols = ['AAPL', 'GOOG', 'BZQ', 'URTY', 'JFT']

        cls.env = cls.enter_class_context(tmp_trading_env(
            equities=pd.DataFrame.from_records([{
                'start_date': start,
                'end_date': end,
                'symbol': symbol
            } for symbol in symbols]),
        ))
        cls.sim_params = factory.create_simulation_parameters(
            start=start,
            num_days=4,
            trading_schedule=cls.trading_schedule
        )

        cls.sim_params2 = sp2 = factory.create_simulation_parameters(
            start=cls.trading_day_before_first_kd, num_days=4
        )

        cls.env2 = cls.enter_class_context(tmp_trading_env(
            equities=pd.DataFrame.from_records([{
                'start_date': sp2.period_start,
                'end_date': sp2.period_end,
                'symbol': symbol
            } for symbol in symbols]),
        ))

        cls.tempdir = cls.enter_class_context(tmp_dir())
        cls.tempdir2 = cls.enter_class_context(tmp_dir())

        cls.data_portal = create_data_portal(
            env=cls.env,
            tempdir=cls.tempdir,
            sim_params=cls.sim_params,
            sids=range(0, 5),
            trading_schedule=cls.trading_schedule,
        )

        cls.data_portal2 = create_data_portal(
            env=cls.env2,
            tempdir=cls.tempdir2,
            sim_params=cls.sim_params2,
            sids=range(0, 5),
            trading_schedule=cls.trading_schedule,
        )

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
                LEVERAGED_ETFS.keys())[0] + timedelta(days=7), num_days=5)

        data_portal = create_data_portal(
            self.env,
            self.tempdir,
            sim_params=sim_params,
            sids=range(0, 5),
            trading_schedule=self.trading_schedule,
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
        sim_params = factory.create_simulation_parameters(
            start=self.extra_knowledge_date,
            num_days=4,
        )
        equities = pd.DataFrame.from_records([{
            'symbol': 'BZQ',
            'start_date': sim_params.period_start,
            'end_date': sim_params.period_end,
        }])
        with TempDirectory() as new_tempdir, \
                security_list_copy(), \
                tmp_trading_env(equities=equities) as env:
            # add a delete statement removing bzq
            # write a new delete statement file to disk
            add_security_data([], ['BZQ'])

            data_portal = create_data_portal(
                env,
                new_tempdir,
                sim_params,
                range(0, 5),
                trading_schedule=self.trading_schedule,
            )

            algo = RestrictedAlgoWithoutCheck(
                symbol='BZQ', sim_params=sim_params, env=env
            )
            algo.run(data_portal)

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
