from datetime import timedelta
import pandas as pd
from parameterized import parameterized

from zipline.algorithm import TradingAlgorithm
from zipline.errors import TradingControlViolation
from zipline.testing import (
    add_security_data,
    security_list_copy,
)
from zipline.testing.fixtures import (
    WithMakeAlgo,
    ZiplineTestCase,
)
from zipline.utils import factory
from zipline.utils.security_list import (
    SecurityListSet,
    load_from_directory,
)
import pytest

LEVERAGED_ETFS = load_from_directory("leveraged_etf_list")


class RestrictedAlgoWithCheck(TradingAlgorithm):
    def initialize(self, symbol):
        self.rl = SecurityListSet(self.get_datetime, self.asset_finder)
        self.set_asset_restrictions(self.rl.restrict_leveraged_etfs)
        self.order_count = 0
        self.sid = self.symbol(symbol)

    def handle_data(self, data):
        if not self.order_count:
            if self.sid not in self.rl.leveraged_etf_list.current_securities(
                self.get_datetime()
            ):
                self.order(self.sid, 100)
                self.order_count += 1


class RestrictedAlgoWithoutCheck(TradingAlgorithm):
    def initialize(self, symbol):
        self.rl = SecurityListSet(self.get_datetime, self.asset_finder)
        self.set_asset_restrictions(self.rl.restrict_leveraged_etfs)
        self.order_count = 0
        self.sid = self.symbol(symbol)

    def handle_data(self, data):
        self.order(self.sid, 100)
        self.order_count += 1


class RestrictedAlgoWithoutCheckSetDoNotOrderList(TradingAlgorithm):
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
        self.set_asset_restrictions(self.rl.restrict_leveraged_etfs)
        self.order_count = 0
        self.sid = self.symbol(symbol)
        self.found = False

    def handle_data(self, data):
        for stock in self.rl.leveraged_etf_list.current_securities(self.get_datetime()):
            if stock == self.sid:
                self.found = True


class SecurityListTestCase(WithMakeAlgo, ZiplineTestCase):
    # XXX: This suite uses way more than it probably needs.
    START_DATE = pd.Timestamp("2002-01-03")
    assert (
        START_DATE == sorted(list(LEVERAGED_ETFS.keys()))[0]
    ), "START_DATE should match start of LEVERAGED_ETF data."

    END_DATE = pd.Timestamp("2015-02-17")

    extra_knowledge_date = pd.Timestamp("2015-01-27")
    trading_day_before_first_kd = pd.Timestamp("2015-01-23")

    SIM_PARAMS_END = pd.Timestamp("2002-01-08")

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    ASSET_FINDER_EQUITY_SIDS = (1, 2, 3, 4, 5)
    ASSET_FINDER_EQUITY_SYMBOLS = ("AAPL", "GOOG", "BZQ", "URTY", "JFT")

    def test_iterate_over_restricted_list(self):
        algo = self.make_algo(
            algo_class=IterateRLAlgo,
            symbol="BZQ",
        )
        algo.run()
        assert algo.found

    def test_security_list(self):
        # set the knowledge date to the first day of the
        # leveraged etf knowledge date.
        def get_datetime():
            return self.START_DATE

        rl = SecurityListSet(get_datetime, self.asset_finder)
        # assert that a sample from the leveraged list are in restricted
        should_exist = [
            asset.sid
            for asset in [
                self.asset_finder.lookup_symbol(
                    symbol, as_of_date=self.extra_knowledge_date
                )
                for symbol in ["BZQ", "URTY", "JFT"]
            ]
        ]
        for sid in should_exist:
            assert sid in rl.leveraged_etf_list.current_securities(get_datetime())

        # assert that a sample of allowed stocks are not in restricted
        shouldnt_exist = [
            asset.sid
            for asset in [
                self.asset_finder.lookup_symbol(
                    symbol, as_of_date=self.extra_knowledge_date
                )
                for symbol in ["AAPL", "GOOG"]
            ]
        ]
        for sid in shouldnt_exist:
            assert sid not in rl.leveraged_etf_list.current_securities(get_datetime())

    def test_security_add(self):
        def get_datetime():
            return pd.Timestamp("2015-01-27")

        with security_list_copy():
            add_security_data(["AAPL", "GOOG"], [])
            rl = SecurityListSet(get_datetime, self.asset_finder)
            should_exist = [
                asset.sid
                for asset in [
                    self.asset_finder.lookup_symbol(
                        symbol, as_of_date=self.extra_knowledge_date
                    )
                    for symbol in ["AAPL", "GOOG", "BZQ", "URTY"]
                ]
            ]
            for sid in should_exist:
                assert sid in rl.leveraged_etf_list.current_securities(get_datetime())

    def test_security_add_delete(self):
        with security_list_copy():

            def get_datetime():
                return pd.Timestamp("2015-01-27")

            rl = SecurityListSet(get_datetime, self.asset_finder)
            assert "BZQ" not in rl.leveraged_etf_list.current_securities(get_datetime())
            assert "URTY" not in rl.leveraged_etf_list.current_securities(
                get_datetime()
            )

    def test_algo_without_rl_violation_via_check(self):
        self.run_algorithm(algo_class=RestrictedAlgoWithCheck, symbol="BZQ")

    def test_algo_without_rl_violation(self):
        self.run_algorithm(
            algo_class=RestrictedAlgoWithoutCheck,
            symbol="AAPL",
        )

    @parameterized.expand(
        [
            (
                "using_set_do_not_order_list",
                RestrictedAlgoWithoutCheckSetDoNotOrderList,
            ),
            ("using_set_restrictions", RestrictedAlgoWithoutCheck),
        ]
    )
    def test_algo_with_rl_violation(self, name, algo_class):
        algo = self.make_algo(algo_class=algo_class, symbol="BZQ")
        with pytest.raises(TradingControlViolation) as ctx:
            algo.run()

        self.check_algo_exception(algo, ctx, 0)

        # repeat with a symbol from a different lookup date
        algo = self.make_algo(
            algo_class=RestrictedAlgoWithoutCheck,
            symbol="JFT",
        )

        with pytest.raises(TradingControlViolation) as ctx:
            algo.run()

        self.check_algo_exception(algo, ctx, 0)

    def test_algo_with_rl_violation_after_knowledge_date(self):
        start = self.START_DATE + timedelta(days=7)
        end = start + self.trading_calendar.day * 4
        algo = self.make_algo(
            algo_class=RestrictedAlgoWithoutCheck,
            symbol="BZQ",
            sim_params=self.make_simparams(
                start_session=start,
                end_session=end,
            ),
        )

        with pytest.raises(TradingControlViolation) as ctx:
            algo.run()

        self.check_algo_exception(algo, ctx, 0)

    def test_algo_with_rl_violation_cumulative(self):
        """
        Add a new restriction, run a test long after both
        knowledge dates, make sure stock from original restriction
        set is still disallowed.
        """
        sim_params = factory.create_simulation_parameters(
            start=self.START_DATE + timedelta(days=7), num_days=4
        )

        with security_list_copy():
            add_security_data(["AAPL"], [])
            algo = self.make_algo(
                algo_class=RestrictedAlgoWithoutCheck,
                symbol="BZQ",
                sim_params=sim_params,
            )
            with pytest.raises(TradingControlViolation) as ctx:
                algo.run()

            self.check_algo_exception(algo, ctx, 0)

    def test_algo_without_rl_violation_after_delete(self):
        sim_params = factory.create_simulation_parameters(
            start=self.extra_knowledge_date,
            num_days=4,
        )

        with security_list_copy():
            # add a delete statement removing bzq
            # write a new delete statement file to disk
            add_security_data([], ["BZQ"])

            algo = self.make_algo(
                algo_class=RestrictedAlgoWithoutCheck,
                symbol="BZQ",
                sim_params=sim_params,
            )
            algo.run()

    def test_algo_with_rl_violation_after_add(self):
        sim_params = factory.create_simulation_parameters(
            start=self.trading_day_before_first_kd,
            num_days=4,
        )
        with security_list_copy():
            add_security_data(["AAPL"], [])

            algo = self.make_algo(
                algo_class=RestrictedAlgoWithoutCheck,
                symbol="AAPL",
                sim_params=sim_params,
            )
            with pytest.raises(TradingControlViolation) as ctx:
                algo.run()

            self.check_algo_exception(algo, ctx, 2)

    def check_algo_exception(self, algo, ctx, expected_order_count):
        assert algo.order_count == expected_order_count
        assert TradingControlViolation == ctx.type
        exc_msg = str(ctx.value)
        assert "RestrictedListOrder" in exc_msg
