from textwrap import dedent

import pandas as pd
import pytest
from parameterized import parameterized

from zipline.assets import Equity, Future
from zipline.errors import IncompatibleCommissionModel
from zipline.finance.commission import (
    CommissionModel,
    EquityCommissionModel,
    FutureCommissionModel,
    PerContract,
    PerDollar,
    PerFutureTrade,
    PerShare,
    PerTrade,
)
from zipline.finance.order import Order
from zipline.finance.transaction import Transaction
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithMakeAlgo


@pytest.fixture(scope="class")
def set_test_commission_unit(request, with_asset_finder):
    ASSET_FINDER_COUNTRY_CODE = "??"

    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-12-29")

    equities = pd.DataFrame.from_dict(
        {
            1: {
                "symbol": "A",
                "start_date": START_DATE,
                "end_date": END_DATE + pd.Timedelta(days=1),
                "exchange": "TEST",
            },
            2: {
                "symbol": "B",
                "start_date": START_DATE,
                "end_date": END_DATE + pd.Timedelta(days=1),
                "exchange": "TEST",
            },
        },
        orient="index",
    )

    futures = pd.DataFrame(
        {
            "sid": [1000, 1001],
            "root_symbol": ["CL", "FV"],
            "symbol": ["CLF07", "FVF07"],
            "start_date": [START_DATE, START_DATE],
            "end_date": [END_DATE, END_DATE],
            "notice_date": [END_DATE, END_DATE],
            "expiration_date": [END_DATE, END_DATE],
            "multiplier": [500, 500],
            "exchange": ["CMES", "CMES"],
        }
    )

    exchange_names = [df["exchange"] for df in (futures, equities) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    request.cls.asset_finder = with_asset_finder(
        **dict(equities=equities, futures=futures, exchanges=exchanges)
    )


@pytest.mark.usefixtures("set_test_commission_unit")
class TestCommissionUnit:
    def generate_order_and_txns(self, sid, order_amount, fill_amounts):
        asset1 = self.asset_finder.retrieve_asset(sid)

        # one order
        order = Order(dt=None, asset=asset1, amount=order_amount)

        # three fills
        txn1 = Transaction(
            asset=asset1, amount=fill_amounts[0], dt=None, price=100, order_id=order.id
        )

        txn2 = Transaction(
            asset=asset1, amount=fill_amounts[1], dt=None, price=101, order_id=order.id
        )

        txn3 = Transaction(
            asset=asset1, amount=fill_amounts[2], dt=None, price=102, order_id=order.id
        )

        return order, [txn1, txn2, txn3]

    def verify_per_trade_commissions(
        self, model, expected_commission, sid, order_amount=None, fill_amounts=None
    ):
        fill_amounts = fill_amounts or [230, 170, 100]
        order_amount = order_amount or sum(fill_amounts)

        order, txns = self.generate_order_and_txns(
            sid,
            order_amount,
            fill_amounts,
        )

        assert expected_commission == model.calculate(order, txns[0])

        order.commission = expected_commission

        assert 0 == model.calculate(order, txns[1])
        assert 0 == model.calculate(order, txns[2])

    def test_allowed_asset_types(self):
        # Custom equities model.
        class MyEquitiesModel(EquityCommissionModel):
            def calculate(self, order, transaction):
                return 0

        assert MyEquitiesModel.allowed_asset_types == (Equity,)

        # Custom futures model.
        class MyFuturesModel(FutureCommissionModel):
            def calculate(self, order, transaction):
                return 0

        assert MyFuturesModel.allowed_asset_types == (Future,)

        # Custom model for both equities and futures.
        class MyMixedModel(EquityCommissionModel, FutureCommissionModel):
            def calculate(self, order, transaction):
                return 0

        assert MyMixedModel.allowed_asset_types == (Equity, Future)

        # Equivalent custom model for both equities and futures.
        class MyMixedModel(CommissionModel):
            def calculate(self, order, transaction):
                return 0

        assert MyMixedModel.allowed_asset_types == (Equity, Future)

        SomeType = type("SomeType", (object,), {})

        # A custom model that defines its own allowed types should take
        # precedence over the parent class definitions.
        class MyCustomModel(EquityCommissionModel, FutureCommissionModel):
            allowed_asset_types = (SomeType,)

            def calculate(self, order, transaction):
                return 0

        assert MyCustomModel.allowed_asset_types == (SomeType,)

    def test_per_trade(self):
        # Test per trade model for equities.
        model = PerTrade(cost=10)
        self.verify_per_trade_commissions(model, expected_commission=10, sid=1)

        # Test per trade model for futures.
        model = PerFutureTrade(cost=10)
        self.verify_per_trade_commissions(
            model,
            expected_commission=10,
            sid=1000,
        )

        # Test per trade model with custom costs per future symbol.
        model = PerFutureTrade(cost={"CL": 5, "FV": 10})
        self.verify_per_trade_commissions(
            model,
            expected_commission=5,
            sid=1000,
        )
        self.verify_per_trade_commissions(
            model,
            expected_commission=10,
            sid=1001,
        )

    def test_per_share_no_minimum(self):
        model = PerShare(cost=0.0075, min_trade_cost=None)

        fill_amounts = [230, 170, 100]
        order, txns = self.generate_order_and_txns(
            sid=1, order_amount=500, fill_amounts=fill_amounts
        )
        expected_commissions = [1.725, 1.275, 0.75]

        # make sure each commission is pro-rated
        for fill_amount, expected_commission, txn in zip(
            fill_amounts,
            expected_commissions,
            txns,
        ):

            commission = model.calculate(order, txn)
            assert round(abs(expected_commission - commission), 7) == 0
            order.filled += fill_amount
            order.commission += commission

    def test_per_share_shrinking_position(self):
        model = PerShare(cost=0.0075, min_trade_cost=None)

        fill_amounts = [-230, -170, -100]
        order, txns = self.generate_order_and_txns(
            sid=1, order_amount=-500, fill_amounts=fill_amounts
        )
        expected_commissions = [1.725, 1.275, 0.75]

        # make sure each commission is positive and pro-rated
        for fill_amount, expected_commission, txn in zip(
            fill_amounts, expected_commissions, txns
        ):

            commission = model.calculate(order, txn)
            assert round(abs(expected_commission - commission), 7) == 0
            order.filled += fill_amount
            order.commission += commission

    def verify_per_unit_commissions(
        self, model, commission_totals, sid, order_amount=None, fill_amounts=None
    ):
        fill_amounts = fill_amounts or [230, 170, 100]
        order_amount = order_amount or sum(fill_amounts)

        order, txns = self.generate_order_and_txns(
            sid,
            order_amount,
            fill_amounts,
        )

        for i, commission_total in enumerate(commission_totals):
            order.commission += model.calculate(order, txns[i])
            assert round(abs(commission_total - order.commission), 7) == 0
            order.filled += txns[i].amount

    def test_per_contract_no_minimum(self):
        # Note that the exchange fee is a one-time cost that is only applied to
        # the first fill of an order.
        #
        # The commission on the first fill is (230 * 0.01) + 0.3 = 2.6
        # The commission on the second fill is 170 * 0.01 = 1.7
        # The total after the second fill is 2.6 + 1.7 = 4.3
        # The commission on the third fill is 100 * 0.01 = 1.0
        # The total after the third fill is 5.3
        model = PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=None)
        self.verify_per_unit_commissions(
            model=model,
            commission_totals=[2.6, 4.3, 5.3],
            sid=1000,
            order_amount=500,
            fill_amounts=[230, 170, 100],
        )

        # Test using custom costs and fees.
        model = PerContract(
            cost={"CL": 0.01, "FV": 0.0075},
            exchange_fee={"CL": 0.3, "FV": 0.5},
            min_trade_cost=None,
        )
        self.verify_per_unit_commissions(model, [2.6, 4.3, 5.3], sid=1000)
        self.verify_per_unit_commissions(model, [2.225, 3.5, 4.25], sid=1001)

    def test_per_share_with_minimum(self):
        # minimum is met by the first trade
        self.verify_per_unit_commissions(
            PerShare(cost=0.0075, min_trade_cost=1),
            commission_totals=[1.725, 3, 3.75],
            sid=1,
        )

        # minimum is met by the second trade
        self.verify_per_unit_commissions(
            PerShare(cost=0.0075, min_trade_cost=2.5),
            commission_totals=[2.5, 3, 3.75],
            sid=1,
        )

        # minimum is met by the third trade
        self.verify_per_unit_commissions(
            PerShare(cost=0.0075, min_trade_cost=3.5),
            commission_totals=[3.5, 3.5, 3.75],
            sid=1,
        )

        # minimum is not met by any of the trades
        self.verify_per_unit_commissions(
            PerShare(cost=0.0075, min_trade_cost=5.5),
            commission_totals=[5.5, 5.5, 5.5],
            sid=1,
        )

    def test_per_contract_with_minimum(self):
        # Minimum is met by the first trade.
        self.verify_per_unit_commissions(
            PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=1),
            commission_totals=[2.6, 4.3, 5.3],
            sid=1000,
        )

        # Minimum is met by the second trade.
        self.verify_per_unit_commissions(
            PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=3),
            commission_totals=[3.0, 4.3, 5.3],
            sid=1000,
        )

        # Minimum is met by the third trade.
        self.verify_per_unit_commissions(
            PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=5),
            commission_totals=[5.0, 5.0, 5.3],
            sid=1000,
        )

        # Minimum is not met by any of the trades.
        self.verify_per_unit_commissions(
            PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=7),
            commission_totals=[7.0, 7.0, 7.0],
            sid=1000,
        )

    def test_per_dollar(self):
        model = PerDollar(cost=0.0015)

        order, txns = self.generate_order_and_txns(
            sid=1,
            order_amount=500,
            fill_amounts=[230, 170, 100],
        )

        # make sure each commission is pro-rated
        assert round(abs(34.5 - model.calculate(order, txns[0])), 7) == 0
        assert round(abs(25.755 - model.calculate(order, txns[1])), 7) == 0
        assert round(abs(15.3 - model.calculate(order, txns[2])), 7) == 0


class CommissionAlgorithmTests(WithMakeAlgo, ZiplineTestCase):
    # make sure order commissions are properly incremented
    SIM_PARAMS_DATA_FREQUENCY = "daily"

    # NOTE: This is required to use futures data with WithDataPortal right now.
    DATA_PORTAL_USE_MINUTE_DATA = True
    (sidint,) = ASSET_FINDER_EQUITY_SIDS = (133,)

    code = dedent(
        """
        from zipline.api import (
            sid, order, set_slippage, slippage, FixedSlippage,
            set_commission, commission
        )

        def initialize(context):
            # for these tests, let us take out the entire bar with no price
            # impact
            set_slippage(
                us_equities=slippage.VolumeShareSlippage(1.0, 0),
                us_futures=slippage.VolumeShareSlippage(1.0, 0),
            )

            {commission}
            context.ordered = False


        def handle_data(context, data):
            if not context.ordered:
                order(sid({sid}), {amount})
                context.ordered = True
        """,
    )

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame(
            {
                "sid": [1000, 1001],
                "root_symbol": ["CL", "FV"],
                "symbol": ["CLF07", "FVF07"],
                "start_date": [cls.START_DATE, cls.START_DATE],
                "end_date": [cls.END_DATE, cls.END_DATE],
                "notice_date": [cls.END_DATE, cls.END_DATE],
                "expiration_date": [cls.END_DATE, cls.END_DATE],
                "multiplier": [500, 500],
                "exchange": ["CMES", "CMES"],
            }
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        sessions = cls.trading_calendar.sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )
        for sid in sids:
            yield sid, pd.DataFrame(
                index=sessions,
                data={
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 100.0,
                },
            )

    def get_results(self, algo_code):
        return self.run_algorithm(script=algo_code)

    def test_per_trade(self):
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerTrade(1))",
                sid=133,
                amount=300,
            )
        )

        # should be 3 fills at 100 shares apiece
        # one order split among 3 days, each copy of the order should have a
        # commission of one dollar
        for orders in results.orders[1:4]:
            assert 1 == orders[0]["commission"]

        self.verify_capital_used(results, [-1001, -1000, -1000])

    def test_futures_per_trade(self):
        results = self.get_results(
            self.code.format(
                commission=("set_commission(us_futures=commission.PerFutureTrade(1))"),
                sid=1000,
                amount=10,
            )
        )

        # The capital used is only -1.0 (the commission cost) because no
        # capital is actually spent to enter into a long position on a futures
        # contract.
        assert results.orders[1][0]["commission"] == 1.0
        assert results.capital_used[1] == -1.0

    def test_per_share_no_minimum(self):
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerShare(0.05, None))",
                sid=133,
                amount=300,
            )
        )

        # should be 3 fills at 100 shares apiece
        # one order split among 3 days, each fill generates an additional
        # 100 * 0.05 = $5 in commission
        for i, orders in enumerate(results.orders[1:4]):
            assert (i + 1) * 5 == orders[0]["commission"]

        self.verify_capital_used(results, [-1005, -1005, -1005])

    def test_per_share_with_minimum(self):
        # minimum hit by first trade
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerShare(0.05, 3))",
                sid=133,
                amount=300,
            )
        )

        # commissions should be 5, 10, 15
        for i, orders in enumerate(results.orders[1:4]):
            assert (i + 1) * 5 == orders[0]["commission"]

        self.verify_capital_used(results, [-1005, -1005, -1005])

        # minimum hit by second trade
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerShare(0.05, 8))",
                sid=133,
                amount=300,
            )
        )

        # commissions should be 8, 10, 15
        assert 8 == results.orders[1][0]["commission"]
        assert 10 == results.orders[2][0]["commission"]
        assert 15 == results.orders[3][0]["commission"]

        self.verify_capital_used(results, [-1008, -1002, -1005])

        # minimum hit by third trade
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerShare(0.05, 12))",
                sid=133,
                amount=300,
            )
        )

        # commissions should be 12, 12, 15
        assert 12 == results.orders[1][0]["commission"]
        assert 12 == results.orders[2][0]["commission"]
        assert 15 == results.orders[3][0]["commission"]

        self.verify_capital_used(results, [-1012, -1000, -1003])

        # minimum never hit
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerShare(0.05, 18))",
                sid=133,
                amount=300,
            )
        )

        # commissions should be 18, 18, 18
        assert 18 == results.orders[1][0]["commission"]
        assert 18 == results.orders[2][0]["commission"]
        assert 18 == results.orders[3][0]["commission"]

        self.verify_capital_used(results, [-1018, -1000, -1000])

    @parameterized.expand(
        [
            # The commission is (10 * 0.05) + 1.3 = 1.8, and the capital used is
            # the same as the commission cost because no capital is actually spent
            # to enter into a long position on a futures contract.
            (None, 1.8),
            # Minimum hit by first trade.
            (1, 1.8),
            # Minimum not hit by first trade, so use the minimum.
            (3, 3.0),
        ]
    )
    def test_per_contract(self, min_trade_cost, expected_commission):
        results = self.get_results(
            self.code.format(
                commission=(
                    "set_commission(us_futures=commission.PerContract("
                    "cost=0.05, exchange_fee=1.3, min_trade_cost={}))"
                ).format(min_trade_cost),
                sid=1000,
                amount=10,
            ),
        )

        assert results.orders[1][0]["commission"] == expected_commission
        assert results.capital_used[1] == -expected_commission

    def test_per_dollar(self):
        results = self.get_results(
            self.code.format(
                commission="set_commission(commission.PerDollar(0.01))",
                sid=133,
                amount=300,
            )
        )

        # should be 3 fills at 100 shares apiece, each fill is worth $1k, so
        # incremental commission of $1000 * 0.01 = $10

        # commissions should be $10, $20, $30
        for i, orders in enumerate(results.orders[1:4]):
            assert (i + 1) * 10 == orders[0]["commission"]

        self.verify_capital_used(results, [-1010, -1010, -1010])

    def test_incorrectly_set_futures_model(self):
        with pytest.raises(IncompatibleCommissionModel):
            # Passing a futures commission model as the first argument, which
            # is for setting equity models, should fail.
            self.get_results(
                self.code.format(
                    commission="set_commission(commission.PerContract(0, 0))",
                    sid=1000,
                    amount=10,
                )
            )

    def verify_capital_used(self, results, values):
        assert values[0] == results.capital_used[1]
        assert values[1] == results.capital_used[2]
        assert values[2] == results.capital_used[3]
