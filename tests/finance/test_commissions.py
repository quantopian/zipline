from datetime import timedelta
from textwrap import dedent

from zipline import TradingAlgorithm
from zipline.finance.commission import PerTrade, PerShare, PerDollar
from zipline.finance.order import Order
from zipline.finance.transaction import Transaction
from zipline.testing import ZiplineTestCase, trades_by_sid_to_dfs
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithSimParams,
    WithDataPortal
)
from zipline.utils import factory


class CommissionUnitTests(WithAssetFinder, ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = 1, 2

    def generate_order_and_txns(self):
        asset1 = self.asset_finder.retrieve_asset(1)

        # one order
        order = Order(dt=None, sid=asset1, amount=500)

        # three fills
        txn1 = Transaction(sid=asset1, amount=230, dt=None,
                           price=100, order_id=order.id)

        txn2 = Transaction(sid=asset1, amount=170, dt=None,
                           price=101, order_id=order.id)

        txn3 = Transaction(sid=asset1, amount=100, dt=None,
                           price=102, order_id=order.id)

        return order, [txn1, txn2, txn3]

    def test_per_trade(self):
        model = PerTrade(cost=10)

        order, txns = self.generate_order_and_txns()

        self.assertEqual(10, model.calculate(order, txns[0]))

        order.commission = 10

        self.assertEqual(0, model.calculate(order, txns[1]))
        self.assertEqual(0, model.calculate(order, txns[2]))

    def test_per_share_no_minimum(self):
        model = PerShare(cost=0.0075, min_trade_cost=None)

        order, txns = self.generate_order_and_txns()

        # make sure each commission is pro-rated
        self.assertAlmostEqual(1.725, model.calculate(order, txns[0]))
        self.assertAlmostEqual(1.275, model.calculate(order, txns[1]))
        self.assertAlmostEqual(0.75, model.calculate(order, txns[2]))

    def verify_per_share_commissions(self, model, commission_totals):
        order, txns = self.generate_order_and_txns()

        for i, commission_total in enumerate(commission_totals):
            order.commission += model.calculate(order, txns[i])
            self.assertAlmostEqual(commission_total, order.commission)
            order.filled += txns[i].amount

    def test_per_share_with_minimum(self):
        # minimum is met by the first trade
        self.verify_per_share_commissions(
            PerShare(cost=0.0075, min_trade_cost=1),
            [1.725, 3, 3.75]
        )

        # minimum is met by the second trade
        self.verify_per_share_commissions(
            PerShare(cost=0.0075, min_trade_cost=2.5),
            [2.5, 3, 3.75]
        )

        # minimum is met by the third trade
        self.verify_per_share_commissions(
            PerShare(cost=0.0075, min_trade_cost=3.5),
            [3.5, 3.5, 3.75]
        )

        # minimum is not met by any of the trades
        self.verify_per_share_commissions(
            PerShare(cost=0.0075, min_trade_cost=5.5),
            [5.5, 5.5, 5.5]
        )

    def test_per_dollar(self):
        model = PerDollar(cost=0.0015)

        order, txns = self.generate_order_and_txns()

        # make sure each commission is pro-rated
        self.assertAlmostEqual(34.5, model.calculate(order, txns[0]))
        self.assertAlmostEqual(25.755, model.calculate(order, txns[1]))
        self.assertAlmostEqual(15.3, model.calculate(order, txns[2]))


class CommissionAlgorithmTests(WithDataPortal, WithSimParams, ZiplineTestCase):
    # make sure order commissions are properly incremented

    sidint, = ASSET_FINDER_EQUITY_SIDS = (133,)

    code = dedent(
        """
        from zipline.api import (
            sid, order, set_slippage, slippage, FixedSlippage,
            set_commission, commission
        )

        def initialize(context):
            # for these tests, let us take out the entire bar with no price
            # impact
            set_slippage(slippage.VolumeShareSlippage(1.0, 0))

            {0}
            context.ordered = False


        def handle_data(context, data):
            if not context.ordered:
                order(sid(133), {1})
                context.ordered = True
        """,
    )

    @classmethod
    def make_equity_daily_bar_data(cls):
        num_days = len(cls.sim_params.sessions)

        return trades_by_sid_to_dfs(
            {
                cls.sidint: factory.create_trade_history(
                    cls.sidint,
                    [10.0] * num_days,
                    [100.0] * num_days,
                    timedelta(days=1),
                    cls.sim_params,
                    trading_calendar=cls.trading_calendar,
                ),
            },
            index=cls.sim_params.sessions,
        )

    def get_results(self, algo_code):
        algo = TradingAlgorithm(
            script=algo_code,
            env=self.env,
            sim_params=self.sim_params
        )

        return algo.run(self.data_portal)

    def test_per_trade(self):
        results = self.get_results(
            self.code.format("set_commission(commission.PerTrade(1))", 300)
        )

        # should be 3 fills at 100 shares apiece
        # one order split among 3 days, each copy of the order should have a
        # commission of one dollar
        for orders in results.orders[1:4]:
            self.assertEqual(1, orders[0]["commission"])

        self.verify_capital_used(results, [-1001, -1000, -1000])

    def test_per_share_no_minimum(self):
        results = self.get_results(
            self.code.format("set_commission(commission.PerShare(0.05, None))",
                             300)
        )

        # should be 3 fills at 100 shares apiece
        # one order split among 3 days, each fill generates an additional
        # 100 * 0.05 = $5 in commission
        for i, orders in enumerate(results.orders[1:4]):
            self.assertEqual((i + 1) * 5, orders[0]["commission"])

        self.verify_capital_used(results, [-1005, -1005, -1005])

    def test_per_share_with_minimum(self):
        # minimum hit by first trade
        results = self.get_results(
            self.code.format("set_commission(commission.PerShare(0.05, 3))",
                             300)
        )

        # commissions should be 5, 10, 15
        for i, orders in enumerate(results.orders[1:4]):
            self.assertEqual((i + 1) * 5, orders[0]["commission"])

        self.verify_capital_used(results, [-1005, -1005, -1005])

        # minimum hit by second trade
        results = self.get_results(
            self.code.format("set_commission(commission.PerShare(0.05, 8))",
                             300)
        )

        # commissions should be 8, 10, 15
        self.assertEqual(8, results.orders[1][0]["commission"])
        self.assertEqual(10, results.orders[2][0]["commission"])
        self.assertEqual(15, results.orders[3][0]["commission"])

        self.verify_capital_used(results, [-1008, -1002, -1005])

        # minimum hit by third trade
        results = self.get_results(
            self.code.format("set_commission(commission.PerShare(0.05, 12))",
                             300)
        )

        # commissions should be 12, 12, 15
        self.assertEqual(12, results.orders[1][0]["commission"])
        self.assertEqual(12, results.orders[2][0]["commission"])
        self.assertEqual(15, results.orders[3][0]["commission"])

        self.verify_capital_used(results, [-1012, -1000, -1003])

        # minimum never hit
        results = self.get_results(
            self.code.format("set_commission(commission.PerShare(0.05, 18))",
                             300)
        )

        # commissions should be 18, 18, 18
        self.assertEqual(18, results.orders[1][0]["commission"])
        self.assertEqual(18, results.orders[2][0]["commission"])
        self.assertEqual(18, results.orders[3][0]["commission"])

        self.verify_capital_used(results, [-1018, -1000, -1000])

    def test_per_dollar(self):
        results = self.get_results(
            self.code.format("set_commission(commission.PerDollar(0.01))", 300)
        )

        # should be 3 fills at 100 shares apiece, each fill is worth $1k, so
        # incremental commission of $1000 * 0.01 = $10

        # commissions should be $10, $20, $30
        for i, orders in enumerate(results.orders[1:4]):
            self.assertEqual((i + 1) * 10, orders[0]["commission"])

        self.verify_capital_used(results, [-1010, -1010, -1010])

    def verify_capital_used(self, results, values):
        self.assertEqual(values[0], results.capital_used[1])
        self.assertEqual(values[1], results.capital_used[2])
        self.assertEqual(values[2], results.capital_used[3])
