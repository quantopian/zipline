from datetime import timedelta
from textwrap import dedent

from nose_parameterized import parameterized
from pandas import DataFrame

from zipline import TradingAlgorithm
from zipline.errors import IncompatibleCommissionModel
from zipline.finance.commission import (
    PerContract,
    PerDollar,
    PerFutureTrade,
    PerShare,
    PerTrade,
)
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

    @classmethod
    def make_futures_info(cls):
        return DataFrame({
            'sid': [1000, 1001],
            'root_symbol': ['CL', 'FV'],
            'symbol': ['CLF07', 'FVF07'],
            'start_date': [cls.START_DATE, cls.START_DATE],
            'end_date': [cls.END_DATE, cls.END_DATE],
            'notice_date': [cls.END_DATE, cls.END_DATE],
            'expiration_date': [cls.END_DATE, cls.END_DATE],
            'multiplier': [500, 500],
            'exchange': ['CME', 'CME'],
        })

    def generate_order_and_txns(self, sid, order_amount, fill_amounts):
        asset1 = self.asset_finder.retrieve_asset(sid)

        # one order
        order = Order(dt=None, asset=asset1, amount=order_amount)

        # three fills
        txn1 = Transaction(asset=asset1, amount=fill_amounts[0], dt=None,
                           price=100, order_id=order.id)

        txn2 = Transaction(asset=asset1, amount=fill_amounts[1], dt=None,
                           price=101, order_id=order.id)

        txn3 = Transaction(asset=asset1, amount=fill_amounts[2], dt=None,
                           price=102, order_id=order.id)

        return order, [txn1, txn2, txn3]

    def verify_per_trade_commissions(self,
                                     model,
                                     expected_commission,
                                     sid,
                                     order_amount=None,
                                     fill_amounts=None):
        fill_amounts = fill_amounts or [230, 170, 100]
        order_amount = order_amount or sum(fill_amounts)

        order, txns = self.generate_order_and_txns(
            sid, order_amount, fill_amounts,
        )

        self.assertEqual(expected_commission, model.calculate(order, txns[0]))

        order.commission = expected_commission

        self.assertEqual(0, model.calculate(order, txns[1]))
        self.assertEqual(0, model.calculate(order, txns[2]))

    def test_per_trade(self):
        # Test per trade model for equities.
        model = PerTrade(cost=10)
        self.verify_per_trade_commissions(model, expected_commission=10, sid=1)

        # Test per trade model for futures.
        model = PerFutureTrade(cost=10)
        self.verify_per_trade_commissions(
            model, expected_commission=10, sid=1000,
        )

        # Test per trade model with custom costs per future symbol.
        model = PerFutureTrade(cost={'CL': 5, 'FV': 10})
        self.verify_per_trade_commissions(
            model, expected_commission=5, sid=1000,
        )
        self.verify_per_trade_commissions(
            model, expected_commission=10, sid=1001,
        )

    def test_per_share_no_minimum(self):
        model = PerShare(cost=0.0075, min_trade_cost=None)

        order, txns = self.generate_order_and_txns(
            sid=1, order_amount=500, fill_amounts=[230, 170, 100],
        )

        # make sure each commission is pro-rated
        self.assertAlmostEqual(1.725, model.calculate(order, txns[0]))
        self.assertAlmostEqual(1.275, model.calculate(order, txns[1]))
        self.assertAlmostEqual(0.75, model.calculate(order, txns[2]))

    def verify_per_unit_commissions(self,
                                    model,
                                    commission_totals,
                                    sid,
                                    order_amount=None,
                                    fill_amounts=None):
        fill_amounts = fill_amounts or [230, 170, 100]
        order_amount = order_amount or sum(fill_amounts)

        order, txns = self.generate_order_and_txns(
            sid, order_amount, fill_amounts,
        )

        for i, commission_total in enumerate(commission_totals):
            order.commission += model.calculate(order, txns[i])
            self.assertAlmostEqual(commission_total, order.commission)
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
            cost={'CL': 0.01, 'FV': 0.0075},
            exchange_fee={'CL': 0.3, 'FV': 0.5},
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
            PerContract(cost=.01, exchange_fee=0.3, min_trade_cost=1),
            commission_totals=[2.6, 4.3, 5.3],
            sid=1000,
        )

        # Minimum is met by the second trade.
        self.verify_per_unit_commissions(
            PerContract(cost=.01, exchange_fee=0.3, min_trade_cost=3),
            commission_totals=[3.0, 4.3, 5.3],
            sid=1000,
        )

        # Minimum is met by the third trade.
        self.verify_per_unit_commissions(
            PerContract(cost=.01, exchange_fee=0.3, min_trade_cost=5),
            commission_totals=[5.0, 5.0, 5.3],
            sid=1000,
        )

        # Minimum is not met by any of the trades.
        self.verify_per_unit_commissions(
            PerContract(cost=.01, exchange_fee=0.3, min_trade_cost=7),
            commission_totals=[7.0, 7.0, 7.0],
            sid=1000,
        )

    def test_per_dollar(self):
        model = PerDollar(cost=0.0015)

        order, txns = self.generate_order_and_txns(
            sid=1, order_amount=500, fill_amounts=[230, 170, 100],
        )

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
        return DataFrame({
            'sid': [1000, 1001],
            'root_symbol': ['CL', 'FV'],
            'symbol': ['CLF07', 'FVF07'],
            'start_date': [cls.START_DATE, cls.START_DATE],
            'end_date': [cls.END_DATE, cls.END_DATE],
            'notice_date': [cls.END_DATE, cls.END_DATE],
            'expiration_date': [cls.END_DATE, cls.END_DATE],
            'multiplier': [500, 500],
            'exchange': ['CME', 'CME'],
        })

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
            self.assertEqual(1, orders[0]["commission"])

        self.verify_capital_used(results, [-1001, -1000, -1000])

    def test_futures_per_trade(self):
        results = self.get_results(
            self.code.format(
                commission=(
                    'set_commission(us_futures=commission.PerFutureTrade(1))'
                ),
                sid=1000,
                amount=10,
            )
        )

        # The capital used is only -1.0 (the commission cost) because no
        # capital is actually spent to enter into a long position on a futures
        # contract.
        self.assertEqual(results.orders[1][0]['commission'], 1.0)
        self.assertEqual(results.capital_used[1], -1.0)

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
            self.assertEqual((i + 1) * 5, orders[0]["commission"])

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
            self.assertEqual((i + 1) * 5, orders[0]["commission"])

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
        self.assertEqual(8, results.orders[1][0]["commission"])
        self.assertEqual(10, results.orders[2][0]["commission"])
        self.assertEqual(15, results.orders[3][0]["commission"])

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
        self.assertEqual(12, results.orders[1][0]["commission"])
        self.assertEqual(12, results.orders[2][0]["commission"])
        self.assertEqual(15, results.orders[3][0]["commission"])

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
        self.assertEqual(18, results.orders[1][0]["commission"])
        self.assertEqual(18, results.orders[2][0]["commission"])
        self.assertEqual(18, results.orders[3][0]["commission"])

        self.verify_capital_used(results, [-1018, -1000, -1000])

    @parameterized.expand([
        # The commission is (10 * 0.05) + 1.3 = 1.8, and the capital used is
        # the same as the commission cost because no capital is actually spent
        # to enter into a long position on a futures contract.
        (None, 1.8),
        # Minimum hit by first trade.
        (1, 1.8),
        # Minimum not hit by first trade, so use the minimum.
        (3, 3.0),
    ])
    def test_per_contract(self, min_trade_cost, expected_commission):
        results = self.get_results(
            self.code.format(
                commission=(
                    'set_commission(us_futures=commission.PerContract('
                    'cost=0.05, exchange_fee=1.3, min_trade_cost={}))'
                ).format(min_trade_cost),
                sid=1000,
                amount=10,
            ),
        )

        self.assertEqual(
            results.orders[1][0]['commission'], expected_commission,
        )
        self.assertEqual(results.capital_used[1], -expected_commission)

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
            self.assertEqual((i + 1) * 10, orders[0]["commission"])

        self.verify_capital_used(results, [-1010, -1010, -1010])

    def test_incorrectly_set_futures_model(self):
        with self.assertRaises(IncompatibleCommissionModel):
            # Passing a futures commission model as the first argument, which
            # is for setting equity models, should fail.
            self.get_results(
                self.code.format(
                    commission='set_commission(commission.PerContract(0, 0))',
                    sid=1000,
                    amount=10,
                )
            )

    def verify_capital_used(self, results, values):
        self.assertEqual(values[0], results.capital_used[1])
        self.assertEqual(values[1], results.capital_used[2])
        self.assertEqual(values[2], results.capital_used[3])
