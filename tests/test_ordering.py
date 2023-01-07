from parameterized import parameterized
import pandas as pd

from zipline.algorithm import TradingAlgorithm
import zipline.api as api
import zipline.errors as ze
from zipline.finance.execution import StopLimitOrder
import zipline.testing.fixtures as zf
from zipline.testing.predicates import assert_equal
import zipline.test_algorithms as zta
import pytest


class TestOrderMethods(
    zf.WithConstantEquityMinuteBarData,
    zf.WithConstantFutureMinuteBarData,
    zf.WithMakeAlgo,
    zf.ZiplineTestCase,
):
    #     January 2006
    # Su Mo Tu We Th Fr Sa
    #  1  2  3  4  5  6  7
    #  8  9 10 11 12 13 14
    # 15 16 17 18 19 20 21
    # 22 23 24 25 26 27 28
    # 29 30 31
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-06")
    SIM_PARAMS_START_DATE = pd.Timestamp("2006-01-04")

    ASSET_FINDER_EQUITY_SIDS = (1,)

    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True

    EQUITY_MINUTE_CONSTANT_LOW = 2.0
    EQUITY_MINUTE_CONSTANT_OPEN = 2.0
    EQUITY_MINUTE_CONSTANT_CLOSE = 2.0
    EQUITY_MINUTE_CONSTANT_HIGH = 2.0
    EQUITY_MINUTE_CONSTANT_VOLUME = 10000.0

    FUTURE_MINUTE_CONSTANT_LOW = 2.0
    FUTURE_MINUTE_CONSTANT_OPEN = 2.0
    FUTURE_MINUTE_CONSTANT_CLOSE = 2.0
    FUTURE_MINUTE_CONSTANT_HIGH = 2.0
    FUTURE_MINUTE_CONSTANT_VOLUME = 10000.0

    SIM_PARAMS_CAPITAL_BASE = 10000

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {2: {"multiplier": 10, "symbol": "F", "exchange": "TEST"}}, orient="index"
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestOrderMethods, cls).init_class_fixtures()
        cls.EQUITY = cls.asset_finder.retrieve_asset(1)
        cls.FUTURE = cls.asset_finder.retrieve_asset(2)

    @parameterized.expand(
        [
            ("order", 1),
            ("order_value", 1000),
            ("order_target", 1),
            ("order_target_value", 1000),
            ("order_percent", 1),
            ("order_target_percent", 1),
        ]
    )
    def test_cannot_order_in_before_trading_start(self, order_method, amount):
        algotext = """
from zipline.api import sid, {order_func}

def initialize(context):
    context.asset = sid(1)

def before_trading_start(context, data):
    {order_func}(context.asset, {arg})
     """.format(
            order_func=order_method, arg=amount
        )

        algo = self.make_algo(script=algotext)
        with pytest.raises(ze.OrderInBeforeTradingStart):
            algo.run()

    @parameterized.expand(
        [
            # These should all be orders for the same amount.
            ("order", 5000),  # 5000 shares times $2 per share
            ("order_value", 10000),  # $10000
            ("order_percent", 1),  # 100% on a $10000 capital base.
        ]
    )
    def test_order_equity_non_targeted(self, order_method, amount):
        # Every day, place an order for $10000 worth of sid(1)
        algotext = """
import zipline.api as api

def initialize(context):
    api.set_slippage(api.slippage.FixedSlippage(spread=0.0))
    api.set_commission(api.commission.PerShare(0))

    context.equity = api.sid(1)

    api.schedule_function(
        func=do_order,
        date_rule=api.date_rules.every_day(),
        time_rule=api.time_rules.market_open(),
    )

def do_order(context, data):
    context.ordered = True
    api.{order_func}(context.equity, {arg})
     """.format(
            order_func=order_method, arg=amount
        )
        result = self.run_algorithm(script=algotext)

        for orders in result.orders.values:
            assert_equal(len(orders), 1)
            assert_equal(orders[0]["amount"], 5000)
            assert_equal(orders[0]["sid"], self.EQUITY)

        for i, positions in enumerate(result.positions.values, start=1):
            assert_equal(len(positions), 1)
            assert_equal(positions[0]["amount"], 5000.0 * i)
            assert_equal(positions[0]["sid"], self.EQUITY)

    @parameterized.expand(
        [
            # These should all be orders for the same amount.
            ("order_target", 5000),  # 5000 shares times $2 per share
            ("order_target_value", 10000),  # $10000
            ("order_target_percent", 1),  # 100% on a $10000 capital base.
        ]
    )
    def test_order_equity_targeted(self, order_method, amount):
        # Every day, place an order for a target of $10000 worth of sid(1).
        # With no commissions or slippage, we should only place one order.
        algotext = """
import zipline.api as api

def initialize(context):
    api.set_slippage(api.slippage.FixedSlippage(spread=0.0))
    api.set_commission(api.commission.PerShare(0))

    context.equity = api.sid(1)

    api.schedule_function(
        func=do_order,
        date_rule=api.date_rules.every_day(),
        time_rule=api.time_rules.market_open(),
    )

def do_order(context, data):
    context.ordered = True
    api.{order_func}(context.equity, {arg})
     """.format(
            order_func=order_method, arg=amount
        )

        result = self.run_algorithm(script=algotext)

        assert_equal([len(ords) for ords in result.orders], [1, 0, 0, 0])
        order = result.orders.iloc[0][0]
        assert_equal(order["amount"], 5000)
        assert_equal(order["sid"], self.EQUITY)

        for positions in result.positions.values:
            assert_equal(len(positions), 1)
            assert_equal(positions[0]["amount"], 5000.0)
            assert_equal(positions[0]["sid"], self.EQUITY)

    @parameterized.expand(
        [
            # These should all be orders for the same amount.
            ("order", 500),  # 500 contracts times $2 per contract * 10x
            # multiplier.
            ("order_value", 10000),  # $10000
            ("order_percent", 1),  # 100% on a $10000 capital base.
        ]
    )
    def test_order_future_non_targeted(self, order_method, amount):
        # Every day, place an order for $10000 worth of sid(2)
        algotext = """
import zipline.api as api

def initialize(context):
    api.set_slippage(us_futures=api.slippage.FixedSlippage(spread=0.0))
    api.set_commission(us_futures=api.commission.PerTrade(0.0))

    context.future = api.sid(2)

    api.schedule_function(
        func=do_order,
        date_rule=api.date_rules.every_day(),
        time_rule=api.time_rules.market_open(),
    )

def do_order(context, data):
    context.ordered = True
    api.{order_func}(context.future, {arg})
     """.format(
            order_func=order_method, arg=amount
        )
        result = self.run_algorithm(script=algotext)

        for orders in result.orders.values:
            assert_equal(len(orders), 1)
            assert_equal(orders[0]["amount"], 500)
            assert_equal(orders[0]["sid"], self.FUTURE)

        for i, positions in enumerate(result.positions.values, start=1):
            assert_equal(len(positions), 1)
            assert_equal(positions[0]["amount"], 500.0 * i)
            assert_equal(positions[0]["sid"], self.FUTURE)

    @parameterized.expand(
        [
            # These should all be orders targeting the same amount.
            ("order_target", 500),  # 500 contracts * $2 per contract * 10x
            # multiplier.
            ("order_target_value", 10000),  # $10000
            ("order_target_percent", 1),  # 100% on a $10000 capital base.
        ]
    )
    def test_order_future_targeted(self, order_method, amount):
        # Every day, place an order for a target of $10000 worth of sid(2).
        # With no commissions or slippage, we should only place one order.
        algotext = """
import zipline.api as api

def initialize(context):
    api.set_slippage(us_futures=api.slippage.FixedSlippage(spread=0.0))
    api.set_commission(us_futures=api.commission.PerTrade(0.0))

    context.future = api.sid(2)

    api.schedule_function(
        func=do_order,
        date_rule=api.date_rules.every_day(),
        time_rule=api.time_rules.market_open(),
    )

def do_order(context, data):
    context.ordered = True
    api.{order_func}(context.future, {arg})
     """.format(
            order_func=order_method, arg=amount
        )

        result = self.run_algorithm(script=algotext)

        # We should get one order on the first day.
        assert_equal([len(ords) for ords in result.orders], [1, 0, 0, 0])
        order = result.orders.iloc[0][0]
        assert_equal(order["amount"], 500)
        assert_equal(order["sid"], self.FUTURE)

        # Our position at the end of each day should be worth $10,000.
        for positions in result.positions.values:
            assert_equal(len(positions), 1)
            assert_equal(positions[0]["amount"], 500.0)
            assert_equal(positions[0]["sid"], self.FUTURE)

    @parameterized.expand(
        [
            (api.order, 5000),
            (api.order_value, 10000),
            (api.order_percent, 1.0),
            (api.order_target, 5000),
            (api.order_target_value, 10000),
            (api.order_target_percent, 1.0),
        ]
    )
    def test_order_method_style_forwarding(self, order_method, order_param):
        # Test that we correctly forward values passed via `style` to Order
        # objects.
        def initialize(context):
            api.schedule_function(
                func=do_order,
                date_rule=api.date_rules.every_day(),
                time_rule=api.time_rules.market_open(),
            )

        def do_order(context, data):
            assert len(context.portfolio.positions.keys()) == 0

            order_method(
                self.EQUITY,
                order_param,
                style=StopLimitOrder(10, 10, asset=self.EQUITY),
            )

            assert len(context.blotter.open_orders[self.EQUITY]) == 1
            result = context.blotter.open_orders[self.EQUITY][0]
            assert result.limit == 10
            assert result.stop == 10

        # We only need to run for a single day for this test.
        self.run_algorithm(
            initialize=initialize,
            sim_params=self.sim_params.create_new(
                start_session=self.END_DATE,
                end_session=self.END_DATE,
            ),
        )


class TestOrderMethodsDailyFrequency(zf.WithMakeAlgo, zf.ZiplineTestCase):
    #     January 2006
    # Su Mo Tu We Th Fr Sa
    #  1  2  3  4  5  6  7
    #  8  9 10 11 12 13 14
    # 15 16 17 18 19 20 21
    # 22 23 24 25 26 27 28
    # 29 30 31
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-06")
    SIM_PARAMS_START_DATE = pd.Timestamp("2006-01-04")
    ASSET_FINDER_EQUITY_SIDS = (1,)

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_invalid_order_parameters(self):
        self.run_algorithm(
            algo_class=zta.InvalidOrderAlgorithm,
            sids=[1],
        )

    def test_cant_order_in_initialize(self):
        algotext = """
from zipline.api import (sid, order)

def initialize(context):
    order(sid(1), 10)"""

        algo = self.make_algo(script=algotext)
        with pytest.raises(ze.OrderDuringInitialize):
            algo.run()


class TestOrderRounding:
    def test_order_rounding(self):
        answer_key = [
            (0, 0),
            (10, 10),
            (1.1, 1),
            (1.5, 1),
            (1.9998, 1),
            (1.99991, 2),
        ]

        for input, answer in answer_key:
            assert answer == TradingAlgorithm.round_order(input)

            assert -1 * answer == TradingAlgorithm.round_order(-1 * input)
