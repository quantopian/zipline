"""
Tests for our testing utilities.
"""
from itertools import product
from unittest import TestCase

from numpy import array, empty

from zipline._protocol import BarData
from zipline.finance.asset_restrictions import NoRestrictions
from zipline.finance.order import Order

from zipline.testing import (
    check_arrays,
    make_alternating_boolean_array,
    make_cascading_boolean_array,
    parameter_space,
)
from zipline.testing.fixtures import (
    WithConstantEquityMinuteBarData,
    WithDataPortal,
    ZiplineTestCase,
)
from zipline.testing.slippage import TestingSlippage
from zipline.utils.numpy_utils import bool_dtype


class TestParameterSpace(TestCase):

    x_args = [1, 2]
    y_args = [3, 4]

    @classmethod
    def setUpClass(cls):
        cls.xy_invocations = []
        cls.yx_invocations = []

    @classmethod
    def tearDownClass(cls):
        # This is the only actual test here.
        assert cls.xy_invocations == list(product(cls.x_args, cls.y_args))
        assert cls.yx_invocations == list(product(cls.y_args, cls.x_args))

    @parameter_space(x=x_args, y=y_args)
    def test_xy(self, x, y):
        self.xy_invocations.append((x, y))

    @parameter_space(x=x_args, y=y_args)
    def test_yx(self, y, x):
        # Ensure that product is called with args in the order that they appear
        # in the function's parameter list.
        self.yx_invocations.append((y, x))

    def test_nothing(self):
        # Ensure that there's at least one "real" test in the class, or else
        # our {setUp,tearDown}Class won't be called if, for example,
        # `parameter_space` returns None.
        pass


class TestMakeBooleanArray(TestCase):

    def test_make_alternating_boolean_array(self):
        check_arrays(
            make_alternating_boolean_array((3, 3)),
            array(
                [[True,  False,  True],
                 [False,  True, False],
                 [True,  False,  True]]
            ),
        )
        check_arrays(
            make_alternating_boolean_array((3, 3), first_value=False),
            array(
                [[False,  True, False],
                 [True,  False,  True],
                 [False,  True, False]]
            ),
        )
        check_arrays(
            make_alternating_boolean_array((1, 3)),
            array([[True, False, True]]),
        )
        check_arrays(
            make_alternating_boolean_array((3, 1)),
            array([[True], [False], [True]]),
        )
        check_arrays(
            make_alternating_boolean_array((3, 0)),
            empty((3, 0), dtype=bool_dtype),
        )

    def test_make_cascading_boolean_array(self):
        check_arrays(
            make_cascading_boolean_array((3, 3)),
            array(
                [[True,   True, False],
                 [True,  False, False],
                 [False, False, False]]
            ),
        )
        check_arrays(
            make_cascading_boolean_array((3, 3), first_value=False),
            array(
                [[False, False, True],
                 [False,  True, True],
                 [True,   True, True]]
            ),
        )
        check_arrays(
            make_cascading_boolean_array((1, 3)),
            array([[True, True, False]]),
        )
        check_arrays(
            make_cascading_boolean_array((3, 1)),
            array([[False], [False], [False]]),
        )
        check_arrays(
            make_cascading_boolean_array((3, 0)),
            empty((3, 0), dtype=bool_dtype),
        )


class TestTestingSlippage(WithConstantEquityMinuteBarData,
                          WithDataPortal,
                          ZiplineTestCase):
    ASSET_FINDER_EQUITY_SYMBOLS = ('A',)
    ASSET_FINDER_EQUITY_SIDS = (1,)

    @classmethod
    def init_class_fixtures(cls):
        super(TestTestingSlippage, cls).init_class_fixtures()
        cls.asset = cls.asset_finder.retrieve_asset(1)
        cls.minute, _ = (
            cls.trading_calendar.open_and_close_for_session(cls.START_DATE)
        )

    def init_instance_fixtures(self):
        super(TestTestingSlippage, self).init_instance_fixtures()
        self.bar_data = BarData(
            self.data_portal,
            lambda: self.minute,
            "minute",
            self.trading_calendar,
            NoRestrictions()
        )

    def make_order(self, amount):
        return Order(
            self.minute,
            self.asset,
            amount,
        )

    def test_constant_filled_per_tick(self):
        filled_per_tick = 1
        model = TestingSlippage(filled_per_tick)
        order = self.make_order(100)

        price, volume = model.process_order(self.bar_data, order)

        self.assertEqual(price, self.EQUITY_MINUTE_CONSTANT_CLOSE)
        self.assertEqual(volume, filled_per_tick)

    def test_fill_all(self):
        filled_per_tick = TestingSlippage.ALL
        order_amount = 100

        model = TestingSlippage(filled_per_tick)
        order = self.make_order(order_amount)

        price, volume = model.process_order(self.bar_data, order)

        self.assertEqual(price, self.EQUITY_MINUTE_CONSTANT_CLOSE)
        self.assertEqual(volume, order_amount)
