#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Algorithm Protocol
===================

For a class to be passed as a trading algorithm to the
:py:class:`zipline.lines.SimulatedTrading` zipline it must follow an
implementation protocol. Examples of this algorithm protocol are provided
below.

The algorithm must expose methods:

  - initialize: method that takes no args, no returns. Simply called to
    enable the algorithm to set any internal state needed.

  - get_sid_filter: method that takes no args, and returns a list of valid
    sids. List must have a length between 1 and 10. If None is returned the
    filter will block all events.

  - handle_data: method that accepts a :py:class:`zipline.protocol.BarData`
    of the current state of the simulation universe. An example data object:

        ..  This outputs the table as an HTML table but for some reason there
            is no bounding box. Make the previous paraagraph ending colon a
            double-colon to turn this back into blockquoted table in ASCII art.

        +-----------------+--------------+----------------+-------------------+
        |                 | sid(133)     |  sid(134)      | sid(135)          |
        +=================+==============+================+===================+
        | price           | $10.10       | $22.50         | $13.37            |
        +-----------------+--------------+----------------+-------------------+
        | volume          | 10,000       | 5,000          | 50,000            |
        +-----------------+--------------+----------------+-------------------+
        | mvg_avg_30      | $9.97        | $22.61         | $13.37            |
        +-----------------+--------------+----------------+-------------------+
        | dt              | 6/30/2012    | 6/30/2011      | 6/29/2012         |
        +-----------------+--------------+----------------+-------------------+

  - set_order: method that accepts a callable. Will be set as the value of the
    order method of trading_client. An algorithm can then place orders with a
    valid sid and a number of shares::

        self.order(sid(133), share_count)

  - set_performance: property which can be set equal to the
    cumulative_trading_performance property of the trading_client. An
    algorithm can then check position information with the
    Portfolio object::

        self.Portfolio[sid(133)]['cost_basis']

  - set_transact_setter: method that accepts a callable. Will
    be set as the value of the set_transact_setter method of
    the trading_client. This allows an algorithm to change the
    slippage model used to predict transactions based on orders
    and trade events.

"""
import numpy as np

from nose.tools import assert_raises

from six.moves import range
from six import itervalues

from zipline.algorithm import TradingAlgorithm
from zipline.api import (
    FixedSlippage,
    order,
    set_slippage,
    record,
    sid,
)
from zipline.errors import UnsupportedOrderParameters
from zipline.assets import Future, Equity
from zipline.finance.commission import PerShare
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.finance.controls import AssetDateBounds
from zipline.utils.math_utils import round_if_near_integer


class TestAlgorithm(TradingAlgorithm):
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """

    def initialize(self,
                   sid,
                   amount,
                   order_count,
                   sid_filter=None,
                   slippage=None,
                   commission=None):
        self.count = order_count
        self.asset = self.sid(sid)
        self.amount = amount
        self.incr = 0

        if sid_filter:
            self.sid_filter = sid_filter
        else:
            self.sid_filter = [self.asset.sid]

        if slippage is not None:
            self.set_slippage(slippage)

        if commission is not None:
            self.set_commission(commission)

    def handle_data(self, data):
        # place an order for amount shares of sid
        if self.incr < self.count:
            self.order(self.asset, self.amount)
            self.incr += 1


class HeavyBuyAlgorithm(TradingAlgorithm):
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """

    def initialize(self, sid, amount):
        self.asset = self.sid(sid)
        self.amount = amount
        self.incr = 0

    def handle_data(self, data):
        # place an order for 100 shares of sid
        self.order(self.asset, self.amount)
        self.incr += 1


class NoopAlgorithm(TradingAlgorithm):
    """
    Dolce fa niente.
    """
    def initialize(self):
        pass

    def handle_data(self, data):
        pass


class ExceptionAlgorithm(TradingAlgorithm):
    """
    Throw an exception from the method name specified in the
    constructor.
    """

    def initialize(self, throw_from, sid):

        self.throw_from = throw_from
        self.asset = self.sid(sid)

        if self.throw_from == "initialize":
            raise Exception("Algo exception in initialize")
        else:
            pass

    def set_portfolio(self, portfolio):
        if self.throw_from == "set_portfolio":
            raise Exception("Algo exception in set_portfolio")
        else:
            pass

    def handle_data(self, data):
        if self.throw_from == "handle_data":
            raise Exception("Algo exception in handle_data")
        else:
            pass

    def get_sid_filter(self):
        if self.throw_from == "get_sid_filter":
            raise Exception("Algo exception in get_sid_filter")
        else:
            return [self.asset]

    def set_transact_setter(self, txn_sim_callable):
        pass


class DivByZeroAlgorithm(TradingAlgorithm):

    def initialize(self, sid):
        self.asset = self.sid(sid)
        self.incr = 0

    def handle_data(self, data):
        self.incr += 1
        if self.incr > 1:
            5 / 0
        pass


class TooMuchProcessingAlgorithm(TradingAlgorithm):

    def initialize(self, sid):
        self.asset = self.sid(sid)

    def handle_data(self, data):
        # Unless we're running on some sort of
        # supercomputer this will hit timeout.
        for i in range(1000000000):
            self.foo = i


class TimeoutAlgorithm(TradingAlgorithm):

    def initialize(self, sid):
        self.asset = self.sid(sid)
        self.incr = 0

    def handle_data(self, data):
        if self.incr > 4:
            import time
            time.sleep(100)
        pass


class RecordAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.incr = 0

    def handle_data(self, data):
        self.incr += 1
        self.record(incr=self.incr)
        name = 'name'
        self.record(name, self.incr)
        record(name, self.incr, 'name2', 2, name3=self.incr)


class TestOrderAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.incr = 0

    def handle_data(self, data):
        if self.incr == 0:
            assert 0 not in self.portfolio.positions
        else:
            assert self.portfolio.positions[0]['amount'] == \
                self.incr, "Orders not filled immediately."
            assert self.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."
        self.incr += 1
        self.order(self.sid(0), 1)


class TestOrderInstantAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.incr = 0
        self.last_price = None

    def handle_data(self, data):
        if self.incr == 0:
            assert 0 not in self.portfolio.positions
        else:
            assert self.portfolio.positions[0]['amount'] == \
                self.incr, "Orders not filled immediately."
            assert self.portfolio.positions[0]['last_sale_price'] == \
                self.last_price, "Orders was not filled at last price."
        self.incr += 1
        self.order_value(self.sid(0), data[0].price)
        self.last_price = data[0].price


class TestOrderStyleForwardingAlgorithm(TradingAlgorithm):
    """
    Test Algorithm for verifying that ExecutionStyles are properly forwarded by
    order API helper methods.  Pass the name of the method to be tested as a
    string parameter to this algorithm's constructor.
    """

    def __init__(self, *args, **kwargs):
        self.method_name = kwargs.pop('method_name')
        super(TestOrderStyleForwardingAlgorithm, self)\
            .__init__(*args, **kwargs)

    def initialize(self):
        self.incr = 0
        self.last_price = None

    def handle_data(self, data):
        if self.incr == 0:
            assert len(self.portfolio.positions.keys()) == 0

            method_to_check = getattr(self, self.method_name)
            method_to_check(self.sid(133),
                            data[0].price,
                            style=StopLimitOrder(10, 10))

            assert len(self.blotter.open_orders[self.sid(133)]) == 1
            result = self.blotter.open_orders[self.sid(133)][0]
            assert result.limit == 10
            assert result.stop == 10

            self.incr += 1


class TestOrderValueAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.incr = 0
        self.sale_price = None

    def handle_data(self, data):
        if self.incr == 0:
            assert 0 not in self.portfolio.positions
        else:
            assert self.portfolio.positions[0]['amount'] == \
                self.incr, "Orders not filled immediately."
            assert self.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."
        self.incr += 2

        multiplier = 2.
        if isinstance(self.sid(0), Future):
            multiplier *= self.sid(0).contract_multiplier

        self.order_value(self.sid(0), data[0].price * multiplier)


class TestTargetAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.target_shares = 0
        self.sale_price = None

    def handle_data(self, data):
        if self.target_shares == 0:
            assert 0 not in self.portfolio.positions
        else:
            assert self.portfolio.positions[0]['amount'] == \
                self.target_shares, "Orders not filled immediately."
            assert self.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."
        self.target_shares = 10
        self.order_target(self.sid(0), self.target_shares)


class TestOrderPercentAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.target_shares = 0
        self.sale_price = None

    def handle_data(self, data):
        if self.target_shares == 0:
            assert 0 not in self.portfolio.positions
            self.order(self.sid(0), 10)
            self.target_shares = 10
            return
        else:

            assert self.portfolio.positions[0]['amount'] == \
                self.target_shares, "Orders not filled immediately."
            assert self.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."

        self.order_percent(self.sid(0), .001)

        if isinstance(self.sid(0), Equity):
            price = data[0].price
            new_shares = (.001 * self.portfolio.portfolio_value) / price
        elif isinstance(self.sid(0), Future):
            new_shares = (.001 * self.portfolio.portfolio_value) / \
                (data[0].price * self.sid(0).contract_multiplier)

        new_shares = int(round_if_near_integer(new_shares))
        self.target_shares += new_shares


class TestTargetPercentAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.ordered = False
        self.sale_price = None

        # this makes the math easier to check
        self.commission = PerShare(0)
        self.slippage = FixedSlippage(spread=0.0)

    def handle_data(self, data):
        if not self.ordered:
            assert 0 not in self.portfolio.positions
        else:
            # Since you can't own fractional shares (at least in this
            # example), we want to make sure that our target amount is
            # no more than a share's value away from our current
            # holdings.
            target_value = self.portfolio.portfolio_value * 0.002
            position_value = self.portfolio.positions[0]['amount'] * \
                self.sale_price

            assert abs(target_value - position_value) <= self.sale_price, \
                "Orders not filled correctly"

            assert self.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."

        self.sale_price = data[0].price
        self.order_target_percent(self.sid(0), .002)
        self.ordered = True


class TestTargetValueAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.target_shares = 0
        self.sale_price = None

    def handle_data(self, data):
        if self.target_shares == 0:
            assert 0 not in self.portfolio.positions
            self.order(self.sid(0), 10)
            self.target_shares = 10
            return
        else:
            assert self.portfolio.positions[0]['amount'] == \
                self.target_shares, "Orders not filled immediately."
            assert self.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."

        self.order_target_value(self.sid(0), 20)
        self.target_shares = np.round(20 / data[0].price)

        if isinstance(self.sid(0), Equity):
            self.target_shares = np.round(20 / data[0].price)
        if isinstance(self.sid(0), Future):
            self.target_shares = np.round(
                20 / (data[0].price * self.sid(0).contract_multiplier))


############################
# AccountControl Test Algos#
############################


class SetMaxLeverageAlgorithm(TradingAlgorithm):
    def initialize(self, max_leverage=None):
        self.set_max_leverage(max_leverage=max_leverage)


############################
# TradingControl Test Algos#
############################


class SetMaxPositionSizeAlgorithm(TradingAlgorithm):
    def initialize(self, sid=None, max_shares=None, max_notional=None):
        self.order_count = 0
        self.set_max_position_size(sid=sid,
                                   max_shares=max_shares,
                                   max_notional=max_notional)


class SetMaxOrderSizeAlgorithm(TradingAlgorithm):
    def initialize(self, sid=None, max_shares=None, max_notional=None):
        self.order_count = 0
        self.set_max_order_size(sid=sid,
                                max_shares=max_shares,
                                max_notional=max_notional)


class SetDoNotOrderListAlgorithm(TradingAlgorithm):
    def initialize(self, sid=None, restricted_list=None):
        self.order_count = 0
        self.set_do_not_order_list(restricted_list)


class SetMaxOrderCountAlgorithm(TradingAlgorithm):
    def initialize(self, count):
        self.order_count = 0
        self.set_max_order_count(count)
        self.minute_count = 0


class SetLongOnlyAlgorithm(TradingAlgorithm):
    def initialize(self):
        self.order_count = 0
        self.set_long_only()


class SetAssetDateBoundsAlgorithm(TradingAlgorithm):
    """
    Algorithm that tries to order 1 share of sid 0 on every bar and has an
    AssetDateBounds() trading control in place.
    """
    def initialize(self):
        self.register_trading_control(AssetDateBounds())

    def handle_data(algo, data):
        algo.order(algo.sid(0), 1)


class TestRegisterTransformAlgorithm(TradingAlgorithm):
    def initialize(self, *args, **kwargs):
        self.set_slippage(FixedSlippage())

    def handle_data(self, data):
        pass


class AmbitiousStopLimitAlgorithm(TradingAlgorithm):
    """
    Algorithm that tries to buy with extremely low stops/limits and tries to
    sell with extremely high versions of same. Should not end up with any
    positions for reasonable data.
    """

    def initialize(self, *args, **kwargs):
        self.asset = self.sid(kwargs.pop('sid'))

    def handle_data(self, data):

        ########
        # Buys #
        ########

        # Buy with low limit, shouldn't trigger.
        self.order(self.asset, 100, limit_price=1)

        # But with high stop, shouldn't trigger
        self.order(self.asset, 100, stop_price=10000000)

        # Buy with high limit (should trigger) but also high stop (should
        # prevent trigger).
        self.order(self.asset, 100, limit_price=10000000, stop_price=10000000)

        # Buy with low stop (should trigger), but also low limit (should
        # prevent trigger).
        self.order(self.asset, 100, limit_price=1, stop_price=1)

        #########
        # Sells #
        #########

        # Sell with high limit, shouldn't trigger.
        self.order(self.asset, -100, limit_price=1000000)

        # Sell with low stop, shouldn't trigger.
        self.order(self.asset, -100, stop_price=1)

        # Sell with low limit (should trigger), but also high stop (should
        # prevent trigger).
        self.order(self.asset, -100, limit_price=1000000, stop_price=1000000)

        # Sell with low limit (should trigger), but also low stop (should
        # prevent trigger).
        self.order(self.asset, -100, limit_price=1, stop_price=1)

        ###################
        # Rounding Checks #
        ###################
        self.order(self.asset, 100, limit_price=.00000001)
        self.order(self.asset, -100, stop_price=.00000001)


class SetPortfolioAlgorithm(TradingAlgorithm):
    """
    An algorithm that tries to set the portfolio directly.

    The portfolio should be treated as a read-only object
    within the algorithm.
    """

    def initialize(self, *args, **kwargs):
        pass

    def handle_data(self, data):
        self.portfolio = 3


class TALIBAlgorithm(TradingAlgorithm):
    """
    An algorithm that applies a TA-Lib transform. The transform object can be
    passed at initialization with the 'talib' keyword argument. The results are
    stored in the talib_results array.
    """
    def initialize(self, *args, **kwargs):

        if 'talib' not in kwargs:
            raise KeyError('No TA-LIB transform specified '
                           '(use keyword \'talib\').')
        elif not isinstance(kwargs['talib'], (list, tuple)):
            self.talib_transforms = (kwargs['talib'],)
        else:
            self.talib_transforms = kwargs['talib']

        self.talib_results = dict((t, []) for t in self.talib_transforms)

    def handle_data(self, data):
        for t in self.talib_transforms:
            result = t.handle_data(data)
            if result is None:
                if len(t.talib_fn.output_names) == 1:
                    result = np.nan
                else:
                    result = (np.nan,) * len(t.talib_fn.output_names)
            self.talib_results[t].append(result)


class EmptyPositionsAlgorithm(TradingAlgorithm):
    """
    An algorithm that ensures that 'phantom' positions do not appear in
    portfolio.positions in the case that a position has been entered
    and fully exited.
    """
    def initialize(self, sids, *args, **kwargs):
        self.ordered = False
        self.exited = False
        self.sids = sids

    def handle_data(self, data):
        if not self.ordered:
            for s in self.sids:
                self.order(self.sid(s), 1)
            self.ordered = True

        if not self.exited:
            amounts = [pos.amount for pos
                       in itervalues(self.portfolio.positions)]

            if (
                len(amounts) > 0 and
                all([(amount == 1) for amount in amounts])
            ):
                for stock in self.portfolio.positions:
                    self.order(self.sid(stock), -1)
                self.exited = True

        # Should be 0 when all positions are exited.
        self.record(num_positions=len(self.portfolio.positions))


class InvalidOrderAlgorithm(TradingAlgorithm):
    """
    An algorithm that tries to make various invalid order calls, verifying that
    appropriate exceptions are raised.
    """
    def initialize(self, *args, **kwargs):
        self.asset = self.sid(kwargs.pop('sids')[0])

    def handle_data(self, data):
        from zipline.api import (
            order_percent,
            order_target,
            order_target_percent,
            order_target_value,
            order_value,
        )

        for style in [MarketOrder(), LimitOrder(10),
                      StopOrder(10), StopLimitOrder(10, 10)]:

            with assert_raises(UnsupportedOrderParameters):
                order(self.asset, 10, limit_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order(self.asset, 10, stop_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_value(self.asset, 300, limit_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_value(self.asset, 300, stop_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_percent(self.asset, .1, limit_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_percent(self.asset, .1, stop_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_target(self.asset, 100, limit_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_target(self.asset, 100, stop_price=10, style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_target_value(self.asset, 100,
                                   limit_price=10,
                                   style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_target_value(self.asset, 100,
                                   stop_price=10,
                                   style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_target_percent(self.asset, .2,
                                     limit_price=10,
                                     style=style)

            with assert_raises(UnsupportedOrderParameters):
                order_target_percent(self.asset, .2,
                                     stop_price=10,
                                     style=style)


##############################
# Quantopian style algorithms

# Noop algo
def initialize_noop(context):
    pass


def handle_data_noop(context, data):
    pass


# API functions
def initialize_api(context):
    context.incr = 0
    context.sale_price = None
    set_slippage(FixedSlippage())


def handle_data_api(context, data):
    if context.incr == 0:
        assert 0 not in context.portfolio.positions
    else:
        assert context.portfolio.positions[0]['amount'] == \
            context.incr, "Orders not filled immediately."
        assert context.portfolio.positions[0]['last_sale_price'] == \
            data[0].price, "Orders not filled at current price."
    context.incr += 1
    order(sid(0), 1)

    record(incr=context.incr)

###########################
# AlgoScripts as strings
noop_algo = """
# Noop algo
def initialize(context):
    pass

def handle_data(context, data):
    pass
"""

api_algo = """
from zipline.api import (order,
                         set_slippage,
                         FixedSlippage,
                         record,
                         sid)

def initialize(context):
    context.incr = 0
    context.sale_price = None
    set_slippage(FixedSlippage())

def handle_data(context, data):
    if context.incr == 0:
        assert 0 not in context.portfolio.positions
    else:
        assert context.portfolio.positions[0]['amount'] == \
                context.incr, "Orders not filled immediately."
        assert context.portfolio.positions[0]['last_sale_price'] == \
                data[0].price, "Orders not filled at current price."
    context.incr += 1
    order(sid(0), 1)

    record(incr=context.incr)
"""

api_get_environment_algo = """
from zipline.api import get_environment, order, symbol


def initialize(context):
    context.environment = get_environment()

def handle_data(context, data):
    pass
"""

api_symbol_algo = """
from zipline.api import (order,
                         symbol)

def initialize(context):
    pass

def handle_data(context, data):
    order(symbol('TEST'), 1)
"""

call_order_in_init = """
from zipline.api import (order)

def initialize(context):
    order(0, 10)
    pass

def handle_data(context, data):
    pass
"""

access_portfolio_in_init = """
def initialize(context):
    var = context.portfolio.cash
    pass

def handle_data(context, data):
    pass
"""

access_account_in_init = """
def initialize(context):
    var = context.account.settled_cash
    pass

def handle_data(context, data):
    pass
"""

call_all_order_methods = """
from zipline.api import (order,
                         order_value,
                         order_percent,
                         order_target,
                         order_target_value,
                         order_target_percent,
                         sid)

def initialize(context):
    pass

def handle_data(context, data):
    order(sid(0), 10)
    order_value(sid(0), 300)
    order_percent(sid(0), .1)
    order_target(sid(0), 100)
    order_target_value(sid(0), 100)
    order_target_percent(sid(0), .2)
"""

record_variables = """
from zipline.api import record

def initialize(context):
    context.stocks = [0, 1]
    context.incr = 0

def handle_data(context, data):
    context.incr += 1
    record(incr=context.incr)
"""

record_float_magic = """
from zipline.api import record

def initialize(context):
    context.stocks = [0, 1]
    context.incr = 0

def handle_data(context, data):
    context.incr += 1
    record(data=float('%s'))
"""
