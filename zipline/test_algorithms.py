#
# Copyright 2013 Quantopian, Inc.
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
            is no bounding box. Make the previous paragraph ending colon a
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
from copy import deepcopy
import numpy as np

from zipline.algorithm import TradingAlgorithm
from zipline.finance.slippage import FixedSlippage


class TestAlgorithm(TradingAlgorithm):
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """

    def initialize(self, sid, amount, order_count, sid_filter=None):
        self.count = order_count
        self.sid = sid
        self.amount = amount
        self.incr = 0

        if sid_filter:
            self.sid_filter = sid_filter
        else:
            self.sid_filter = [self.sid]

    def handle_data(self, data):
        #place an order for 100 shares of sid
        if self.incr < self.count:
            self.order(self.sid, self.amount)
            self.incr += 1


class HeavyBuyAlgorithm(TradingAlgorithm):
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """

    def initialize(self, sid, amount):
        self.sid = sid
        self.amount = amount
        self.incr = 0

    def handle_data(self, data):
        #place an order for 100 shares of sid
        self.order(self.sid, self.amount)
        self.incr += 1


class NoopAlgorithm(TradingAlgorithm):
    """
    Dolce fa niente.
    """
    def get_sid_filter(self):
        return []

    def set_transact_setter(self, txn_sim_callable):
        pass


class ExceptionAlgorithm(TradingAlgorithm):
    """
    Throw an exception from the method name specified in the
    constructor.
    """

    def initialize(self, throw_from, sid):

        self.throw_from = throw_from
        self.sid = sid

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
            return [self.sid]

    def set_transact_setter(self, txn_sim_callable):
        pass


class DivByZeroAlgorithm(TradingAlgorithm):

    def initialize(self, sid):
        self.sid = sid
        self.incr = 0

    def handle_data(self, data):
        self.incr += 1
        if self.incr > 4:
            5 / 0
        pass


class TooMuchProcessingAlgorithm(TradingAlgorithm):

    def initialize(self, sid):
        self.sid = sid

    def handle_data(self, data):
        # Unless we're running on some sort of
        # supercomputer this will hit timeout.
        for i in xrange(1000000000):
            self.foo = i


class TimeoutAlgorithm(TradingAlgorithm):

    def initialize(self, sid):
        self.sid = sid
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

from zipline.algorithm import TradingAlgorithm
from zipline.transforms import BatchTransform, batch_transform
from zipline.transforms import MovingAverage


class TestRegisterTransformAlgorithm(TradingAlgorithm):
    def initialize(self, *args, **kwargs):
        self.add_transform(MovingAverage, 'mavg', ['price'],
                           market_aware=True,
                           window_length=2)

        self.set_slippage(FixedSlippage())

    def handle_data(self, data):
        pass


##########################################
# Algorithm using simple batch transforms

class ReturnPriceBatchTransform(BatchTransform):
    def get_value(self, data):
        assert data.shape[1] == self.window_length, \
            "data shape={0} does not equal window_length={1} for data={2}".\
            format(data.shape[1], self.window_length, data)
        return data.price


@batch_transform
def return_price_batch_decorator(data):
    return data.price


@batch_transform
def return_args_batch_decorator(data, *args, **kwargs):
    return args, kwargs


@batch_transform
def return_data(data, *args, **kwargs):
    return data


@batch_transform
def uses_ufunc(data, *args, **kwargs):
    # ufuncs like np.log should not crash
    return np.log(data)


@batch_transform
def price_multiple(data, multiplier, extra_arg=1):
    return data.price * multiplier * extra_arg


class BatchTransformAlgorithm(TradingAlgorithm):
    def initialize(self, *args, **kwargs):
        self.refresh_period = kwargs.pop('refresh_period', 1)
        self.window_length = kwargs.pop('window_length', 3)

        self.args = args
        self.kwargs = kwargs

        self.history_return_price_class = []
        self.history_return_price_decorator = []
        self.history_return_args = []
        self.history_return_arbitrary_fields = []
        self.history_return_nan = []
        self.history_return_sid_filter = []
        self.history_return_field_filter = []
        self.history_return_field_no_filter = []
        self.history_return_ticks = []
        self.history_return_not_full = []

        self.return_price_class = ReturnPriceBatchTransform(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=False
        )

        self.return_price_decorator = return_price_batch_decorator(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=False
        )

        self.return_args_batch = return_args_batch_decorator(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=False
        )

        self.return_arbitrary_fields = return_data(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=False
        )

        self.return_nan = return_price_batch_decorator(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=True
        )

        self.return_sid_filter = return_price_batch_decorator(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=True,
            sids=[0]
        )

        self.return_field_filter = return_data(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=True,
            fields=['price']
        )

        self.return_field_no_filter = return_data(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=True
        )

        self.return_not_full = return_data(
            refresh_period=1,
            window_length=self.window_length,
            compute_only_full=False
        )

        self.uses_ufunc = uses_ufunc(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=False
        )

        self.price_multiple = price_multiple(
            refresh_period=self.refresh_period,
            window_length=self.window_length,
            clean_nans=False
        )

        self.iter = 0

        self.set_slippage(FixedSlippage())

    def handle_data(self, data):
        self.history_return_price_class.append(
            self.return_price_class.handle_data(data))
        self.history_return_price_decorator.append(
            self.return_price_decorator.handle_data(data))
        self.history_return_args.append(
            self.return_args_batch.handle_data(
                data, *self.args, **self.kwargs))
        self.history_return_not_full.append(
            self.return_not_full.handle_data(data))
        self.uses_ufunc.handle_data(data)

        # check that calling transforms with the same arguments
        # is idempotent
        self.price_multiple.handle_data(data, 1, extra_arg=1)

        if self.price_multiple.full:
            pre = self.price_multiple.rolling_panel.get_current().shape[0]
            result1 = self.price_multiple.handle_data(data, 1, extra_arg=1)
            post = self.price_multiple.rolling_panel.get_current().shape[0]
            assert pre == post, "batch transform is appending redundant events"
            result2 = self.price_multiple.handle_data(data, 1, extra_arg=1)
            assert result1 is result2, "batch transform is not idempotent"

            # check that calling transform with the same data, but
            # different supplemental arguments results in new
            # results.
            result3 = self.price_multiple.handle_data(data, 2, extra_arg=1)
            assert result1 is not result3, \
                "batch transform is not updating for new args"

            result4 = self.price_multiple.handle_data(data, 1, extra_arg=2)
            assert result1 is not result4,\
                "batch transform is not updating for new kwargs"

        new_data = deepcopy(data)
        for sid in new_data:
            new_data[sid]['arbitrary'] = 123

        self.history_return_arbitrary_fields.append(
            self.return_arbitrary_fields.handle_data(new_data))

        # nan every second event price
        if self.iter % 2 == 0:
            self.history_return_nan.append(
                self.return_nan.handle_data(data))
        else:
            nan_data = deepcopy(data)
            for sid in nan_data.iterkeys():
                nan_data[sid].price = np.nan
            self.history_return_nan.append(
                self.return_nan.handle_data(nan_data))

        self.iter += 1

        # Add a new sid to check that it does not get included
        extra_sid_data = deepcopy(data)
        extra_sid_data[1] = extra_sid_data[0]
        self.history_return_sid_filter.append(
            self.return_sid_filter.handle_data(extra_sid_data)
        )

        # Add a field to check that it does not get included
        extra_field_data = deepcopy(data)
        extra_field_data[0]['ignore'] = extra_sid_data[0]['price']
        self.history_return_field_filter.append(
            self.return_field_filter.handle_data(extra_field_data)
        )
        self.history_return_field_no_filter.append(
            self.return_field_no_filter.handle_data(extra_field_data)
        )


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
