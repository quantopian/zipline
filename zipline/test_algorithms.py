#
# Copyright 2012 Quantopian, Inc.
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
:py:class:`zipline.lines.SimulatedTrading` zipline
it must follow an implementation protocol. Examples of this algorithm protocol
are provided below.

The algorithm must expose methods:

  - initialize: method that takes no args, no returns. Simply called to
  enable the algorithm to set any internal state needed.

  - get_sid_filter: method that takes no args, and returns a list
  of valid sids. List must have a length between 1 and 10. If None is returned
  the filter will block all events.

  - handle_data: method that accepts a :py:class:`zipline.protocol_utils.ndict`
  of the current state of the simulation universe. An example data ndict::

    +-----------------+--------------+----------------+--------------------+
    |                 | sid(133)     |  sid(134)      | sid(135)           |
    +=================+==============+================+====================+
    | price           | $10.10       | $22.50         | $13.37             |
    +-----------------+--------------+----------------+--------------------+
    | volume          | 10,000       | 5,000          | 50,000             |
    +-----------------+--------------+----------------+--------------------+
    | mvg_avg_30      | $9.97        | $22.61         | $13.37             |
    +-----------------+--------------+----------------+--------------------+
    | dt              | 6/30/2012    | 6/30/2011      | 6/29/2012          |
    +-----------------+--------------+----------------+--------------------+

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
        self.frame_count += 1
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
        self.frame_count += 1
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

    def set_order(self, order_callable):
        if self.throw_from == "set_order":
            raise Exception("Algo exception in set_order")
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

from datetime import timedelta
from zipline.algorithm import TradingAlgorithm
from zipline.gens.transform import BatchTransform, batch_transform
from zipline.gens.mavg import MovingAverage


class TestRegisterTransformAlgorithm(TradingAlgorithm):
    def initialize(self, *args, **kwargs):
        self.add_transform(MovingAverage, 'mavg', ['price'],
                           market_aware=True,
                           days=2)

        self.set_slippage(FixedSlippage())

    def handle_data(self, data):
        pass


##########################################
# Algorithm using simple batch transforms

class ReturnPriceBatchTransform(BatchTransform):
    def get_value(self, data):
        return data.price


@batch_transform
def return_price_batch_decorator(data):
    return data.price


@batch_transform
def return_args_batch_decorator(data, *args, **kwargs):
    return args, kwargs


class BatchTransformAlgorithm(TradingAlgorithm):
    def initialize(self, *args, **kwargs):
        self.history_return_price_class = []
        self.history_return_price_decorator = []
        self.history_return_args = []

        self.days = 3

        self.args = args
        self.kwargs = kwargs

        self.return_price_class = ReturnPriceBatchTransform(
            market_aware=False,
            refresh_period=2,
            delta=timedelta(days=self.days)
        )

        self.return_price_decorator = return_price_batch_decorator(
            market_aware=False,
            refresh_period=2,
            delta=timedelta(days=self.days)
        )

        self.return_args_batch = return_args_batch_decorator(
            market_aware=False,
            refresh_period=2,
            delta=timedelta(days=self.days)
        )

        self.set_slippage(FixedSlippage())

    def handle_data(self, data):
        self.history_return_price_class.append(
            self.return_price_class.handle_data(data))
        self.history_return_price_decorator.append(
            self.return_price_decorator.handle_data(data))
        self.history_return_args.append(
            self.return_args_batch.handle_data(
                data, *self.args, **self.kwargs))
