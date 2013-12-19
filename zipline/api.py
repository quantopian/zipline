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

from copy import copy

import pytz
import numpy as np

import zipline

from zipline.errors import (
    UnsupportedSlippageModel,
    OverrideSlippagePostInit,
    UnsupportedCommissionModel,
    OverrideCommissionPostInit
)

from zipline.finance.slippage import (
    SlippageModel,
    FixedSlippage,
    VolumeShareSlippage,
)
from zipline.finance.commission import PerShare, PerTrade, PerDollar

from zipline.utils.algo_instance import get_algo_instance

slippage = zipline.finance.slippage
math_utils = zipline.utils.math_utils
batch_transform = zipline.transforms.BatchTransform


def record(**kwargs):
    """
    Track and record local variable (i.e. attributes) each day.
    """
    algo = get_algo_instance()
    for name, value in kwargs.items():
        algo._recorded_vars[name] = value


def order(sid, amount, limit_price=None, stop_price=None):
    algo = get_algo_instance()
    return algo.blotter.order(sid, amount, limit_price, stop_price)


def order_value(sid, value, limit_price=None, stop_price=None):
    """
    Place an order by desired value rather than desired number of shares.
    If the requested sid is found in the universe, the requested value is
    divided by its price to imply the number of shares to transact.

    value > 0 :: Buy/Cover
    value < 0 :: Sell/Short
    Market order:    order(sid, value)
    Limit order:     order(sid, value, limit_price)
    Stop order:      order(sid, value, None, stop_price)
    StopLimit order: order(sid, value, limit_price, stop_price)
    """
    algo = get_algo_instance()

    last_price = algo.trading_client.current_data[sid].price
    if np.allclose(last_price, 0):
        zero_message = "Price of 0 for {psid}; can't infer value".format(
            psid=sid
        )
        algo.logger.debug(zero_message)
        # Don't place any order
        return
    else:
        amount = value / last_price
        return order(sid, amount, limit_price, stop_price)


def get_datetime(algo):
    """
    Returns a copy of the datetime.
    """
    algo = get_algo_instance()
    date_copy = copy(algo.datetime)
    assert date_copy.tzinfo == pytz.utc, \
        "Algorithm should have a utc datetime"
    return date_copy


def set_slippage(slippage):
    algo = get_algo_instance()

    if not isinstance(slippage, SlippageModel):
        raise UnsupportedSlippageModel()
    if algo.initialized:
        raise OverrideSlippagePostInit()
    algo.slippage = slippage


def set_commission(commission):
    algo = get_algo_instance()

    if not isinstance(commission, (PerShare, PerTrade, PerDollar)):
        raise UnsupportedCommissionModel()

    if algo.initialized:
        raise OverrideCommissionPostInit()
    algo.commission = commission


def order_percent(sid, percent, limit_price=None, stop_price=None):
    """
    Place an order in the specified security corresponding to the given
    percent of the current portfolio value.

    Note that percent must expressed as a decimal (0.50 means 50\%).
    """
    algo = get_algo_instance()

    value = algo.portfolio.portfolio_value * percent
    return order_value(sid, value, limit_price, stop_price)


def order_target(sid, target, limit_price=None, stop_price=None):
    """
    Place an order to adjust a position to a target number of shares. If
    the position doesn't already exist, this is equivalent to placing a new
    order. If the position does exist, this is equivalent to placing an
    order for the difference between the target number of shares and the
    current number of shares.
    """
    algo = get_algo_instance()

    if sid in algo.portfolio.positions:
        current_position = algo.portfolio.positions[sid].amount
        req_shares = target - current_position
        return order(sid, req_shares, limit_price, stop_price)
    else:
        return order(sid, target, limit_price, stop_price)


def order_target_value(sid, target, limit_price=None,
                       stop_price=None):
    """
    Place an order to adjust a position to a target value. If
    the position doesn't already exist, this is equivalent to placing a new
    order. If the position does exist, this is equivalent to placing an
    order for the difference between the target value and the
    current value.
    """
    algo = get_algo_instance()

    if sid in algo.portfolio.positions:
        current_position = algo.portfolio.positions[sid].amount
        current_price = algo.portfolio.positions[sid].last_sale_price
        current_value = current_position * current_price
        req_value = target - current_value
        return order_value(sid, req_value, limit_price, stop_price)
    else:
        return order_value(sid, target, limit_price, stop_price)


def order_target_percent(sid, target, limit_price=None,
                         stop_price=None):
    """
    Place an order to adjust a position to a target percent of the
    current portfolio value. If the position doesn't already exist, this is
    equivalent to placing a new order. If the position does exist, this is
    equivalent to placing an order for the difference between the target
    percent and the current percent.

    Note that target must expressed as a decimal (0.50 means 50\%).
    """
    algo = get_algo_instance()

    if sid in algo.portfolio.positions:
        current_position = algo.portfolio.positions[sid].amount
        current_price = algo.portfolio.positions[sid].last_sale_price
        current_value = current_position * current_price
    else:
        current_value = 0
    target_value = algo.portfolio.portfolio_value * target

    req_value = target_value - current_value
    return order_value(sid, req_value, limit_price, stop_price)


def get_open_orders(sid=None):
    algo = get_algo_instance()

    if sid is None:
        return {key: [order.to_api_obj() for order in orders]
                for key, orders
                in algo.blotter.open_orders.iteritems()}
    if sid in algo.blotter.open_orders:
        orders = algo.blotter.open_orders[sid]
        return [order.to_api_obj() for order in orders]
    return []


def get_order(order_id):
    algo = get_algo_instance()

    if order_id in algo.blotter.orders:
        return algo.blotter.orders[order_id].to_api_obj()


def cancel_order(order_param):
    algo = get_algo_instance()

    order_id = order_param
    if isinstance(order_param, zipline.protocol.Order):
        order_id = order_param.id

    algo.blotter.cancel(order_id)


__all__ = [
    'order',
    'order_value',
    'order_percent',
    'order_target',
    'order_target_value',
    'order_target_percent',
    'get_open_orders',
    'get_order',
    'cancel_order',
    'set_commission',
    'slippage',
    'math_utils',
    'set_slippage',
    'batch_transform',
    'get_datetime',
    'record',
    'FixedSlippage',
    'VolumeShareSlippage'
]
