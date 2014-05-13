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


class ZiplineError(Exception):
    msg = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.message = str(self)

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg

    __unicode__ = __str__
    __repr__ = __str__


class WrongDataForTransform(ZiplineError):
    """
    Raised whenever a rolling transform is called on an event that
    does not have the necessary properties.
    """
    msg = "{transform} requires {fields}. Event cannot be processed."


class UnsupportedSlippageModel(ZiplineError):
    """
    Raised if a user script calls the override_slippage magic
    with a slipage object that isn't a VolumeShareSlippage or
    FixedSlipapge
    """
    msg = """
You attempted to override slippage with an unsupported class. \
Please use VolumeShareSlippage or FixedSlippage.
""".strip()


class OverrideSlippagePostInit(ZiplineError):
    # Raised if a users script calls override_slippage magic
    # after the initialize method has returned.
    msg = """
You attempted to override slippage after the simulation has \
started. You may only call override_slippage in your initialize \
method.
""".strip()


class RegisterTradingControlPostInit(ZiplineError):
    # Raised if a user's script register's a trading control after initialize
    # has been run.
    msg = """
You attempted to set a trading control after the simulation has \
started. Trading controls may only be set during initialize.
""".strip()


class UnsupportedCommissionModel(ZiplineError):
    """
    Raised if a user script calls the override_commission magic
    with a commission object that isn't a PerShare, PerTrade or
    PerDollar commission
    """
    msg = """
You attempted to override commission with an unsupported class. \
Please use PerShare or PerTrade.
""".strip()


class OverrideCommissionPostInit(ZiplineError):
    """
    Raised if a users script calls override_commission magic
    after the initialize method has returned.
    """
    msg = """
You attempted to override commission after the simulation has \
started. You may only call override_commission in your initialize \
method.
""".strip()


class TransactionWithNoVolume(ZiplineError):
    """
    Raised if a transact call returns a transaction with zero volume.
    """
    msg = """
Transaction {txn} has a volume of zero.
""".strip()


class TransactionWithWrongDirection(ZiplineError):
    """
    Raised if a transact call returns a transaction with a direction that
    does not match the order.
    """
    msg = """
Transaction {txn} not in same direction as corresponding order {order}.
""".strip()


class TransactionWithNoAmount(ZiplineError):
    """
    Raised if a transact call returns a transaction with zero amount.
    """
    msg = """
Transaction {txn} has an amount of zero.
""".strip()


class TransactionVolumeExceedsOrder(ZiplineError):
    """
    Raised if a transact call returns a transaction with a volume greater than
the corresponding order.
    """
    msg = """
Transaction volume of {txn} exceeds the order volume of {order}.
""".strip()


class UnsupportedOrderParameters(ZiplineError):
    """
    Raised if a set of mutually exclusive parameters are passed to an order
    call.
    """
    msg = "{msg}"


class TradingControlViolation(ZiplineError):
    """
    Raised if an order would violate a constraint set by a TradingControl.
    """
    msg = """
Order for {amount} shares of {sid} violates trading constraint {constraint}.
""".strip()
