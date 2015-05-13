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
You attempted to override slippage outside of `initialize`. \
You may only call override_slippage in your initialize method.
""".strip()


class RegisterTradingControlPostInit(ZiplineError):
    # Raised if a user's script register's a trading control after initialize
    # has been run.
    msg = """
You attempted to set a trading control outside of `initialize`. \
Trading controls may only be set in your initialize method.
""".strip()


class RegisterAccountControlPostInit(ZiplineError):
    # Raised if a user's script register's a trading control after initialize
    # has been run.
    msg = """
You attempted to set an account control outside of `initialize`. \
Account controls may only be set in your initialize method.
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
You attempted to override commission outside of `initialize`. \
You may only call override_commission in your initialize method.
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


class BadOrderParameters(ZiplineError):
    """
    Raised if any impossible parameters (nan, negative limit/stop)
    are passed to an order call.
    """
    msg = "{msg}"


class OrderDuringInitialize(ZiplineError):
    """
    Raised if order is called during initialize()
    """
    msg = "{msg}"


class AccountControlViolation(ZiplineError):
    """
    Raised if the account violates a constraint set by a AccountControl.
    """
    msg = """
Account violates account constraint {constraint}.
""".strip()


class TradingControlViolation(ZiplineError):
    """
    Raised if an order would violate a constraint set by a TradingControl.
    """
    msg = """
Order for {amount} shares of {asset} violates trading constraint {constraint}.
""".strip()


class IncompatibleHistoryFrequency(ZiplineError):
    """
    Raised when a frequency is given to history which is not supported.
    At least, not yet.
    """
    msg = """
Requested history at frequency '{frequency}' cannot be created with data
at frequency '{data_frequency}'.
""".strip()


class MultipleSymbolsFound(ZiplineError):
    """
    Raised when a symbol() call contains a symbol that changed over
    time and is thus not resolvable without additional information
    provided via as_of_date.
    """
    msg = """
Multiple symbols with the name '{symbol}' found. Use the
as_of_date' argument to to specify when the date symbol-lookup
should be valid.

Possible options:{options}
    """.strip()


class SymbolNotFound(ZiplineError):
    """
    Raised when a symbol() call contains a non-existant symbol.
    """
    msg = """
Symbol '{symbol}' was not found.
""".strip()


class SidNotFound(ZiplineError):
    """
    Raised when a retrieve_asset() call contains a non-existent sid.
    """
    msg = """
Asset with sid '{sid}' was not found.
""".strip()


class IdentifierNotFound(ZiplineError):
    """
    Raised when a retrieve_asset_by_identifier() call contains a non-existent
    identifier.
    """
    msg = """
Asset with identifier '{identifier}' was not found.
""".strip()


class InvalidAssetType(ZiplineError):
    """
    Raised when an AssetFinder tries to build an Asset with an invalid
    AssetType.
    """
    msg = """
AssetMetaData contained an invalid Asset type: '{asset_type}'.
""".strip()


class UpdateAssetFinderTypeError(ZiplineError):
    """
    Raised when TradingEnvironment.update_asset_finder() gets an asset_finder
    arg that is not of AssetFinder class.
    """
    msg = """
TradingEnvironment can not set asset_finder to object of class {cls}.
""".strip()


class ConsumeAssetMetaDataError(ZiplineError):
    """
    Raised when AssetMetaData.consume() is called on a non-AssetMetaData object
    """
    msg = """
AssetMetaData can not consume {obj}.
""".strip()


class NoSourceError(ZiplineError):
    """
    Raised when no source is given to the pipeline
    """
    msg = """
No data source given.
""".strip()


class PipelineDateError(ZiplineError):
    """
    Raised when only one date is passed to the pipeline
    """
    msg = """
Only one simulation date given. Please specify both the 'start' and 'end' for
the simulation, or neither. If neither is given, the start and end of the
DataSource will be used. Given start = '{start}', end = '{end}'
""".strip()