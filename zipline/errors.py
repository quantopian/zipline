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
Order for {amount} shares of {asset} at {datetime} violates trading constraint
{constraint}.
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


class HistoryInInitialize(ZiplineError):
    """
    Raised when an algorithm calls history() in initialize.
    """
    msg = "history() should only be called in handle_data()"


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


class RootSymbolNotFound(ZiplineError):
    """
    Raised when a lookup_future_chain() call contains a non-existant symbol.
    """
    msg = """
Root symbol '{root_symbol}' was not found.
""".strip()


class SidNotFound(ZiplineError):
    """
    Raised when a retrieve_asset() call contains a non-existent sid.
    """
    msg = """
Asset with sid '{sid}' was not found.
""".strip()


class ConsumeAssetMetaDataError(ZiplineError):
    """
    Raised when AssetFinder.consume() is called on an invalid object.
    """
    msg = """
AssetFinder can not consume metadata of type {obj}. Metadata must be a dict, a
DataFrame, or a tables.Table. If the provided metadata is a Table, the rows
must contain both or one of 'sid' or 'symbol'.
""".strip()


class MapAssetIdentifierIndexError(ZiplineError):
    """
    Raised when AssetMetaData.map_identifier_index_to_sids() is called on an
    index of invalid objects.
    """
    msg = """
AssetFinder can not map an index with values of type {obj}. Asset indices of
DataFrames or Panels must be integer sids, string symbols, or Asset objects.
""".strip()


class SidAssignmentError(ZiplineError):
    """
    Raised when an AssetFinder tries to build an Asset that does not have a sid
    and that AssetFinder is not permitted to assign sids.
    """
    msg = """
AssetFinder metadata is missing a SID for identifier '{identifier}'.
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


class WindowLengthTooLong(ZiplineError):
    """
    Raised when a trailing window is instantiated with a lookback greater than
    the length of the underlying array.
    """
    msg = (
        "Can't construct a rolling window of length "
        "{window_length} on an array of length {nrows}."
    ).strip()


class WindowLengthNotPositive(ZiplineError):
    """
    Raised when a trailing window would be instantiated with a length less than
    1.
    """
    msg = (
        "Expected a window_length greater than 0, got {window_length}."
    ).strip()


class InputTermNotAtomic(ZiplineError):
    """
    Raised when a non-atomic term is specified as an input to a Pipeline API
    term with a lookback window.
    """
    msg = (
        "Can't compute {parent} with non-atomic input {child}."
    )


class TermInputsNotSpecified(ZiplineError):
    """
    Raised if a user attempts to construct a term without specifying inputs and
    that term does not have class-level default inputs.
    """
    msg = "{termname} requires inputs, but no inputs list was passed."


class WindowLengthNotSpecified(ZiplineError):
    """
    Raised if a user attempts to construct a term without specifying inputs and
    that term does not have class-level default inputs.
    """
    msg = (
        "{termname} requires a window_length, but no window_length was passed."
    )


class DTypeNotSpecified(ZiplineError):
    """
    Raised if a user attempts to construct a term without specifying dtype and
    that term does not have class-level default dtype.
    """
    msg = (
        "{termname} requires a dtype, but no dtype was passed."
    )


class BadPercentileBounds(ZiplineError):
    """
    Raised by API functions accepting percentile bounds when the passed bounds
    are invalid.
    """
    msg = (
        "Percentile bounds must fall between 0.0 and 100.0, and min must be "
        "less than max."
        "\nInputs were min={min_percentile}, max={max_percentile}."
    )


class UnknownRankMethod(ZiplineError):
    """
    Raised during construction of a Rank factor when supplied a bad Rank
    method.
    """
    msg = (
        "Unknown ranking method: '{method}'. "
        "`method` must be one of {choices}"
    )


class AttachPipelineAfterInitialize(ZiplineError):
    """
    Raised when a user tries to call add_pipeline outside of initialize.
    """
    msg = (
        "Attempted to attach a pipeline after initialize()."
        "attach_pipeline() can only be called during initialize."
    )


class PipelineOutputDuringInitialize(ZiplineError):
    """
    Raised when a user tries to call `pipeline_output` during initialize.
    """
    msg = (
        "Attempted to call pipeline_output() during initialize. "
        "pipeline_output() can only be called once initialize has completed."
    )


class NoSuchPipeline(ZiplineError, KeyError):
    """
    Raised when a user tries to access a non-existent pipeline by name.
    """
    msg = (
        "No pipeline named '{name}' exists. Valid pipeline names are {valid}. "
        "Did you forget to call attach_pipeline()?"
    )


class UnsupportedDataType(ZiplineError):
    """
    Raised by CustomFactors with unsupported dtypes.
    """
    msg = "CustomFactors with dtype {dtype} are not supported."


class NoFurtherDataError(ZiplineError):
    """
    Raised by calendar operations that would ask for dates beyond the extent of
    our known data.
    """
    # This accepts an arbitrary message string because it's used in more places
    # that can be usefully templated.
    msg = '{msg}'


class UnsupportedDatetimeFormat(ZiplineError):
    """
    Raised when an unsupported datetime is passed to an API method.
    """
    msg = ("The input '{input}' passed to '{method}' is not "
           "coercible to a pandas.Timestamp object.")


class PositionTrackerMissingAssetFinder(ZiplineError):
    """
    Raised by a PositionTracker if it is asked to update an Asset but does not
    have an AssetFinder
    """
    msg = (
        "PositionTracker attempted to update its Asset information but does "
        "not have an AssetFinder. This may be caused by a failure to properly "
        "de-serialize a TradingAlgorithm."
    )
