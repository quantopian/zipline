import collections
from zipline.assets import Asset, Equity, Future
from zipline.assets.futures import FutureChain
from zipline.finance.asset_restrictions import Restrictions
from zipline.finance.cancel_policy import CancelPolicy
from zipline.pipeline import Pipeline
from zipline.protocol import Order
from zipline.utils.events import EventRule
from zipline.utils.security_list import SecurityList


def attach_pipeline(pipeline, name, chunks=None):
    """Register a pipeline to be computed at the start of each day.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to have computed.
    name : str
        The name of the pipeline.
    chunks : int or iterator, optional
        The number of days to compute pipeline results for. Increasing
        this number will make it longer to get the first results but
        may improve the total runtime of the simulation. If an iterator
        is passed, we will run in chunks based on values of the itereator.

    Returns
    -------
    pipeline : Pipeline
        Returns the pipeline that was attached unchanged.

    See Also
    --------
    :func:`zipline.api.pipeline_output`
    """

def batch_order_target_percent(weights):
    """Place orders towards a given portfolio of weights.

    Parameters
    ----------
    weights : collections.Mapping[Asset -> float]

    Returns
    -------
    order_ids : pd.Series[Asset -> str]
        The unique identifiers for the orders that were placed.

    See Also
    --------
    :func:`zipline.api.order_target_percent`
    """

def cancel_order(order_param):
    """Cancel an open order.

    Parameters
    ----------
    order_param : str or Order
        The order_id or order object to cancel.
    """

def continuous_future(root_symbol_str, offset, roll):
    """Create a specifier for a continuous contract.

    Parameters
    ----------
    root_symbol_str : str
        The root symbol for the future chain.

    offset : int
        The distance from the primary contract.

    roll_style : str
        How rolls are determined.

    Returns
    -------
    continuous_future : ContinuousFuture
        The continuous future specifier.
    """

def fetch_csv(url, pre_func=None, post_func=None, date_column='date', date_format=None, timezone='UTC', symbol=None, mask=True, symbol_column=None, special_params_checker=None, **kwargs):
    """Fetch a csv from a remote url and register the data so that it is
    queryable from the ``data`` object.

    Parameters
    ----------
    url : str
        The url of the csv file to load.
    pre_func : callable[pd.DataFrame -> pd.DataFrame], optional
        A callback to allow preprocessing the raw data returned from
        fetch_csv before dates are paresed or symbols are mapped.
    post_func : callable[pd.DataFrame -> pd.DataFrame], optional
        A callback to allow postprocessing of the data after dates and
        symbols have been mapped.
    date_column : str, optional
        The name of the column in the preprocessed dataframe containing
        datetime information to map the data.
    date_format : str, optional
        The format of the dates in the ``date_column``. If not provided
        ``fetch_csv`` will attempt to infer the format. For information
        about the format of this string, see :func:`pandas.read_csv`.
    timezone : tzinfo or str, optional
        The timezone for the datetime in the ``date_column``.
    symbol : str, optional
        If the data is about a new asset or index then this string will
        be the name used to identify the values in ``data``. For example,
        one may use ``fetch_csv`` to load data for VIX, then this field
        could be the string ``'VIX'``.
    mask : bool, optional
        Drop any rows which cannot be symbol mapped.
    symbol_column : str
        If the data is attaching some new attribute to each asset then this
        argument is the name of the column in the preprocessed dataframe
        containing the symbols. This will be used along with the date
        information to map the sids in the asset finder.
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    csv_data_source : zipline.sources.requests_csv.PandasRequestsCSV
        A requests source that will pull data from the url specified.
    """

def future_symbol(symbol):
    """Lookup a futures contract with a given symbol.

    Parameters
    ----------
    symbol : str
        The symbol of the desired contract.

    Returns
    -------
    future : Future
        The future that trades with the name ``symbol``.

    Raises
    ------
    SymbolNotFound
        Raised when no contract named 'symbol' is found.
    """

def get_datetime(tz=None):
    """
Returns the current simulation datetime.

Parameters
----------
tz : tzinfo or str, optional
    The timezone to return the datetime in. This defaults to utc.

Returns
-------
dt : datetime
    The current simulation datetime converted to ``tz``.
    """

def get_environment(field='platform'):
    """Query the execution environment.

    Parameters
    ----------
    field : {'platform', 'arena', 'data_frequency',
             'start', 'end', 'capital_base', 'platform', '*'}
        The field to query. The options have the following meanings:
          arena : str
              The arena from the simulation parameters. This will normally
              be ``'backtest'`` but some systems may use this distinguish
              live trading from backtesting.
          data_frequency : {'daily', 'minute'}
              data_frequency tells the algorithm if it is running with
              daily data or minute data.
          start : datetime
              The start date for the simulation.
          end : datetime
              The end date for the simulation.
          capital_base : float
              The starting capital for the simulation.
          platform : str
              The platform that the code is running on. By default this
              will be the string 'zipline'. This can allow algorithms to
              know if they are running on the Quantopian platform instead.
          * : dict[str -> any]
              Returns all of the fields in a dictionary.

    Returns
    -------
    val : any
        The value for the field queried. See above for more information.

    Raises
    ------
    ValueError
        Raised when ``field`` is not a valid option.
    """

def get_order(order_id):
    """Lookup an order based on the order id returned from one of the
    order functions.

    Parameters
    ----------
    order_id : str
        The unique identifier for the order.

    Returns
    -------
    order : Order
        The order object.
    """

def history(bar_count, frequency, field, ffill=True):
    """DEPRECATED: use ``data.history`` instead.
    """

def order(asset, amount, limit_price=None, stop_price=None, style=None):
    """Place an order.

    Parameters
    ----------
    asset : Asset
        The asset that this order is for.
    amount : int
        The amount of shares to order. If ``amount`` is positive, this is
        the number of shares to buy or cover. If ``amount`` is negative,
        this is the number of shares to sell or short.
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle, optional
        The execution style for the order.

    Returns
    -------
    order_id : str or None
        The unique identifier for this order, or None if no order was
        placed.

    Notes
    -----
    The ``limit_price`` and ``stop_price`` arguments provide shorthands for
    passing common execution styles. Passing ``limit_price=N`` is
    equivalent to ``style=LimitOrder(N)``. Similarly, passing
    ``stop_price=M`` is equivalent to ``style=StopOrder(M)``, and passing
    ``limit_price=N`` and ``stop_price=M`` is equivalent to
    ``style=StopLimitOrder(N, M)``. It is an error to pass both a ``style``
    and ``limit_price`` or ``stop_price``.

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order_value`
    :func:`zipline.api.order_percent`
    """

def order_percent(asset, percent, limit_price=None, stop_price=None, style=None):
    """Place an order in the specified asset corresponding to the given
    percent of the current portfolio value.

    Parameters
    ----------
    asset : Asset
        The asset that this order is for.
    percent : float
        The percentage of the porfolio value to allocate to ``asset``.
        This is specified as a decimal, for example: 0.50 means 50%.
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle
        The execution style for the order.

    Returns
    -------
    order_id : str
        The unique identifier for this order.

    Notes
    -----
    See :func:`zipline.api.order` for more information about
    ``limit_price``, ``stop_price``, and ``style``

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order`
    :func:`zipline.api.order_value`
    """

def order_target(asset, target, limit_price=None, stop_price=None, style=None):
    """Place an order to adjust a position to a target number of shares. If
    the position doesn't already exist, this is equivalent to placing a new
    order. If the position does exist, this is equivalent to placing an
    order for the difference between the target number of shares and the
    current number of shares.

    Parameters
    ----------
    asset : Asset
        The asset that this order is for.
    target : int
        The desired number of shares of ``asset``.
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle
        The execution style for the order.

    Returns
    -------
    order_id : str
        The unique identifier for this order.


    Notes
    -----
    ``order_target`` does not take into account any open orders. For
    example:

    .. code-block:: python

       order_target(sid(0), 10)
       order_target(sid(0), 10)

    This code will result in 20 shares of ``sid(0)`` because the first
    call to ``order_target`` will not have been filled when the second
    ``order_target`` call is made.

    See :func:`zipline.api.order` for more information about
    ``limit_price``, ``stop_price``, and ``style``

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order`
    :func:`zipline.api.order_target_percent`
    :func:`zipline.api.order_target_value`
    """

def order_target_percent(asset, target, limit_price=None, stop_price=None, style=None):
    """Place an order to adjust a position to a target percent of the
    current portfolio value. If the position doesn't already exist, this is
    equivalent to placing a new order. If the position does exist, this is
    equivalent to placing an order for the difference between the target
    percent and the current percent.

    Parameters
    ----------
    asset : Asset
        The asset that this order is for.
    target : float
        The desired percentage of the porfolio value to allocate to
        ``asset``. This is specified as a decimal, for example:
        0.50 means 50%.
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle
        The execution style for the order.

    Returns
    -------
    order_id : str
        The unique identifier for this order.

    Notes
    -----
    ``order_target_value`` does not take into account any open orders. For
    example:

    .. code-block:: python

       order_target_percent(sid(0), 10)
       order_target_percent(sid(0), 10)

    This code will result in 20% of the portfolio being allocated to sid(0)
    because the first call to ``order_target_percent`` will not have been
    filled when the second ``order_target_percent`` call is made.

    See :func:`zipline.api.order` for more information about
    ``limit_price``, ``stop_price``, and ``style``

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order`
    :func:`zipline.api.order_target`
    :func:`zipline.api.order_target_value`
    """

def order_target_value(asset, target, limit_price=None, stop_price=None, style=None):
    """Place an order to adjust a position to a target value. If
    the position doesn't already exist, this is equivalent to placing a new
    order. If the position does exist, this is equivalent to placing an
    order for the difference between the target value and the
    current value.
    If the Asset being ordered is a Future, the 'target value' calculated
    is actually the target exposure, as Futures have no 'value'.

    Parameters
    ----------
    asset : Asset
        The asset that this order is for.
    target : float
        The desired total value of ``asset``.
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle
        The execution style for the order.

    Returns
    -------
    order_id : str
        The unique identifier for this order.

    Notes
    -----
    ``order_target_value`` does not take into account any open orders. For
    example:

    .. code-block:: python

       order_target_value(sid(0), 10)
       order_target_value(sid(0), 10)

    This code will result in 20 dollars of ``sid(0)`` because the first
    call to ``order_target_value`` will not have been filled when the
    second ``order_target_value`` call is made.

    See :func:`zipline.api.order` for more information about
    ``limit_price``, ``stop_price``, and ``style``

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order`
    :func:`zipline.api.order_target`
    :func:`zipline.api.order_target_percent`
    """

def order_value(asset, value, limit_price=None, stop_price=None, style=None):
    """Place an order by desired value rather than desired number of
    shares.

    Parameters
    ----------
    asset : Asset
        The asset that this order is for.
    value : float
        If the requested asset exists, the requested value is
        divided by its price to imply the number of shares to transact.
        If the Asset being ordered is a Future, the 'value' calculated
        is actually the exposure, as Futures have no 'value'.

        value > 0 :: Buy/Cover
        value < 0 :: Sell/Short
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle
        The execution style for the order.

    Returns
    -------
    order_id : str
        The unique identifier for this order.

    Notes
    -----
    See :func:`zipline.api.order` for more information about
    ``limit_price``, ``stop_price``, and ``style``

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order`
    :func:`zipline.api.order_percent`
    """

def pipeline_output(name):
    """Get the results of the pipeline that was attached with the name:
    ``name``.

    Parameters
    ----------
    name : str
        Name of the pipeline for which results are requested.

    Returns
    -------
    results : pd.DataFrame
        DataFrame containing the results of the requested pipeline for
        the current simulation date.

    Raises
    ------
    NoSuchPipeline
        Raised when no pipeline with the name `name` has been registered.

    See Also
    --------
    :func:`zipline.api.attach_pipeline`
    :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
    """

def record(*args, **kwargs):
    """Track and record values each day.

    Parameters
    ----------
    **kwargs
        The names and values to record.

    Notes
    -----
    These values will appear in the performance packets and the performance
    dataframe passed to ``analyze`` and returned from
    :func:`~zipline.run_algorithm`.
    """

def schedule_function(func, date_rule=None, time_rule=None, half_days=True, calendar=None):
    """Schedules a function to be called according to some timed rules.

    Parameters
    ----------
    func : callable[(context, data) -> None]
        The function to execute when the rule is triggered.
    date_rule : EventRule, optional
        The rule for the dates to execute this function.
    time_rule : EventRule, optional
        The rule for the times to execute this function.
    half_days : bool, optional
        Should this rule fire on half days?

    See Also
    --------
    :class:`zipline.api.date_rules`
    :class:`zipline.api.time_rules`
    """

def set_asset_restrictions(restrictions, on_error='fail'):
    """Set a restriction on which assets can be ordered.

    Parameters
    ----------
    restricted_list : Restrictions
        An object providing information about restricted assets.

    See Also
    --------
    zipline.finance.asset_restrictions.Restrictions
    """

def set_benchmark(benchmark):
    """Set the benchmark asset.

    Parameters
    ----------
    benchmark : Asset
        The asset to set as the new benchmark.

    Notes
    -----
    Any dividends payed out for that new benchmark asset will be
    automatically reinvested.
    """

def set_cancel_policy(cancel_policy):
    """Sets the order cancellation policy for the simulation.

    Parameters
    ----------
    cancel_policy : CancelPolicy
        The cancellation policy to use.

    See Also
    --------
    :class:`zipline.api.EODCancel`
    :class:`zipline.api.NeverCancel`
    """

def set_commission(commission):
    """Sets the commission model for the simulation.

    Parameters
    ----------
    commission : CommissionModel
        The commission model to use.

    See Also
    --------
    :class:`zipline.finance.commission.PerShare`
    :class:`zipline.finance.commission.PerTrade`
    :class:`zipline.finance.commission.PerDollar`
    """

def set_do_not_order_list(restricted_list, on_error='fail'):
    """Set a restriction on which assets can be ordered.

    Parameters
    ----------
    restricted_list : container[Asset], SecurityList
        The assets that cannot be ordered.
    """

def set_long_only(on_error='fail'):
    """Set a rule specifying that this algorithm cannot take short
    positions.
    """

def set_max_leverage(max_leverage):
    """Set a limit on the maximum leverage of the algorithm.

    Parameters
    ----------
    max_leverage : float
        The maximum leverage for the algorithm. If not provided there will
        be no maximum.
    """

def set_max_order_count(max_count, on_error='fail'):
    """Set a limit on the number of orders that can be placed in a single
    day.

    Parameters
    ----------
    max_count : int
        The maximum number of orders that can be placed on any single day.
    """

def set_max_order_size(asset=None, max_shares=None, max_notional=None, on_error='fail'):
    """Set a limit on the number of shares and/or dollar value of any single
    order placed for sid.  Limits are treated as absolute values and are
    enforced at the time that the algo attempts to place an order for sid.

    If an algorithm attempts to place an order that would result in
    exceeding one of these limits, raise a TradingControlException.

    Parameters
    ----------
    asset : Asset, optional
        If provided, this sets the guard only on positions in the given
        asset.
    max_shares : int, optional
        The maximum number of shares that can be ordered at one time.
    max_notional : float, optional
        The maximum value that can be ordered at one time.
    """

def set_max_position_size(asset=None, max_shares=None, max_notional=None, on_error='fail'):
    """Set a limit on the number of shares and/or dollar value held for the
    given sid. Limits are treated as absolute values and are enforced at
    the time that the algo attempts to place an order for sid. This means
    that it's possible to end up with more than the max number of shares
    due to splits/dividends, and more than the max notional due to price
    improvement.

    If an algorithm attempts to place an order that would result in
    increasing the absolute value of shares/dollar value exceeding one of
    these limits, raise a TradingControlException.

    Parameters
    ----------
    asset : Asset, optional
        If provided, this sets the guard only on positions in the given
        asset.
    max_shares : int, optional
        The maximum number of shares to hold for an asset.
    max_notional : float, optional
        The maximum value to hold for an asset.
    """

def set_slippage(slippage):
    """Set the slippage model for the simulation.

    Parameters
    ----------
    slippage : SlippageModel
        The slippage model to use.

    See Also
    --------
    :class:`zipline.finance.slippage.SlippageModel`
    """

def set_symbol_lookup_date(dt):
    """Set the date for which symbols will be resolved to their assets
    (symbols may map to different firms or underlying assets at
    different times)

    Parameters
    ----------
    dt : datetime
        The new symbol lookup date.
    """

def sid(sid):
    """Lookup an Asset by its unique asset identifier.

    Parameters
    ----------
    sid : int
        The unique integer that identifies an asset.

    Returns
    -------
    asset : Asset
        The asset with the given ``sid``.

    Raises
    ------
    SidsNotFound
        When a requested ``sid`` does not map to any asset.
    """

def symbol(symbol_str):
    """Lookup an Equity by its ticker symbol.

    Parameters
    ----------
    symbol_str : str
        The ticker symbol for the equity to lookup.

    Returns
    -------
    equity : Equity
        The equity that held the ticker symbol on the current
        symbol lookup date.

    Raises
    ------
    SymbolNotFound
        Raised when the symbols was not held on the current lookup date.

    See Also
    --------
    :func:`zipline.api.set_symbol_lookup_date`
    """

def symbols(*args):
    """Lookup multuple Equities as a list.

    Parameters
    ----------
    *args : iterable[str]
        The ticker symbols to lookup.

    Returns
    -------
    equities : list[Equity]
        The equities that held the given ticker symbols on the current
        symbol lookup date.

    Raises
    ------
    SymbolNotFound
        Raised when one of the symbols was not held on the current
        lookup date.

    See Also
    --------
    :func:`zipline.api.set_symbol_lookup_date`
    """

