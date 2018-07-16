API Reference
-------------

Running a Backtest
~~~~~~~~~~~~~~~~~~

.. autofunction:: zipline.run_algorithm(...)

Algorithm API
~~~~~~~~~~~~~

The following methods are available for use in the ``initialize``,
``handle_data``, and ``before_trading_start`` API functions.

In all listed functions, the ``self`` argument is implicitly the
currently-executing :class:`~zipline.algorithm.TradingAlgorithm` instance.

Data Object
```````````

.. autoclass:: zipline.protocol.BarData
   :members:

Scheduling Functions
````````````````````

.. autofunction:: zipline.api.schedule_function

.. autoclass:: zipline.api.date_rules
   :members:
   :undoc-members:

.. autoclass:: zipline.api.time_rules
   :members:

Orders
``````

.. autofunction:: zipline.api.order

.. autofunction:: zipline.api.order_value

.. autofunction:: zipline.api.order_percent

.. autofunction:: zipline.api.order_target

.. autofunction:: zipline.api.order_target_value

.. autofunction:: zipline.api.order_target_percent

.. autoclass:: zipline.finance.execution.ExecutionStyle
   :members:

.. autoclass:: zipline.finance.execution.MarketOrder

.. autoclass:: zipline.finance.execution.LimitOrder

.. autoclass:: zipline.finance.execution.StopOrder

.. autoclass:: zipline.finance.execution.StopLimitOrder

.. autofunction:: zipline.api.get_order

.. autofunction:: zipline.api.get_open_orders

.. autofunction:: zipline.api.cancel_order

Order Cancellation Policies
'''''''''''''''''''''''''''

.. autofunction:: zipline.api.set_cancel_policy

.. autoclass:: zipline.finance.cancel_policy.CancelPolicy
   :members:

.. autofunction:: zipline.api.EODCancel

.. autofunction:: zipline.api.NeverCancel


Assets
``````

.. autofunction:: zipline.api.symbol

.. autofunction:: zipline.api.symbols

.. autofunction:: zipline.api.future_symbol

.. autofunction:: zipline.api.set_symbol_lookup_date

.. autofunction:: zipline.api.sid


Trading Controls
````````````````

Zipline provides trading controls to help ensure that the algorithm is
performing as expected. The functions help protect the algorithm from certian
bugs that could cause undesirable behavior when trading with real money.

.. autofunction:: zipline.api.set_do_not_order_list

.. autofunction:: zipline.api.set_long_only

.. autofunction:: zipline.api.set_max_leverage

.. autofunction:: zipline.api.set_max_order_count

.. autofunction:: zipline.api.set_max_order_size

.. autofunction:: zipline.api.set_max_position_size


Simulation Parameters
`````````````````````

.. autofunction:: zipline.api.set_benchmark

Commission Models
'''''''''''''''''

.. autofunction:: zipline.api.set_commission

.. autoclass:: zipline.finance.commission.CommissionModel
   :members:

.. autoclass:: zipline.finance.commission.PerShare

.. autoclass:: zipline.finance.commission.PerTrade

.. autoclass:: zipline.finance.commission.PerDollar

Slippage Models
'''''''''''''''

.. autofunction:: zipline.api.set_slippage

.. autoclass:: zipline.finance.slippage.SlippageModel
   :members:

.. autoclass:: zipline.finance.slippage.FixedSlippage

.. autoclass:: zipline.finance.slippage.VolumeShareSlippage

Pipeline
````````

For more information, see :ref:`pipeline-api`

.. autofunction:: zipline.api.attach_pipeline

.. autofunction:: zipline.api.pipeline_output


Miscellaneous
`````````````

.. autofunction:: zipline.api.record

.. autofunction:: zipline.api.get_environment

.. autofunction:: zipline.api.fetch_csv

Blotters
~~~~~~~~

.. autoclass:: zipline.finance.blotter.blotter.Blotter
   :members:

.. autoclass:: zipline.finance.blotter.SimulationBlotter
   :members:

.. _pipeline-api:

Pipeline API
~~~~~~~~~~~~

.. autoclass:: zipline.pipeline.Pipeline
   :members:
   :member-order: groupwise

.. autoclass:: zipline.pipeline.CustomFactor
   :members:
   :member-order: groupwise

.. autoclass:: zipline.pipeline.filters.Filter
   :members: __and__, __or__
   :exclude-members: dtype

.. autoclass:: zipline.pipeline.factors.Factor
   :members: bottom, deciles, demean, linear_regression, pearsonr,
             percentile_between, quantiles, quartiles, quintiles, rank,
             spearmanr, top, winsorize, zscore, isnan, notnan, isfinite, eq,
             __add__, __sub__, __mul__, __div__, __mod__, __pow__, __lt__,
             __le__, __ne__, __ge__, __gt__
   :exclude-members: dtype
   :member-order: bysource

.. autoclass:: zipline.pipeline.term.Term
   :members:
   :exclude-members: compute_extra_rows, dependencies, inputs, mask, windowed

.. autoclass:: zipline.pipeline.data.DataSet
   :members:

.. autoclass:: zipline.pipeline.data.USEquityPricing
   :members: open, high, low, close, volume
   :undoc-members:

Built-in Factors
````````````````

.. autoclass:: zipline.pipeline.factors.AverageDollarVolume
   :members:

.. autoclass:: zipline.pipeline.factors.BollingerBands
   :members:

.. autoclass:: zipline.pipeline.factors.BusinessDaysSincePreviousEvent
   :members:

.. autoclass:: zipline.pipeline.factors.BusinessDaysUntilNextEvent
   :members:

.. autoclass:: zipline.pipeline.factors.DailyReturns
   :members:

.. autoclass:: zipline.pipeline.factors.ExponentialWeightedMovingAverage
   :members:

.. autoclass:: zipline.pipeline.factors.ExponentialWeightedMovingStdDev
   :members:

.. autoclass:: zipline.pipeline.factors.Latest
   :members:

.. autoclass:: zipline.pipeline.factors.MACDSignal
   :members:

.. autoclass:: zipline.pipeline.factors.MaxDrawdown
   :members:

.. autoclass:: zipline.pipeline.factors.Returns
   :members:

.. autoclass:: zipline.pipeline.factors.RollingLinearRegressionOfReturns
   :members:

.. autoclass:: zipline.pipeline.factors.RollingPearsonOfReturns
   :members:

.. autoclass:: zipline.pipeline.factors.RollingSpearmanOfReturns
   :members:

.. autoclass:: zipline.pipeline.factors.SimpleBeta
   :members:

.. autoclass:: zipline.pipeline.factors.RSI
   :members:

.. autoclass:: zipline.pipeline.factors.SimpleMovingAverage
   :members:

.. autoclass:: zipline.pipeline.factors.VWAP
   :members:

.. autoclass:: zipline.pipeline.factors.WeightedAverageValue
   :members:

Built-in Filters
````````````````

.. autoclass:: zipline.pipeline.filters.All
   :members:

.. autoclass:: zipline.pipeline.filters.AllPresent
   :members:

.. autoclass:: zipline.pipeline.filters.Any
   :members:

.. autoclass:: zipline.pipeline.filters.AtLeastN
   :members:

.. autoclass:: zipline.pipeline.filters.SingleAsset
   :members:

.. autoclass:: zipline.pipeline.filters.StaticAssets
   :members:

.. autoclass:: zipline.pipeline.filters.StaticSids
   :members:

Pipeline Engine
```````````````

.. autoclass:: zipline.pipeline.engine.PipelineEngine
   :members: run_pipeline, run_chunked_pipeline
   :member-order: bysource

.. autoclass:: zipline.pipeline.engine.SimplePipelineEngine
   :members: __init__, run_pipeline, run_chunked_pipeline
   :member-order: bysource

.. autofunction:: zipline.pipeline.engine.default_populate_initial_workspace

Data Loaders
````````````

.. autoclass:: zipline.pipeline.loaders.equity_pricing_loader.USEquityPricingLoader
   :members: __init__, from_files, load_adjusted_array
   :member-order: bysource

Asset Metadata
~~~~~~~~~~~~~~

.. autoclass:: zipline.assets.Asset
   :members:

.. autoclass:: zipline.assets.Equity
   :members:

.. autoclass:: zipline.assets.Future
   :members:

.. autoclass:: zipline.assets.AssetConvertible
   :members:


Trading Calendar API
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: zipline.utils.calendars.get_calendar

.. autoclass:: zipline.utils.calendars.TradingCalendar
   :members:

.. autofunction:: zipline.utils.calendars.register_calendar

.. autofunction:: zipline.utils.calendars.register_calendar_type

.. autofunction:: zipline.utils.calendars.deregister_calendar

.. autofunction:: zipline.utils.calendars.clear_calendars


Data API
~~~~~~~~

Writers
```````
.. autoclass:: zipline.data.minute_bars.BcolzMinuteBarWriter
   :members:

.. autoclass:: zipline.data.us_equity_pricing.BcolzDailyBarWriter
   :members:

.. autoclass:: zipline.data.us_equity_pricing.SQLiteAdjustmentWriter
   :members:

.. autoclass:: zipline.assets.AssetDBWriter
   :members:

Readers
```````
.. autoclass:: zipline.data.minute_bars.BcolzMinuteBarReader
   :members:

.. autoclass:: zipline.data.us_equity_pricing.BcolzDailyBarReader
   :members:

.. autoclass:: zipline.data.us_equity_pricing.SQLiteAdjustmentReader
   :members:

.. autoclass:: zipline.assets.AssetFinder
   :members:

.. autoclass:: zipline.data.data_portal.DataPortal
   :members:

.. autoclass:: zipline.sources.benchmark_source.BenchmarkSource
   :members:

Bundles
```````
.. autofunction:: zipline.data.bundles.register

.. autofunction:: zipline.data.bundles.ingest(name, environ=os.environ, date=None, show_progress=True)

.. autofunction:: zipline.data.bundles.load(name, environ=os.environ, date=None)

.. autofunction:: zipline.data.bundles.unregister

.. data:: zipline.data.bundles.bundles

   The bundles that have been registered as a mapping from bundle name to bundle
   data. This mapping is immutable and may only be updated through
   :func:`~zipline.data.bundles.register` or
   :func:`~zipline.data.bundles.unregister`.


Risk Metrics
~~~~~~~~~~~~

Algorithm State
```````````````

.. autoclass:: zipline.finance.ledger.Ledger
   :members:

.. autoclass:: zipline.protocol.Portfolio
   :members:

.. autoclass:: zipline.protocol.Account
   :members:

.. autoclass:: zipline.finance.ledger.PositionTracker
   :members:

.. autoclass:: zipline.finance._finance_ext.PositionStats

Built-in Metrics
````````````````

.. autoclass:: zipline.finance.metrics.metric.SimpleLedgerField

.. autoclass:: zipline.finance.metrics.metric.DailyLedgerField

.. autoclass:: zipline.finance.metrics.metric.StartOfPeriodLedgerField

.. autoclass:: zipline.finance.metrics.metric.StartOfPeriodLedgerField

.. autoclass:: zipline.finance.metrics.metric.Returns

.. autoclass:: zipline.finance.metrics.metric.BenchmarkReturnsAndVolatility

.. autoclass:: zipline.finance.metrics.metric.CashFlow

.. autoclass:: zipline.finance.metrics.metric.Orders

.. autoclass:: zipline.finance.metrics.metric.Transactions

.. autoclass:: zipline.finance.metrics.metric.Positions

.. autoclass:: zipline.finance.metrics.metric.ReturnsStatistic

.. autoclass:: zipline.finance.metrics.metric.AlphaBeta

.. autoclass:: zipline.finance.metrics.metric.MaxLeverage

Metrics Sets
````````````

.. autofunction:: zipline.finance.metrics.register

.. autofunction:: zipline.finance.metrics.load

.. autofunction:: zipline.finance.metrics.unregister

.. data:: zipline.data.finance.metrics.metrics_sets

   The metrics sets that have been registered as a mapping from metrics set name
   to load function. This mapping is immutable and may only be updated through
   :func:`~zipline.finance.metrics.register` or
   :func:`~zipline.finance.metrics.unregister`.


Utilities
~~~~~~~~~

Caching
```````

.. autoclass:: zipline.utils.cache.CachedObject

.. autoclass:: zipline.utils.cache.ExpiringCache

.. autoclass:: zipline.utils.cache.dataframe_cache

.. autoclass:: zipline.utils.cache.working_file

.. autoclass:: zipline.utils.cache.working_dir

Command Line
````````````
.. autofunction:: zipline.utils.cli.maybe_show_progress
