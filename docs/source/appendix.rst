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

.. automodule:: zipline.api
   :members:

.. autoclass:: zipline.algorithm.TradingAlgorithm

Pipeline API
~~~~~~~~~~~~

.. autoclass:: zipline.pipeline.Pipeline
   :members:
   :member-order: groupwise

.. autoclass:: zipline.pipeline.CustomFactor
   :members:
   :member-order: groupwise

.. autoclass:: zipline.pipeline.factors.Factor
   :members: top, bottom, rank, percentile_between, isnan, notnan, isfinite,
             eq, __add__, __sub__, __mul__, __div__, __mod__, __pow__, __lt__,
             __le__, __ne__, __ge__, __gt__
   :exclude-members: dtype
   :member-order: bysource

.. autoclass:: zipline.pipeline.factors.Latest
   :members:

.. autoclass:: zipline.pipeline.factors.MaxDrawdown
   :members:

.. autoclass:: zipline.pipeline.factors.Returns
   :members:

.. autoclass:: zipline.pipeline.factors.RSI
   :members:

.. autoclass:: zipline.pipeline.factors.BusinessDaysUntilNextEarnings
   :members:

.. autoclass:: zipline.pipeline.factors.BusinessDaysSincePreviousEarnings
   :members:

.. autoclass:: zipline.pipeline.factors.SimpleMovingAverage
   :members:

.. autoclass:: zipline.pipeline.factors.VWAP
   :members:

.. autoclass:: zipline.pipeline.factors.WeightedAverageValue
   :members:

.. autoclass:: zipline.pipeline.factors.ExponentialWeightedMovingAverage
   :members:

.. autoclass:: zipline.pipeline.factors.ExponentialWeightedMovingStdDev
   :members:

.. autoclass:: zipline.pipeline.factors.AverageDollarVolume
   :members:

.. autoclass:: zipline.pipeline.factors.BollingerBands
   :members:

.. autoclass:: zipline.pipeline.filters.Filter
   :members: __and__, __or__
   :exclude-members: dtype

.. autoclass:: zipline.pipeline.data.EarningsCalendar
   :members: next_announcement, previous_announcement
   :undoc-members:

.. autoclass:: zipline.pipeline.data.USEquityPricing
   :members: open, high, low, close, volume
   :undoc-members:


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

.. autoclass:: zipline.assets.AssetFinderCachedEquities
   :members:

Bundles
```````
.. autofunction:: zipline.data.bundles.register

.. autofunction:: zipline.data.bundles.ingest(name, environ=os.environ, date=None, show_progress=True)

.. autofunction:: zipline.data.bundles.load(name, environ=os.environ, date=None)

.. autofunction:: zipline.data.bundles.unregister

.. data:: zipline.data.bundles.bundles

   The bundles that have been registered as a mapping from bundle name to bundle
   data. This mapping is immutable and should only be updated through
   :func:`~zipline.data.bundles.register` or
   :func:`~zipline.data.bundles.unregister`.

.. autofunction:: zipline.data.bundles.yahoo_equities


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
