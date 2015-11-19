API Reference
-------------

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

.. autoclass:: zipline.pipeline.filters.Filter
   :members: __and__, __or__
   :exclude-members: dtype

Asset Metadata
~~~~~~~~~~~~~~

.. autoclass:: zipline.assets.assets.Asset
   :members:

.. autoclass:: zipline.assets.assets.Equity
   :members:

.. autoclass:: zipline.assets.assets.Future
   :members:

.. autoclass:: zipline.assets.assets.AssetFinder
   :members:

.. autoclass:: zipline.assets.assets.AssetFinderCachedEquities
   :members:

.. autoclass:: zipline.assets.asset_writer.AssetDBWriter
   :members:

.. autoclass:: zipline.assets.assets.AssetConvertible
   :members:
