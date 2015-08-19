..
   --- BEGIN EMACS POWER USER STUFF ---
   Local Variables:
   mode: rst
   compile-command: "make -C .. html"
   End:
   --- END EMACS POWER USER STUFF ---

   To compile the Zipline docs from Emacs, run M-x compile.
   To compile from the command line, cd to zipline_repo/sphinx-doc and run ``make html``.

Compute API
===========

.. warning::

   - All references below to ``zipline.modelling.*`` are slated to be changed in
     the near future to ``zipline.compute``.

   - The name of the ``Factor`` class is likely to change in the future, due to
     worries that it collides with a term of art that specifically refers to
     computations involving correlations with known time-series data.

     Suggestions for alternative names currently include:
       - ``Indicator``
       - ``Metric``
       - ``Function``
       - ``Signal``

     Feedback on these names (or other suggestions) would is encouraged and
     appeciated.

Motivation
----------

Many trading algorithms are variations on the following structure:

1. For each asset in a known (large) universe, compute **N** scalar values for
   the asset based on a trailing window of data.
2. Select a smaller "tradable universe" of assets based on the values computed
   in **(1)**.
3. Calculate desired portfolio weights on the trading universe computed in
   **(2)**.
4. Place orders to move the algorithm's current portfolio allocations to the
   desired weights computed in **(3)**.

The :mod:`zipline.modelling` module provides a framework for expressing this
style of algorithm.  Users interact with the **Compute API** by creating and
registering instances of :class:`~zipline.modelling.DataSet`,
:class:`~zipline.modelling.filter.Factor` and
:class:`~zipline.modelling.filter.Filter`. Zipline converts these objects
internally into a `Directed Acyclic Graph`_ representation and feeds them into
a compute engine to be processed in an efficient manner.[#dasknote]

The Compute API is intended for use in the context of trading algorithms that
screen large universes of stocks. It can also used standalone with Zipline,
allowing users to prototype the modelling-based components of a larger
algorithm without having to run full simulations.

Core Concepts
-------------

DataSets
~~~~~~~~

:class:`~zipline.data.DataSet` objects represent primitive inputs to Compute
API algorithms.

There are currently two DataSets available on Quantopian:

.. autoclass:: zipline.data.equities.USEquityPricing

   .. autoattribute:: zipline.data.equities.USEquityPricing.open
   .. autoattribute:: zipline.data.equities.USEquityPricing.high
   .. autoattribute:: zipline.data.equities.USEquityPricing.low
   .. autoattribute:: zipline.data.equities.USEquityPricing.close
   .. autoattribute:: zipline.data.equities.USEquityPricing.volume


Factors
~~~~~~~

.. figure :: images/Factor.svg
   :width: 100 %
   :alt: Example Factor Computation
   :align: right


Filters
~~~~~~~
pass


Classifiers
~~~~~~~~~~~
**NOT YET IMPLEMENTED**


Quick Start
-----------

Compute 10-day Simple Moving Average of close price.  Then, filter out stocks
with moving average price less than $5.

Set our universe to the first 10 sids that the criteria defined in initialize.

.. code-block:: Python

   from zipline.data import USEquityPricing
   from zipline.modelling.factors import SimpleMovingAverage

   def initalize(context):
       sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)
       add_factor(sma, name='sma')
       add_filter(sma > 5)

   def before_trading_start(context, data):
       update_universe(sorted(data.factors.index)[:10])

   def handle_data(context, data):
       factors = data.factors


Core Concepts
-------------

Factors
-------

.. _`Directed Acyclic Graph`: https://en.wikipedia.org/wiki/Directed_acyclic_graph
.. rubric:: Footnotes
.. [#dasknote] This approach to the problem of working with large datasets is
               similar to that of many other PyData ecosystem libraries, most
               notably the Dask_ project.

.. _Dask: http://dask.pydata.org
