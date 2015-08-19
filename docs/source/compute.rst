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
registering instances of :class:`~zipline.modelling.filter.Filter`,
:class:`~zipline.modelling.filter.Factor`,
:class:`~zipline.modelling.classifier.Classifier`, which take instances of
:class:`~zipline.data.dataset.DataSet` as inputs. Internally, Zipline converts
these objects into a `Directed Acyclic Graph`_ and feeds them into a compute
engine to be processed efficiently.[#dasknote]

The Compute API is intended for use in the context of trading algorithms that
screen large universes of stocks. It can also used standalone with Zipline,
allowing users to prototype the modelling-based components of a larger
algorithm without having to run full simulations.

Core Concepts
-------------

It's easiest to understand the core concepts of the Compute API after walking
through a simple example.

In this algorithm, we compute a 10-day Simple Moving Average of close price.
We then filter out stocks each day with an average price of $5.00 or less.
Finally, we print the first 5 rows of the computed result during every
handle_data.

.. code-block:: Python

   from zipline.data.equities import USEquityPricing
   from zipline.modelling.factors import SimpleMovingAverage

   def initialize(context):
       sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)
       add_factor(sma, name='sma')
       add_filter(sma > 5)

   def handle_data(context, data):
       print data.factors.head(5)

DataSets
~~~~~~~~

.. warning::

   The names and means of importing Compute API datasets is subject to change
   in the near future.

:class:`~zipline.data.DataSet` objects are containers for the primitive inputs
for Compute API expressions.  Each ``DataSet`` holds some number of ``Column``
objects, which can be passed as inputs to ``Filters``, ``Factors``, and
``Classifiers``.

There are currently two DataSets available on Quantopian.  The simpler of the
two is :class:`~zipline.data.equities.USEquityPricing`, which can be imported
from :mod:`zipline.data.equities`, and has ``open``, ``high``, ``close``,
``low``, and ``volume`` columns.  More complex is the :class:`Fundamentals`
dataset.  It is currently built into the default algorithm namespace on
Quantopian, but will ultimately be imported from a new ``quantopian.datasets``
module.


Declarative Computation
~~~~~~~~~~~~~~~~~~~~~~~

Most of the important work in our example algorithm happens in the user's
``initialize`` method.

When we execute

.. code-block:: Python

   sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)

we construct an instance of :class:`~zipline.modelling.factor.Factor`.  **No
computation has happened at this point!** All we've done is constructed an
object that knows how to compute a moving average when it's passed data in a
specific format.

.. figure :: images/Factor.svg
   :width: 100 %
   :alt: Example Factor Computation
   :align: right


API Reference
-------------

.. autoclass:: zipline.modelling.factor.Factor
   :members:

.. autoclass:: zipline.modelling.factor.factor.Rank
   :members:

.. autoclass:: zipline.modelling.factor.CustomFactor
   :members:

.. autoclass:: zipline.modelling.factor.technical.RSI
   :members:

.. autoclass:: zipline.modelling.factor.technical.VWAP
   :members:

.. autoclass:: zipline.modelling.factor.technical.SimpleMovingAverage
   :members:

.. autoclass:: zipline.modelling.factor.technical.MaxDrawdown
   :members:

.. autoclass:: zipline.modelling.filter.Filter
   :members:

.. autoclass:: zipline.modelling.filter.SequencedFilter
   :members:

.. autoclass:: zipline.modelling.filter.PercentileFilter
   :members:

.. autoclass:: zipline.data.equities.USEquityPricing

   .. autoattribute:: zipline.data.equities.USEquityPricing.open
   .. autoattribute:: zipline.data.equities.USEquityPricing.high
   .. autoattribute:: zipline.data.equities.USEquityPricing.low
   .. autoattribute:: zipline.data.equities.USEquityPricing.close
   .. autoattribute:: zipline.data.equities.USEquityPricing.volume


.. _Dask: http://dask.pydata.org
.. _Blaze: http://blaze.pydata.org
.. _`Directed Acyclic Graph`: https://en.wikipedia.org/wiki/Directed_acyclic_graph

.. rubric:: Footnotes
.. [#dasknote] This approach to the problem of working with large datasets is
               similar to that of many other PyData ecosystem libraries, most
               notably the Dask_ and Blaze_ projects.
