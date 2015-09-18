==========
Motivation
==========

Many trading algorithms are variations on the following structure:

1. For each asset in a known (large) universe, compute **N** scalar values for
   the asset based on a trailing window of data.
2. Select a smaller "tradeable universe" of assets based on the values computed
   in **(1)**.
3. Calculate desired portfolio weights on the trading universe computed in
   **(2)**.
4. Place orders to move the algorithm's current portfolio allocations to the
   desired weights computed in **(3)**.

The :mod:`zipline.modelling` module provides a framework for expressing this
style of algorithm.  Users interact with the **Modeling API** by creating and
manipulating objects that form a **computational pipeline**.  Internally,
Zipline converts these objects into a `Directed Acyclic Graph`_ and feeds them
into a compute engine to be processed efficiently. [#dasknote]_

The Modeling API was designed for use in the context of trading algorithms that
screen large universes of stocks. It can also used standalone, however,
allowing users to prototype models and quickly test hypotheses without running
full backtests.

.. _`Directed Acyclic Graph`: https://en.wikipedia.org/wiki/Directed_acyclic_graph
.. _Dask: http://dask.pydata.org
.. _Blaze: http://blaze.pydata.org

.. rubric:: Footnotes
.. [#dasknote] This approach to the problem of working with large datasets is
               similar to that of many other PyData ecosystem libraries, most
               notably the Dask_ and Blaze_ projects.
