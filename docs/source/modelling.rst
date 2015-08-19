..
   --- BEGIN EMACS POWER USER STUFF ---
   Local Variables:
   mode: rst
   compile-command: "make -C .. html"
   End:
   --- END EMACS POWER USER STUFF ---

Modelling API
=============

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
style of algorithm.  Users interact with the **Modelling API** by creating and
registering instances of :class:`~zipline.data.DataSet`,
:class:`~zipline.modelling.filter.Factor` and
:class:`~zipline.modelling.filter.Filter`. Zipline converts these objects
internally into a `Directed Acyclic Graph`_ representation and feeds them into
a compute engine to be processed in an efficient manner.[#dasknote]

The Modelling API is intended for use in the context of trading algorithms that
screen large universes of stocks. It can also used standalone, allowing users
to prototype the modelling-based components of a larger algorithm without
having to run full simulations.

.. toctree::
    :maxdepth: 2

    modelling-overview.rst
    modelling-design.rst
    modelling-usage.rst

Quickstart
----------



Factors
-------

.. figure :: images/Factor.svg
   :width: 100 %
   :alt: Example Factor Computation
   :align: right


Filters
-------

Stuff about Filters

Classifiers
-----------

Engines
-------

Stuff about Engines.

.. autoclass:: zipline.modelling.factor.Factor

   :autofunction:
   :members: rank, percentile_between
   :special-members:
