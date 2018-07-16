.. _pipeline-api:

.. py:module:: zipline.pipeline

Pipeline API
------------

.. note::

   This page explains how to configure and run Pipelines using your own
   data. It assumes that you're already familiar with the basic concepts of
   Pipeline API. For an in-depth tutorial on using the Pipeline API to define
   cross-sectional trailing-window computations, see the `Pipeline Tutorial on Quantopian`_.

Many algorithms depend on calculations that follow a specific pattern:

    Every day, for some set of data sources, fetch the last N days' worth of
    data for a large number of assets and apply a reduction function to produce
    a single value per asset.

This kind of calculation is called a **cross-sectional trailing-window**
computation.

A simple example of a cross-sectional trailing-window computation is
"close-to-close daily returns", which has the form:

    Every day, fetch the last two days of close prices for all assets. For each
    asset, calculate the percent change between the asset's previous close
    price and its current close price.

The purpose of the **Pipeline API** is to make it easy to define and execute
cross-sectional trailing-window computations on large, dynamic universes of
assets.

Important Concepts
~~~~~~~~~~~~~~~~~~

Datasets and Loaders
````````````````````

There are many source of financial data in the world, and there are equally
many ways to store that data. It is relatively unlikely that any two Zipline
users will have exactly the same data in exactly the same format. In spite of
this diversity of inputs, however,

One of the important ideas within the **Pipeline API** is to separate the
abstract description of a dataset (e.g., "pricing data", or "corporate
fundamentals"), from the concrete way that the dataset is loaded into memory
from persistent storage.

from concrete sources of data

The distinguishing between the **description** of a dataset, from the concrete
source of data for


One of the goals of the Pipeline API is to allow users to define computations
that require a particular kind of data while being independent of the source of
that data. 

(represented as a subclass of :class:`zipline.pipeline.data.DataSet`)

For example, the :class:`~factors.DailyReturns` factor computes close-to-close
daily returns, which depends on knowing the last two days of close prices, and
the :class:`~factors.VWAP` factor computes volume-weighted average price, which
requires trailing closes as well as trailing volumes.


Expressions and Graphs
``````````````````````

The core idea behind the Pipeline API is that it allows users to create an
object, called a :class:`Pipeline`, which represents a collection of
computations to be performed every day using data from previously-available
days.


These computations often share common inputs (for example, we may
want to perform multiple computations that depend on daily close prices)

foo

:class

.. _`Pipeline Tutorial on Quantopian` : https://www.quantopian.com/tutorials/pipeline
