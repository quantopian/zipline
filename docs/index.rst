.. QSim documentation master file, created by
   sphinx-quickstart on Wed Feb  8 15:29:56 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contents:

.. toctree::
   :maxdepth: 2

   notes.rst
   modules.rst
   messaging.rst

Quantopian Simulator: QSim
================================

Qsim runs backtests using asynchronous components and zeromq messaging for communication and coordination. 

Simulator is the heart of QSim, and the primary access point for creating, launching, and tracking simulations. You can find it in :py:class:`~zipline.core.Simulator`

Simulator Sub-Components
==========================

Each simulation contains numerous subcomponents, each operating asynchronously from all others, and communicating
via zeromq.

DataSources
--------------------

A DataSource represents a historical event record, which will be played back during simulation. A simulation may have one or more DataSources, which will be combined in DataFeed. Generally, datasources read records from a persistent store (db, csv file, remote service), format the messages for downstream simulation components, and send them to a PUSH socket. See the base class for all datasources :py:class:`~zipline.sources.DataSource`

DataFeed
--------------------

All simulations start with a collection of :py:class:`DataSources <zipline.sources.DataSource>`, which need to be fed to an algorithm. Each :py:class:`~zipline.sources.DataSource`can contain events of differing content (trades, quotes, corporate event) and frequency (quarterly, intraday). To simplify the process of managing the data sources, :py:class:`~zipline.core.DataFeed` can receive events from multiple :py:class:`DataSources <zipline.sources.DataSource>` and combine them into a serial chronological stream. 

Transforms
--------------------

Often, an algorithm will require a running calculation on top of a :py:class:`~zipline.sources.DataSource`, or on the consolidated feed. A simple example is a technical indicator or a moving average. Transforms can be described in :py:class:`~zipline.core.Simulator`'s configuration. Subclass :py:class:`~zipline.transforms.core.Transform` to add your own Transform. Transforms must hold their own state between events, and serialize their current values into messages.


Data Alignment
--------------------

Like Datasources, Transforms have differing frequencies and results. Simulator manages the results of parallel transforms and **aligns** transform results with the raw DataFeed. Client algorithms simply receive a map of data, which includes the current event and all the transformed values. 

Time Compression
--------------------


Review the unit test coverage_.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _coverage: cover/index.html	