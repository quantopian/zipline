*******************************************
Overview
*******************************************

Simulations
===========

:mod:`zipline` runs backtests using asynchronous components and zeromq messaging for
communication and coordination. 

:class:`.algorithm.TradingAlgorithm` is the heart of :mod:`zipline`, and the primary access point for creating,
launching, and tracking simulations. You can find it in
:py:class:`~zipline.algorithm.TradingAlgorithm`

Simulator Sub-Components
========================

Each simulation contains numerous subcomponents, each operating asynchronously
from all others, and communicating via zeromq.

DataSources
--------------------

A DataSource represents a historical event record, which will be played back
during simulation. A simulation may have one or more DataSources, which will be
combined in DataFeed. Generally, datasources read records from a persistent
store (db, csv file, remote service), format the messages for downstream
simulation components, and send them to a PUSH socket. See the base class for
all datasources :py:class:`~zipline.messaging.DataSource` and the module
holding all datasources :py:mod:`zipline.sources`

DataFeed
--------------------

All simulations start with a collection of
:py:class:`~zipline.messaging.DataSource`, which need to be fed to an
algorithm. Each :py:class:`~zipline.sources.DataSource`can contain events of
differing content (trades, quotes, corporate event) and frequency (quarterly,
intraday). To simplify the process of managing the data sources,
:py:class:`~zipline.core.DataFeed` can receive events from multiple
:py:class:`DataSources <zipline.sources.DataSource>` and combine them into a
serial chronological stream. 

Transforms
--------------------

Often, an algorithm will require a running calculation on top of a
:py:class:`~zipline.messaging.DataSource`, or on the consolidated feed. A
simple example is a technical indicator or a moving average. Transforms can be
described in :py:class:`~zipline.core.Simulator`'s configuration. Subclass
:py:class:`~zipline.transforms.core.Transform` to add your own Transform.
Transforms must hold their own state between events, and serialize their
current values into messages.


Data Alignment
--------------------

Like Datasources, Transforms have differing frequencies and results. Simulator
manages the results of parallel transforms and **aligns** transform results
with the raw DataFeed. Client algorithms simply receive a map of data, which
includes the current event and all the transformed values. 

Time Compression
--------------------

According to `this post
<https://www.quantopian.com/posts/help-with-runtime-error>`_ on the Quantopian
forums, time periods during which none of the selected SIDs were traded are
skipped.


Review the unit test coverage_.



.. _coverage: cover/index.html	
