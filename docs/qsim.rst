============
qsim Package
============

Simulator is the heart of QSim, and the primary access point for creating, launching, and tracking simulations. You can find it in :py:class:`~qsim.core.Simulator`

Simulator Sub-Components
==========================

Each simulation contains numerous subcomponents, each operating asynchronously from all others, and communicating
via zeromq.

DataSources
--------------------

A DataSource represents a historical event record, which will be played back during simulation. A simulation may have one or more DataSources, which will be combined in DataFeed. Generally, datasources read records from a persistent store (db, csv file, remote service), format the messages for downstream simulation components, and send them to a PUSH socket. See the base class for all datasources :py:class:`~qsim.sources.DataSource`

DataFeed
--------------------

All simulations start with a collection of :py:class:`DataSources <qsim.sources.DataSource>`, which need to be fed to an algorithm. Each :py:class:`~qsim.sources.DataSource`can contain events of differing content (trades, quotes, corporate event) and frequency (quarterly, intraday). To simplify the process of managing the data sources, :py:class:`~qsim.core.DataFeed` can receive events from multiple :py:class:`DataSources <qsim.sources.DataSource>` and combine them into a serial chronological stream. 

Transforms
--------------------

Often, an algorithm will require a running calculation on top of a :py:class:`~qsim.sources.DataSource`, or on the consolidated feed. A simple example is a technical indicator or a moving average. Transforms can be described in :py:class:`~qsim.core.Simulator`'s configuration. Subclass :py:class:`~qsim.transforms.core.Transform` to add your own Transform. Transforms must hold their own state between events, and serialize their current values into messages.


Data Alignment
--------------------

Like Datasources, Transforms have differing frequencies and results. Simulator manages the results of parallel transforms and **aligns** transform results with the raw DataFeed. Client algorithms simply receive a map of data, which includes the current event and all the transformed values. 



QSim API
===========================

:mod:`qsim` Package
-------------------

.. automodule:: qsim.__init__
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`config` Module
--------------------

.. automodule:: qsim.config
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`core` Module
------------------

.. automodule:: qsim.core
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`messaging` Module
-----------------------

.. automodule:: qsim.messaging
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`sources` Module
---------------------

.. automodule:: qsim.sources
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`util` Module
------------------

.. automodule:: qsim.util
    :members:
    :undoc-members:
    :show-inheritance:

Subpackages
-----------

.. toctree::

    qsim.test
    qsim.transforms

