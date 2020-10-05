.. _metrics:

Risk and Performance Metrics
----------------------------

The risk and performance metrics are summarizing values calculated by Zipline
when running a simulation. These metrics can be about the performance of an
algorithm, like returns or cash flow, or the riskiness of an algorithm, like
volatility or beta. Metrics may be reported minutely, daily, or once at the end
of a simulation. A single metric may choose to report at multiple time-scales
where appropriate.

Metrics Sets
~~~~~~~~~~~~

Zipline groups risk and performance metrics into collections called "metrics
sets". A single metrics set defines all of the metrics to track during a single
backtest. A metrics set may contain metrics that report at different time
scales. The default metrics set will compute a host of metrics, such as
algorithm returns, volatility, Sharpe ratio, and beta.

Selecting the Metrics Set
~~~~~~~~~~~~~~~~~~~~~~~~~

When running a simulation, the user may select the metrics set to report. How
you select the metrics set depends on the interface being used to run the
algorithm.

Command Line and IPython Magic
``````````````````````````````

When running with the command line or IPython magic interfaces, the metrics set
may be selected by passing the ``--metrics-set`` argument. For example:

.. code-block:: bash

   $ zipline run algorithm.py -s 2014-01-01 -e 2014-02-01 --metrics-set my-metrics-set

``run_algorithm``
`````````````````

When running through the :func:`~zipline.run_algorithm` interface, the metrics
set may be passed with the ``metrics_set`` argument. This may either be the name
of a registered metrics set, or a set of metric object. For example:

.. code-block:: python

   run_algorithm(..., metrics_set='my-metrics-set')
   run_algorithm(..., metrics_set={MyMetric(), MyOtherMetric(), ...})

Running Without Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Computing risk and performance metrics is not free, and contributes to the total
runtime of a backtest. When actively developing an algorithm, it is often
helpful to skip these computations to speed up the debugging cycle. To disable
the calculation and reporting of all metrics, users may select the built-in
metrics set ``none``. For example:

.. code-block:: bash

   $ zipline run algorithm.py -s 2014-01-01 -e 2014-02-01 --metrics-set none

Defining New Metrics
~~~~~~~~~~~~~~~~~~~~

A metric is any object that implements some subset of the following methods:

- ``start_of_simulation``
- ``end_of_simulation``
- ``start_of_session``
- ``end_of_session``
- ``end_of_bar``

These functions will be called at the time indicated by their name, at which
point the metric object may collect any needed information and optionally report
a computed value. If a metric does not need to do any processing at one of these
times, it may omit a definition for the given method.

A metric should be reusable, meaning that a single instance of a metric class
should be able to be used across multiple backtests. Metrics do not need to
support multiple simulations at once, meaning that internal caches and data are
consistent between ``start_of_simulation`` and ``end_of_simulation``.

``start_of_simulation``
```````````````````````

The ``start_of_simulation`` method should be thought of as a per-simulation
constructor. This method should initialize any caches needed for the duration of
a single simulation.

The ``start_of_simulation`` method should have the following signature:

.. code-block:: python

   def start_of_simulation(self,
                           ledger,
                           emission_rate,
                           trading_calendar,
                           sessions,
                           benchmark_source):
       ...

``ledger`` is an instance of :class:`~zipline.finance.ledger.Ledger` which is
maintaining the simulation's state. This may be used to lookup the algorithm's
starting portfolio values.

``emission_rate`` is a string representing the smallest frequency at which
metrics should be reported. ``emission_rate`` will be either ``minute`` or
``daily``. When ``emission_rate`` is ``daily``, ``end_of_bar`` will not be
called at all.

``trading_calendar`` is an instance of
:class:`~zipline.utils.calendars.TradingCalendar` which is the trading calendar
being used by the simulation.

``sessions`` is a :class:`pandas.DatetimeIndex` which is holds the session
labels, in sorted order, that the simulation will execute.

``benchmark_source`` is an instance of
:class:`~zipline.sources.benchmark_source.BenchmarkSource` which is the
interface to the returns of the benchmark specified by
:func:`~zipline.api.set_benchmark`.

``end_of_simulation``
`````````````````````

The ``end_of_simulation`` method should have the following signature:

.. code-block:: python

   def end_of_simulation(self,
                         packet,
                         ledger,
                         trading_calendar,
                         sessions,
                         data_portal,
                         benchmark_source):
       ...

``ledger`` is an instance of :class:`~zipline.finance.ledger.Ledger` which is
maintaining the simulation's state. This may be used to lookup the algorithm's
final portfolio values.

``packet`` is a dictionary to write the end of simulation values for the given
metric into.

``trading_calendar`` is an instance of
:class:`~zipline.utils.calendars.TradingCalendar` which is the trading calendar
being used by the simulation.

``sessions`` is a :class:`pandas.DatetimeIndex` which is holds the session
labels, in sorted order, that the simulation has executed.

``data_portal`` is an instance of :class:`~zipline.data.data_portal.DataPortal`
which is the metric's interface to pricing data.

``benchmark_source`` is an instance of
:class:`~zipline.sources.benchmark_source.BenchmarkSource` which is the
interface to the returns of the benchmark specified by
:func:`~zipline.api.set_benchmark`.

``start_of_session``
````````````````````

The ``start_of_session`` method may see a slightly different view of the
``ledger`` or ``data_portal`` than the previous ``end_of_session`` if the price
of any futures owned move between trading sessions or if a capital change
occurs.

The ``start_of_session`` method should have the following signature:

.. code-block:: python

   def start_of_session(self,
                        ledger,
                        session_label,
                        data_portal):
       ...

``ledger`` is an instance of :class:`~zipline.finance.ledger.Ledger` which is
maintaining the simulation's state. This may be used to lookup the algorithm's
current portfolio values.

``session_label`` is a :class:`~pandas.Timestamp` which is the label of the
session which is about to run.

``data_portal`` is an instance of :class:`~zipline.data.data_portal.DataPortal`
which is the metric's interface to pricing data.

``end_of_session``
``````````````````

The ``end_of_session`` method should have the following signature:

.. code-block:: python

   def end_of_session(self,
                      packet,
                      ledger,
                      session_label,
                      session_ix,
                      data_portal):

``packet`` is a dictionary to write the end of session values. This dictionary
contains two sub-dictionaries: ``daily_perf`` and ``cumulative_perf``. When
applicable, the ``daily_perf`` should hold the current day's value, and
``cumulative_perf`` should hold a cumulative value for the entire simulation up
to the current time.

``ledger`` is an instance of :class:`~zipline.finance.ledger.Ledger` which is
maintaining the simulation's state. This may be used to lookup the algorithm's
current portfolio values.

``session_label`` is a :class:`~pandas.Timestamp` which is the label of the
session which is has just completed.

``session_ix`` is an :class:`int` which is the index of the current trading
session being run. This is provided to allow for efficient access to the daily
returns through ``ledger.daily_returns_array[:session_ix + 1]``.

``data_portal`` is an instance of :class:`~zipline.data.data_portal.DataPortal`
which is the metric's interface to pricing data

``end_of_bar``
``````````````

.. note::

   ``end_of_bar`` is only called when ``emission_mode`` is ``minute``.

The ``end_of_bar`` method should have the following signature:

.. code-block:: python

   def end_of_bar(self,
                  packet,
                  ledger,
                  dt,
                  session_ix,
                  data_portal):

``packet`` is a dictionary to write the end of session values. This dictionary
contains two sub-dictionaries: ``minute_perf`` and ``cumulative_perf``. When
applicable, the ``minute_perf`` should hold the current partial day's value, and
``cumulative_perf`` should hold a cumulative value for the entire simulation up
to the current time.

``ledger`` is an instance of :class:`~zipline.finance.ledger.Ledger` which is
maintaining the simulation's state. This may be used to lookup the algorithm's
current portfolio values.

``dt`` is a :class:`~pandas.Timestamp` which is the label of bar that has just
completed.

``session_ix`` is an :class:`int` which is the index of the current trading
session being run. This is provided to allow for efficient access to the daily
returns through ``ledger.daily_returns_array[:session_ix + 1]``.

``data_portal`` is an instance of :class:`~zipline.data.data_portal.DataPortal`
which is the metric's interface to pricing data.

Defining New Metrics Sets
~~~~~~~~~~~~~~~~~~~~~~~~~

Users may use :func:`zipline.finance.metrics.register` to register a new metrics
set. This may be used to decorate a function taking no arguments which returns a
new set of metric object instances. For example:

.. code-block:: python

   from zipline.finance import metrics

   @metrics.register('my-metrics-set')
   def my_metrics_set():
       return {MyMetric(), MyOtherMetric(), ...}


This may be embedded in the user's ``extension.py``.

The reason that a metrics set is defined as a function which produces a set,
instead of just a set, is that users may want to fetch external data or
resources to construct their metrics. By putting this behind a callable, users
do not need to fetch the resources when the metrics set is not being used.
