.. _data-bundles:

Data Bundles
------------

A data bundle is a collection of pricing data, adjustment data, and an asset
database. Bundles allow us to preload all of the data we will need to run
backtests and store the data for future runs.

.. _bundles-command:

Discovering Available Bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zipline comes with a few bundles by default as well as the ability to register
new bundles. To see which bundles we have available, we may run the
``bundles`` command, for example:

.. code-block:: bash

   $ zipline bundles
   my-custom-bundle 2016-05-05 20:35:19.809398
   my-custom-bundle 2016-05-05 20:34:53.654082
   my-custom-bundle 2016-05-05 20:34:48.401767
   quandl <no ingestions>
   quantopian-quandl 2016-05-05 20:06:40.894956

The output here shows that there are 3 bundles available:

- ``my-custom-bundle`` (added by the user)
- ``quandl`` (provided by zipline)
- ``quantopian-quandl`` (provided by zipline)

The dates and times next to the name show the times when the data for this
bundle was ingested. We have run three different ingestions for
``my-custom-bundle``. We have never ingested any data for the ``quandl`` bundle
so it just shows ``<no ingestions>`` instead. Finally, there is only one
ingestion for ``quantopian-quandl``.

.. _ingesting-data:

Ingesting Data
~~~~~~~~~~~~~~

The first step to using a data bundle is to ingest the data. The ingestion
process will invoke some custom bundle command and then write the data to a
standard location that zipline can find. By default the location where ingested
data will be written is ``$ZIPLINE_ROOT/data/<bundle>`` where by default
``ZIPLINE_ROOT=~/.zipline``. The ingestion step may take some time as it could
involve downloading and processing a lot of data. You'll need a
`Quandl <https://docs.quandl.com/docs#section-authentication>`__ API key to ingest the default bundle. This can be run with:

.. code-block:: bash

   $ QUANDL_API_KEY=<yourkey> zipline ingest [-b <bundle>]


where ``<bundle>`` is the name of the bundle to ingest, defaulting to ``quandl``.

Old Data
~~~~~~~~

When the ``ingest`` command is used it will write the new data to a subdirectory
of ``$ZIPLINE_ROOT/data/<bundle>`` which is named with the current date. This
makes it possible to look at older data or even run backtests with the older
copies. Running a backtest with an old ingestion makes it easier to reproduce
backtest results later.

One drawback of saving all of the data by default is that the data directory
may grow quite large even if you do not want to use the data. As shown earlier,
we can list all of the ingestions with the :ref:`bundles command
<bundles-command>`. To solve the problem of leaking old data there is another
command: ``clean``, which will clear data bundles based on some time
constraints.

For example:

.. code-block:: bash

   # clean everything older than <date>
   $ zipline clean [-b <bundle>] --before <date>

   # clean everything newer than <date>
   $ zipline clean [-b <bundle>] --after <date>

   # keep everything in the range of [before, after] and delete the rest
   $ zipline clean [-b <bundle>] --before <date> --after <after>

   # clean all but the last <int> runs
   $ zipline clean [-b <bundle>] --keep-last <int>


Running Backtests with Data Bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that the data has been ingested we can use it to run backtests with the
``run`` command. The bundle to use can be specified with the ``--bundle`` option
like:

.. code-block:: bash

   $ zipline run --bundle <bundle> --algofile algo.py ...


We may also specify the date to use to look up the bundle data with the
``--bundle-timestamp`` option. Setting the ``--bundle-timestamp`` will cause
``run`` to use the most recent bundle ingestion that is less than or equal to
the ``bundle-timestamp``. This is how we can run backtests with older data.
``bundle-timestamp`` uses a less-than-or-equal-to relationship so that we can
specify the date that we ran an old backtest and get the same data that would
have been available to us on that date. The ``bundle-timestamp`` defaults to
the current day to use the most recent data.

Default Data Bundles
~~~~~~~~~~~~~~~~~~~~

.. _quandl-data-bundle:

Quandl WIKI Bundle
``````````````````

By default zipline comes with the ``quandl`` data bundle which uses quandl's
`WIKI dataset <https://www.quandl.com/data/WIKI>`_. The quandl data bundle
includes daily pricing data, splits, cash dividends, and asset metadata. To
ingest the ``quandl`` data bundle we recommend creating an account on quandl.com
to get an API key to be able to make more API requests per day. Once we have an
API key we may run:

.. code-block:: bash

   $ QUANDL_API_KEY=<api-key> zipline ingest -b quandl

though we may still run ``ingest`` as an anonymous quandl user (with no API
key). We may also set the ``QUANDL_DOWNLOAD_ATTEMPTS`` environment variable to
an integer which is the number of attempts that should be made to download data
from quandls servers. By default ``QUANDL_DOWNLOAD_ATTEMPTS`` will be 5, meaning
that we will retry each attempt 5 times.

.. note::

   ``QUANDL_DOWNLOAD_ATTEMPTS`` is not the total number of allowed failures,
   just the number of allowed failures per request. The quandl loader will make
   one request per 100 equities for the metadata followed by one request per
   equity.


Writing a New Bundle
~~~~~~~~~~~~~~~~~~~~

Data bundles exist to make it easy to use different data sources with
zipline. To add a new bundle, one must implement an ``ingest`` function.

The ``ingest`` function is responsible for loading the data into memory and
passing it to a set of writer objects provided by zipline to convert the data to
zipline's internal format. The ingest function may work by downloading data from
a remote location like the ``quandl`` bundle or it may just
load files that are already on the machine. The function is provided with
writers that will write the data to the correct location transactionally. If an
ingestion fails part way through the bundle will not be written in an incomplete
state.

The signature of the ingest function should be:

.. code-block:: python

   ingest(environ,
          asset_db_writer,
          minute_bar_writer,
          daily_bar_writer,
          adjustment_writer,
          calendar,
          start_session,
          end_session,
          cache,
          show_progress,
          output_dir)

``environ``
```````````

``environ`` is a mapping representing the environment variables to use. This is
where any custom arguments needed for the ingestion should be passed, for
example: the ``quandl`` bundle uses the enviornment to pass the API key and the
download retry attempt count.

``asset_db_writer``
```````````````````

``asset_db_writer`` is an instance of :class:`~zipline.assets.AssetDBWriter`.
This is the writer for the asset metadata which provides the asset lifetimes and
the symbol to asset id (sid) mapping. This may also contain the asset name,
exchange and a few other columns. To write data, invoke
:meth:`~zipline.assets.AssetDBWriter.write` with dataframes for the various
pieces of metadata. More information about the format of the data exists in the
docs for write.

``minute_bar_writer``
`````````````````````

``minute_bar_writer`` is an instance of
:class:`~zipline.data.minute_bars.BcolzMinuteBarWriter`. This writer is used to
convert data to zipline's internal bcolz format to later be read by a
:class:`~zipline.data.minute_bars.BcolzMinuteBarReader`. If minute data is
provided, users should call
:meth:`~zipline.data.minute_bars.BcolzMinuteBarWriter.write` with an iterable of
(sid, dataframe) tuples. The ``show_progress`` argument should also be forwarded
to this method. If the data source does not provide minute level data, then
there is no need to call the write method. It is also acceptable to pass an
empty iterator to :meth:`~zipline.data.minute_bars.BcolzMinuteBarWriter.write`
to signal that there is no minutely data.

.. note::

   The data passed to
   :meth:`~zipline.data.minute_bars.BcolzMinuteBarWriter.write` may be a lazy
   iterator or generator to avoid loading all of the minute data into memory at
   a single time. A given sid may also appear multiple times in the data as long
   as the dates are strictly increasing.

``daily_bar_writer``
````````````````````

``daily_bar_writer`` is an instance of
:class:`~zipline.data.us_equity_pricing.BcolzDailyBarWriter`. This writer is
used to convert data into zipline's internal bcolz format to later be read by a
:class:`~zipline.data.us_equity_pricing.BcolzDailyBarReader`. If daily data is
provided, users should call
:meth:`~zipline.data.minute_bars.BcolzDailyBarWriter.write` with an iterable of
(sid dataframe) tuples. The ``show_progress`` argument should also be forwarded
to this method. If the data shource does not provide daily data, then there is
no need to call the write method. It is also acceptable to pass an empty
iterable to :meth:`~zipline.data.minute_bars.BcolzMinuteBarWriter.write` to
signal that there is no daily data. If no daily data is provided but minute data
is provided, a daily rollup will happen to service daily history requests.

.. note::

   Like the ``minute_bar_writer``, the data passed to
   :meth:`~zipline.data.minute_bars.BcolzMinuteBarWriter.write` may be a lazy
   iterable or generator to avoid loading all of the data into memory at once.
   Unlike the ``minute_bar_writer``, a sid may only appear once in the data
   iterable.

``adjustment_writer``
`````````````````````

``adjustment_writer`` is an instance of
:class:`~zipline.data.us_equity_pricing.SQLiteAdjustmentWriter`. This writer is
used to store splits, mergers, dividends, and stock dividends. The data should
be provided as dataframes and passed to
:meth:`~zipline.data.us_equity_pricing.SQLiteAdjustmentWriter.write`. Each of
these fields are optional, but the writer can accept as much of the data as you
have.

``calendar``
````````````

``calendar`` is an instance of
:class:`zipline.utils.calendars.TradingCalendar`. The calendar is provided to
help some bundles generate queries for the days needed.

``start_session``
`````````````````

``start_session`` is a :class:`pandas.Timestamp` object indicating the first
day that the bundle should load data for.

``end_session``
```````````````

``end_session`` is a :class:`pandas.Timestamp` object indicating the last day
that the bundle should load data for.

``cache``
`````````

``cache`` is an instance of :class:`~zipline.utils.cache.dataframe_cache`. This
object is a mapping from strings to dataframes. This object is provided in case
an ingestion crashes part way through. The idea is that the ingest function
should check the cache for raw data, if it doesn't exist in the cache, it should
acquire it and then store it in the cache. Then it can parse and write the
data. The cache will be cleared only after a successful load, this prevents the
ingest function from needing to redownload all the data if there is some bug in
the parsing. If it is very fast to get the data, for example if it is coming
from another local file, then there is no need to use this cache.

``show_progress``
`````````````````

``show_progress`` is a boolean indicating that the user would like to receive
feedback about the ingest function's progress fetching and writing the
data. Some examples for where to show how many files you have downloaded out of
the total needed, or how far into some data conversion the ingest function
is. One tool that may help with implementing ``show_progress`` for a loop is
:class:`~zipline.utils.cli.maybe_show_progress`. This argument should always be
forwarded to ``minute_bar_writer.write`` and ``daily_bar_writer.write``.


``output_dir``
``````````````

``output_dir`` is a string representing the file path where all the data will be
written. ``output_dir`` will be some subdirectory of ``$ZIPLINE_ROOT`` and will
contain the time of the start of the current ingestion. This can be used to
directly move resources here if for some reason your ingest function can produce
it's own outputs without the writers. For example, the ``quantopian:quandl``
bundle uses this to directly untar the bundle into the ``output_dir``.

Ingesting Data from .csv Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zipline provides a bundle called ``csvdir``, which allows users to ingest data
from ``.csv`` files. The format of the files should be in OHLCV format, with dates,
dividends, and splits. A sample is provided below. There are other samples for testing
purposes in ``zipline/tests/resources/csvdir_samples``.

.. code-block:: text

	 date,open,high,low,close,volume,dividend,split
	 2012-01-03,58.485714,58.92857,58.42857,58.747143,75555200,0.0,1.0
	 2012-01-04,58.57143,59.240002,58.468571,59.062859,65005500,0.0,1.0
	 2012-01-05,59.278572,59.792858,58.952858,59.718571,67817400,0.0,1.0
	 2012-01-06,59.967144,60.392857,59.888573,60.342857,79573200,0.0,1.0
	 2012-01-09,60.785713,61.107143,60.192856,60.247143,98506100,0.0,1.0
	 2012-01-10,60.844284,60.857143,60.214287,60.462856,64549100,0.0,1.0
	 2012-01-11,60.382858,60.407143,59.901428,60.364285,53771200,0.0,1.0

Once you have your data in the correct format, you can edit your ``extension.py`` file in
``~/.zipline/extension.py`` and import the csvdir bundle, along with ``pandas``.

.. code-block:: python

	 import pandas as pd
	 
	 from zipline.data.bundles import register
	 from zipline.data.bundles.csvdir import csvdir_equities

We'll then want to specify the start and end sessions of our bundle data:

.. code-block:: python

	 start_session = pd.Timestamp('2016-1-1', tz='utc')
	 end_session = pd.Timestamp('2018-1-1', tz='utc')

And then we can ``register()`` our bundle, and pass the location of the directory in which
our ``.csv`` files exist:

.. code-block:: python

    register(
        'custom-csvdir-bundle',
        csvdir_equities(
            ['daily'],
            '/path/to/your/csvs',
        ),
        calendar_name='NYSE', # US equities
        start_session=start_session,
        end_session=end_session
    )

To finally ingest our data, we can run:

.. code-block:: bash

	 $ zipline ingest -b custom-csvdir-bundle
	 Loading custom pricing data:   [############------------------------]   33% | FAKE: sid 0
	 Loading custom pricing data:   [########################------------]   66% | FAKE1: sid 1
	 Loading custom pricing data:   [####################################]  100% | FAKE2: sid 2
	 Loading custom pricing data:   [####################################]  100%
	 Merging daily equity files:  [####################################]  
	 
	 # optionally, we can pass the location of our csvs via the command line
	 $ CSVDIR=/path/to/your/csvs zipline ingest -b custom-csvdir-bundle


If you would like to use equities that are not in the NYSE calendar, or the existing zipline calendars,
you can look at the ``Trading Calendar Tutorial`` to build a custom trading calendar that you can then pass
the name of to ``register()``.
