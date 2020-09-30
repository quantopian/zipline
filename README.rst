.. image:: https://media.quantopian.com/logos/open_source/zipline-logo-03_.png
    :target: https://www.zipline.io
    :width: 212px
    :align: center
    :alt: Zipline

=============

|Gitter|
|pypi version status|
|pypi pyversion status|
|travis status|
|appveyor status|
|Coverage Status|

Zipline is a Pythonic algorithmic trading library. It is an event-driven
system for backtesting. Zipline is currently used in production as the backtesting and live-trading
engine powering `Quantopian <https://www.quantopian.com>`_ -- a free,
community-centered, hosted platform for building and executing trading
strategies. Quantopian also offers a `fully managed service for professionals <https://factset.quantopian.com>`_
that includes Zipline, Alphalens, Pyfolio, FactSet data, and more.

- `Join our Community! <https://groups.google.com/forum/#!forum/zipline>`_
- `Documentation <https://www.zipline.io>`_
- Want to Contribute? See our `Development Guidelines <https://www.zipline.io/development-guidelines>`_

Features
========

- **Ease of Use:** Zipline tries to get out of your way so that you can
  focus on algorithm development. See below for a code example.
- **"Batteries Included":** many common statistics like
  moving average and linear regression can be readily accessed from
  within a user-written algorithm.
- **PyData Integration:** Input of historical data and output of performance statistics are
  based on Pandas DataFrames to integrate nicely into the existing
  PyData ecosystem.
- **Statistics and Machine Learning Libraries:** You can use libraries like matplotlib, scipy,
  statsmodels, and sklearn to support development, analysis, and
  visualization of state-of-the-art trading systems.

Installation
============

Zipline currently supports Python 2.7, 3.5, and 3.6, and may be installed via
either pip or conda.

**Note:** Installing Zipline is slightly more involved than the average Python
package. See the full `Zipline Install Documentation`_ for detailed
instructions.

For a development installation (used to develop Zipline itself), create and
activate a virtualenv, then run the ``etc/dev-install`` script.

Quickstart
==========

See our `getting started tutorial <https://www.zipline.io/beginner-tutorial>`_.

The following code implements a simple dual moving average algorithm.

.. code:: python

    from zipline.api import order_target, record, symbol

    def initialize(context):
        context.i = 0
        context.asset = symbol('AAPL')


    def handle_data(context, data):
        # Skip first 300 days to get full windows
        context.i += 1
        if context.i < 300:
            return

        # Compute averages
        # data.history() has to be called with the same params
        # from above and returns a pandas dataframe.
        short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
        long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()

        # Trading logic
        if short_mavg > long_mavg:
            # order_target orders as many shares as needed to
            # achieve the desired number of shares.
            order_target(context.asset, 100)
        elif short_mavg < long_mavg:
            order_target(context.asset, 0)

        # Save values for later inspection
        record(AAPL=data.current(context.asset, 'price'),
               short_mavg=short_mavg,
               long_mavg=long_mavg)


You can then run this algorithm using the Zipline CLI.
First, you must download some sample pricing and asset data:

.. code:: bash

    $ zipline ingest
    $ zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark

This will download asset pricing data data sourced from Quandl, and stream it through the algorithm over the specified time range.
Then, the resulting performance DataFrame is saved in ``dma.pickle``, which you can load and analyze from within Python.

You can find other examples in the ``zipline/examples`` directory.

Questions?
==========

If you find a bug, feel free to `open an issue <https://github.com/quantopian/zipline/issues/new>`_ and fill out the issue template.

Contributing
============

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. Details on how to set up a development environment can be found in our `development guidelines <https://www.zipline.io/development-guidelines>`_.

If you are looking to start working with the Zipline codebase, navigate to the GitHub `issues` tab and start looking through interesting issues. Sometimes there are issues labeled as `Beginner Friendly <https://github.com/quantopian/zipline/issues?q=is%3Aissue+is%3Aopen+label%3A%22Beginner+Friendly%22>`_ or `Help Wanted <https://github.com/quantopian/zipline/issues?q=is%3Aissue+is%3Aopen+label%3A%22Help+Wanted%22>`_.

Feel free to ask questions on the `mailing list <https://groups.google.com/forum/#!forum/zipline>`_ or on `Gitter <https://gitter.im/quantopian/zipline>`_.

.. note::

   Please note that Zipline is not a community-led project. Zipline is
   maintained by the Quantopian engineering team, and we are quite small and
   often busy.

   Because of this, we want to warn you that we may not attend to your pull
   request, issue, or direct mention in months, or even years. We hope you
   understand, and we hope that this note might help reduce any frustration or
   wasted time.


.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/quantopian/zipline?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |pypi version status| image:: https://img.shields.io/pypi/v/zipline.svg
   :target: https://pypi.python.org/pypi/zipline
.. |pypi pyversion status| image:: https://img.shields.io/pypi/pyversions/zipline.svg
   :target: https://pypi.python.org/pypi/zipline
.. |travis status| image:: https://travis-ci.org/quantopian/zipline.svg?branch=master
   :target: https://travis-ci.org/quantopian/zipline
.. |appveyor status| image:: https://ci.appveyor.com/api/projects/status/3dg18e6227dvstw6/branch/master?svg=true
   :target: https://ci.appveyor.com/project/quantopian/zipline/branch/master
.. |Coverage Status| image:: https://coveralls.io/repos/quantopian/zipline/badge.svg
   :target: https://coveralls.io/r/quantopian/zipline

.. _`Zipline Install Documentation` : https://www.zipline.io/install
