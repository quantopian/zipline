.. image:: https://media.quantopian.com/logos/open_source/zipline-logo-03_.png
    :target: http://www.zipline.io
    :width: 212px
    :align: center
    :alt: Zipline

Zipline-live
============

|travis status|
|appveyor status|
|Coverage Status|

Zipline is a Pythonic algorithmic trading library. It is an event-driven
system that supports both backtesting and live-trading.

Zipline is currently used in production as the backtesting and live-trading
engine powering `Quantopian <https://www.quantopian.com>`_ -- a free,
community-centered, hosted platform for building and executing trading
strategies.

`Join our
community! <https://groups.google.com/forum/#!forum/zipline>`_

`Documentation <http://www.zipline.io>`_

Want to contribute? See our `open
requests <https://github.com/quantopian/zipline/wiki/Contribution-Requests>`_
and our `general
guidelines <https://github.com/quantopian/zipline#contributions>`_
below.

Features
========

- Ease of use: Zipline tries to get out of your way so that you can
  focus on algorithm development. See below for a code example.
- Zipline comes "batteries included" as many common statistics like
  moving average and linear regression can be readily accessed from
  within a user-written algorithm.
- Input of historical data and output of performance statistics are
  based on Pandas DataFrames to integrate nicely into the existing
  PyData eco-system.
- Statistic and machine learning libraries like matplotlib, scipy,
  statsmodels, and sklearn support development, analysis, and
  visualization of state-of-the-art trading systems.

Installation
============

Installing With ``pip``
-----------------------

Assuming you have all required (see note below) non-Python dependencies, you
can install Zipline with ``pip`` via:

.. code-block:: bash

    $ pip install zipline

**Note:** Installing Zipline via ``pip`` is slightly more involved than the
average Python package.  Simply running ``pip install zipline`` will likely
fail if you've never installed any scientific Python packages before.

There are two reasons for the additional complexity:

1. Zipline ships several C extensions that require access to the CPython C API.
   In order to build the C extensions, ``pip`` needs access to the CPython
   header files for your Python installation.

2. Zipline depends on `numpy <http://www.numpy.org/>`_, the core library for
   numerical array computing in Python.  Numpy depends on having the `LAPACK
   <http://www.netlib.org/lapack>`_ linear algebra routines available.

Because LAPACK and the CPython headers are binary dependencies, the correct way
to install them varies from platform to platform.  On Linux, users generally
acquire these dependencies via a package manager like ``apt``, ``yum``, or
``pacman``.  On OSX, `Homebrew <http://www.brew.sh>`_ is a popular choice
providing similar functionality.

See the full `Zipline Install Documentation`_ for more information on acquiring
binary dependencies for your specific platform.

conda
-----

Another way to install Zipline is via the ``conda`` package manager, which
comes as part of `Anaconda <http://continuum.io/downloads>`_ or can be
installed via ``pip install conda``.

Once set up, you can install Zipline from our ``Quantopian`` channel:

.. code-block:: bash

    conda install -c Quantopian zipline

Currently supported platforms include:

-  GNU/Linux 64-bit
-  OSX 64-bit
-  Windows 64-bit

.. note::

   Windows 32-bit may work; however, it is not currently included in
   continuous integration tests.

Quickstart
==========

See our `getting started
tutorial <http://www.zipline.io/#quickstart>`_.

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


You can then run this algorithm using the Zipline CLI. From the command
line, run:

.. code:: bash

    zipline ingest
	  
.. code:: bash

    zipline run -f dual_moving_average.py --start 2011-1-1 --end 2012-1-1 -o dma.pickle

This will download the AAPL price data from `quantopian-quandl` in the
specified time range and stream it through the algorithm and save the
resulting performance dataframe to dma.pickle which you can then load
and analyze from within python.

You can find other examples in the zipline/examples directory.

Contributions
=============

If you would like to contribute, please see our Contribution Requests:
https://github.com/quantopian/zipline/wiki/Contribution-Requests


.. |travis status| image:: https://travis-ci.org/zipline-live/zipline.svg?branch=live
    :target: https://travis-ci.org/zipline-live/zipline
.. |appveyor status| image:: https://ci.appveyor.com/api/projects/status/jkuo1hca8v8aueym?svg=true
   :target: https://ci.appveyor.com/project/pbharrin/zipline/branch/live
.. |Coverage Status| image:: https://coveralls.io/repos/github/zipline-live/zipline/badge.svg?branch=live
   :target: https://coveralls.io/github/zipline-live/zipline?branch=live



.. _`Zipline Install Documentation` : http://www.zipline.io/install.html
