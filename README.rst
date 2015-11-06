Zipline
=======

|Gitter|
|version status|
|downloads|
|build status|
|Coverage Status|
|Code quality|

Zipline is a Pythonic algorithmic trading library. It is an event-driven
system that supports both backtesting and live-trading.

Zipline is currently used in production as the backtesting and live-trading
engine powering `Quantopian <https://www.quantopian.com>`__ -- a free,
community-centered, hosted platform for building and executing trading
strategies.

`Join our
community! <https://groups.google.com/forum/#!forum/zipline>`__

Want to contribute? See our `open
requests <https://github.com/quantopian/zipline/wiki/Contribution-Requests>`__
and our `general
guidelines <https://github.com/quantopian/zipline#contributions>`__
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

pip
---

You can install Zipline via the ``pip`` command:
::

    $ pip install zipline


conda
-----

Another way to install Zipline is via ``conda`` which comes as part
of `Anaconda <http://continuum.io/downloads>`__ or can be installed via
``pip install conda``.

Once set up, you can install Zipline from our ``Quantopian`` channel:

::

    conda install -c Quantopian zipline

Currently supported platforms include:

-  GNU/Linux 64-bit
-  OSX 64-bit

.. note::

   Windows may work; however, it is currently untested.

Dependencies
------------

See our `requirements file
<https://github.com/quantopian/zipline/blob/master/etc/requirements.txt>`__

Quickstart
==========

See our `getting started
tutorial <http://www.zipline.io/#quickstart>`__.

The following code implements a simple dual moving average algorithm.

.. code:: python

    from zipline.api import (
        add_history,
        history,
        order_target,
        record,
        symbol,
    )


    def initialize(context):
        # Register 2 histories that track daily prices,
        # one with a 100 window and one with a 300 day window
        add_history(100, '1d', 'price')
        add_history(300, '1d', 'price')
        context.i = 0


    def handle_data(context, data):
        # Skip first 300 days to get full windows
        context.i += 1
        if context.i < 300:
            return

        # Compute averages
        # history() has to be called with the same params
        # from above and returns a pandas dataframe.
        short_mavg = history(100, '1d', 'price').mean()
        long_mavg = history(300, '1d', 'price').mean()

        sym = symbol('AAPL')

        # Trading logic
        if short_mavg[sym] > long_mavg[sym]:
            # order_target orders as many shares as needed to
            # achieve the desired number of shares.
            order_target(sym, 100)
        elif short_mavg[sym] < long_mavg[sym]:
            order_target(sym, 0)

        # Save values for later inspection
        record(AAPL=data[sym].price,
               short_mavg=short_mavg[sym],
               long_mavg=long_mavg[sym])

You can then run this algorithm using the Zipline CLI. From the command
line, run:

.. code:: bash

    python run_algo.py -f dual_moving_average.py --symbols AAPL --start 2011-1-1 --end 2012-1-1 -o dma.pickle

This will download the AAPL price data from Yahoo! Finance in the
specified time range and stream it through the algorithm and save the
resulting performance dataframe to dma.pickle which you can then load
and analyze from within python.

You can find other examples in the zipline/examples directory.

Contributions
=============

If you would like to contribute, please see our Contribution Requests:
https://github.com/quantopian/zipline/wiki/Contribution-Requests

.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/quantopian/zipline?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |version status| image:: https://img.shields.io/pypi/pyversions/zipline.svg
   :target: https://pypi.python.org/pypi/zipline
.. |downloads| image:: https://img.shields.io/pypi/dd/zipline.svg
   :target: https://pypi.python.org/pypi/zipline
.. |build status| image:: https://travis-ci.org/quantopian/zipline.png?branch=master
   :target: https://travis-ci.org/quantopian/zipline
.. |Coverage Status| image:: https://coveralls.io/repos/quantopian/zipline/badge.png
   :target: https://coveralls.io/r/quantopian/zipline
.. |Code quality| image:: https://scrutinizer-ci.com/g/quantopian/zipline/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/quantopian/zipline/
