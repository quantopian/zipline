[![Zipline](https://media.quantopian.com/logos/open_source/zipline-logo-03_.png){.align-center
width="212px"}](https://stefan-jansen.github.io/zipline)

------------------------------------------------------------------------

[![pypi version
status](https://img.shields.io/pypi/v/zipline.svg)](https://pypi.python.org/pypi/zipline)
[![pypi pyversion
status](https://img.shields.io/pypi/pyversions/zipline.svg)](https://pypi.python.org/pypi/zipline)
[![travis
status](https://travis-ci.org/stefan-jansen/zipline.svg?branch=master)](https://travis-ci.org/stefan-jansen/zipline)
[![appveyor
status](https://ci.appveyor.com/api/projects/status/3dg18e6227dvstw6/branch/master?svg=true)](https://ci.appveyor.com/project/stefan-jansen/zipline/branch/master)
[![Coverage
Status](https://coveralls.io/repos/stefan-jansen/zipline/badge.svg)](https://coveralls.io/r/stefan-jansen/zipline)

Zipline is a Pythonic event-driven system for backtesting, used as the
backtesting and live-trading engine by Quantopian before the company
[closed
down](https://www.bizjournals.com/boston/news/2020/11/10/quantopian-shuts-down-cofounders-head-elsewhere.html)
in late 2020. Since then, the domain that originally hosted these docs
have expired. The library is used extensively in the book [Machine
Larning for Algorithmic Trading](https://ml4trading.io) by [Stefan
Jansen](https://www.linkedin.com/in/applied-ai/) who is trying to keep
the library up to date and available to his readers and the wider Python
algotrading community.

-   [Join our Community!](https://exchange.ml4trading.io)
-   [Documentation](https://zipline.ml4trading.io)

Features
========

-   **Ease of Use:** Zipline tries to get out of your way so that you
    can focus on algorithm development. See below for a code example.
-   **\"Batteries Included\":** many common statistics like moving
    average and linear regression can be readily accessed from within a
    user-written algorithm.
-   **PyData Integration:** Input of historical data and output of
    performance statistics are based on Pandas DataFrames to integrate
    nicely into the existing PyData ecosystem.
-   **Statistics and Machine Learning Libraries:** You can use libraries
    like matplotlib, scipy, statsmodels, and sklearn to support
    development, analysis, and visualization of state-of-the-art trading
    systems.

Installation
============

Zipline supports Python 3.7, 3.8, and 3.9, and may be
installed via either pip or conda.

**Note:** Installing Zipline is slightly more involved than the average
Python package. See the full [Zipline Install Documentation](https://zipline.ml4trading.io) for
detailed instructions.

For a development installation (used to develop Zipline itself), create
and activate a virtualenv, then run the `etc/dev-install` script.

Quickstart
==========

See our [getting started
tutorial](https://zipline.ml4trading.io/beginner-tutorial).

The following code implements a simple dual moving average algorithm.

``` {.sourceCode .python}
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
```

You can then run this algorithm using the Zipline CLI. First, you must
download some sample pricing and asset data:

``` {.sourceCode .bash}
$ zipline ingest
$ zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark
```

This will download asset pricing data data sourced from Quandl, and
stream it through the algorithm over the specified time range. Then, the
resulting performance DataFrame is saved in `dma.pickle`, which you can
load and analyze from within Python.

You can find other examples in the `zipline/examples` directory.

Questions?
==========

If you find a bug, feel free to [open an
issue](https://github.com/stefan-jansen/zipline/issues/new) and fill out
the issue template.
