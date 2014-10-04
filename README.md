Zipline
=======
[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/quantopian/zipline?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![version status](https://pypip.in/v/zipline/badge.png)](https://pypi.python.org/pypi/zipline)
[![downloads](https://pypip.in/d/zipline/badge.png)](https://pypi.python.org/pypi/zipline)
[![build status](https://travis-ci.org/quantopian/zipline.png?branch=master)](https://travis-ci.org/quantopian/zipline)
[![Coverage Status](https://coveralls.io/repos/quantopian/zipline/badge.png)](https://coveralls.io/r/quantopian/zipline)

Zipline is a Pythonic algorithmic trading library.  The system is
fundamentally event-driven and a close approximation of how
live-trading systems operate.  Currently, backtesting is well
supported, but the intent is to develop the library for both paper and
live trading, so that the same logic used for backtesting can be
applied to the market.

Zipline is currently used in production as the backtesting engine
powering Quantopian (https://www.quantopian.com) -- a free,
community-centered platform that allows development and real-time
backtesting of trading algorithms in the web browser.

Want to contribute? See our [open requests](https://github.com/quantopian/zipline/wiki/Contribution-Requests)
and our [general guidelines](https://github.com/quantopian/zipline#contributions) below.

Discussion and Help
===================

Discussion of the project is held at the Google Group,
<zipline@googlegroups.com>,
<https://groups.google.com/forum/#!forum/zipline>.

Features
========

* Ease of use: Zipline tries to get out of your way so that you can
focus on algorithm development. See below for a code example.

* Zipline comes "batteries included" as many common statistics like
moving average and linear regression can be readily accessed from
within a user-written algorithm.

* Input of historical data and output of performance statistics is
based on Pandas DataFrames to integrate nicely into the existing
Python eco-system.

* Statistic and machine learning libraries like matplotlib, scipy,
statsmodels, and sklearn support development, analysis and
visualization of state-of-the-art trading systems.

Installation
============

The easiest way to install Zipline is via `conda` which comes as part of [Anaconda](http://continuum.io/downloads) or can be installed via `pip install conda`.

Once set up, you can install Zipline from our Quantopian channel:

```
conda install -c Quantopian zipline
```

Currently supported platforms include:
* Windows 32-bit (can be 64-bit Windows but has to be 32-bit Anaconda)
* OSX 64-bit
* Linux 64-bit

PIP
---

Alternatively you can install Zipline via the more traditional `pip`
command. Since zipline is pure-python code it should be very easy to
install and set up:

```
pip install numpy   # Pre-install numpy to handle dependency chain quirk
pip install zipline
```

If there are problems installing the dependencies or zipline we
recommend installing these packages via some other means. For Windows,
the [Enthought Python Distribution](http://www.enthought.com/products/epd.php)
includes most of the necessary dependencies. On OSX, the
[Scipy Superpack](http://fonnesbeck.github.com/ScipySuperpack/)
works very well.

Dependencies
------------

* Python (2.7 or 3.3)
* numpy (>= 1.6.0)
* pandas (>= 0.9.0)
* pytz
* Logbook
* requests
* [python-dateutil](https://pypi.python.org/pypi/python-dateutil) (>= 2.1)
* ta-lib


Quickstart
==========

See our [tutorial](http://nbviewer.ipython.org/github/quantopian/zipline/blob/master/docs/tutorial.ipynb) to get started.

The following code implements a simple dual moving average algorithm.

```python
from zipline.api import order_target, record, symbol, history, add_history


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
```

You can then run this algorithm using the Zipline CLI. From the
command line, run:

```bash
python run_algo.py -f dual_moving_avg.py --symbols AAPL --start 2011-1-1 --end 2012-1-1 -o dma.pickle
```

This will download the AAPL price data from Yahoo! Finance in the
specified time range and stream it through the algorithm and save the
resulting performance dataframe to dma.pickle which you can then load
and analyze from within python.

You can find other examples in the zipline/examples directory.

Contributions
============

If you would like to contribute, please see our Contribution Requests: https://github.com/quantopian/zipline/wiki/Contribution-Requests

Credits
--------
Thank you for all the help so far!

- @rday for sortino ratio, information ratio, and exponential moving average transform
- @snth
- @yinhm for integrating zipline with @yinhm/datafeed
- [Jeremiah Lowin](http://www.lowindata.com) for teaching us the nuances of Sharpe and Sortino Ratios,
  and for implementing new order methods.
- Brian Cappello
- @verdverm (Tony Worm), Order types (stop, limit)
- @benmccann for benchmarking contributions
- @jkp and @bencpeters for bugfixes to benchmark.
- @dstephens for adding Canadian treasury curves.
- @mtrovo for adding BMF&Bovespa calendars.
- @sdrdis for bugfixes.
- @humdings for refactoring the order methods.
- Quantopian Team

(alert us if we've inadvertantly missed listing you here!)

Development Environment
-----------------------

The following guide assumes your system has [virtualenvwrapper](https://bitbucket.org/dhellmann/virtualenvwrapper)
and [pip](http://www.pip-installer.org/en/latest/) already installed.

You'll need to install some C library dependencies:

```
sudo apt-get install libopenblas-dev liblapack-dev gfortran

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

Suggested installation of Python library dependencies used for development:

```
mkvirtualenv zipline
./etc/ordered_pip.sh ./etc/requirements.txt
pip install -r ./etc/requirements_dev.txt
```

Finally, install zipline in develop mode (from the zipline source root dir):

```
python setup.py develop
```

Style Guide
------------

To ensure that changes and patches are focused on behavior changes,
the zipline codebase adheres to both PEP-8,
<http://www.python.org/dev/peps/pep-0008/>, and pyflakes,
<https://launchpad.net/pyflakes/>.

The maintainers check the code using the flake8 script,
<https://bitbucket.org/tarek/flake8/wiki/Home>, which is included in the
requirements_dev.txt.

Before submitting patches or pull requests, please ensure that your
changes pass ```flake8 zipline tests``` and ```nosetests```

Source
======

The source for Zipline is hosted at
<https://github.com/quantopian/zipline>.

Documentation
------------

You can compile the documentation using Sphinx:

```
sudo apt-get install python-sphinx
make html
```

Contact
=======

For other questions, please contact <opensource@quantopian.com>.
