Zipline
=======

Zipline is a Pythonic algorithmic trading library.
The system is fundamentally event-driven and a close
approximation of how live-trading systems operate.
Currently, backtesting is well supported, but the intent is
to develop the library for both paper and live trading,
so that the same logic used for backtesting can be applied
to the market.

Zipline is currently used in production as the backtesting engine
powering Quantopian (https://www.quantopian.com) -- a free, community-centered
platform that allows development and real-time backtesting of trading
algorithms in the web browser.

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

Since zipline is pure-python code it should be very easy to install
and set up with pip:

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

* Python (>= 2.7.2)
* numpy (>= 1.6.0)
* pandas (>= 0.9.0)
* pytz
* Logbook
* requests
* [python-dateutil](https://pypi.python.org/pypi/python-dateutil) (>= 2.1)


Conda
-----

We provide experimental support for conda packages. Thus if you installed [Anaconda](http://continuum.io/downloads)
you can try:
```
conda install -c Quantopian zipline
```

Currently this only works for linux 64 bit. If you want to help extend this,
have a look at the `conda` subdirectory.

Quickstart
==========

The following code implements a simple dual moving average algorithm
and tests it on data extracted from yahoo finance.

```python
from zipline import TradingAlgorithm
from zipline.transforms import MovingAverage
from zipline.utils.factory import load_from_yahoo

from datetime import datetime
import pytz
import matplotlib.pyplot as plt

class DualMovingAverage(TradingAlgorithm):
    """Dual Moving Average Crossover algorithm.

    This algorithm buys apple once its short moving average crosses
    its long moving average (indicating upwards momentum) and sells
    its shares once the averages cross again (indicating downwards
    momentum).

    """
    def initialize(self, short_window=100, long_window=400):
        # Add 2 mavg transforms, one with a long window, one
        # with a short window.
        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           window_length=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           window_length=long_window)

        # To keep track of whether we invested in the stock or not
        self.invested = False

    def handle_data(self, data):
        short_mavg = data['AAPL'].short_mavg['price']
        long_mavg = data['AAPL'].long_mavg['price']
        buy = False
        sell = False

	# Has short mavg crossed long mavg?
        if short_mavg > long_mavg and not self.invested:
            self.order('AAPL', 100)
            self.invested = True
            buy = True
        elif short_mavg < long_mavg and self.invested:
            self.order('AAPL', -100)
            self.invested = False
            sell = True

	# Record state variables. A column for each
	# variable will be added to the performance
	# DataFrame returned by .run()
        self.record(short_mavg=short_mavg,
                    long_mavg=long_mavg,
                    buy=buy,
                    sell=sell)

# Load data
start = datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
       		       end=end, adjusted=False)

# Run algorithm
dma = DualMovingAverage()
perf = dma.run(data)

# Plot results
fig = plt.figure()
ax1 = fig.add_subplot(211,  ylabel='Price in $')
data['AAPL'].plot(ax=ax1, color='r', lw=2.)
perf[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

ax1.plot(perf.ix[perf.buy].index, perf.short_mavg[perf.buy],
         '^', markersize=10, color='m')
ax1.plot(perf.ix[perf.sell].index, perf.short_mavg[perf.sell],
         'v', markersize=10, color='k')

ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
perf.portfolio_value.plot(ax=ax2, lw=2.)

ax2.plot(perf.ix[perf.buy].index, perf.portfolio_value[perf.buy],
         '^', markersize=10, color='m')
ax2.plot(perf.ix[perf.sell].index, perf.portfolio_value[perf.sell],
         'v', markersize=10, color='k')
```

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

Build Status
============

[![Build Status](https://travis-ci.org/quantopian/zipline.png)](https://travis-ci.org/quantopian/zipline)

Contact
=======

For other questions, please contact <opensource@quantopian.com>.
