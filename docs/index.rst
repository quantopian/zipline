.. Zipline documentation master file, created by
   sphinx-quickstart on Wed Feb  8 15:29:56 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: zipline

****************************************************
Zipline: Financial Backtester for Trading Algorithms
****************************************************

Python is quickly becoming the glue language which holds together data science
and related fields like quantitative finance. Zipline is a new, BSD-licensed
quantitative trading system which allows easy backtesting of investment
algorithms on historical data. The system is fundamentally event-driven and a
close approximation of how live-trading systems operate. Moreover, Zipline
comes "batteries included" as many common statistics like
moving average and linear regression can be readily accessed from within a
user-written algorithm. Input of historical data and output of performance
statistics is based on Pandas DataFrames to integrate nicely into the existing
Python eco-system. Furthermore, statistic and machine learning libraries like
matplotlib, scipy, statsmodels, and sklearn support development, analysis and
visualization of state-of-the-art trading systems.

Zipline is currently used in production as the backtesting engine
powering `quantopian.com <https://app.quantopian.com>`_ -- a free, community-centered
platform that allows development and real-time backtesting of trading
algorithms in the web browser.

Features
========

* Ease of use: Zipline tries to get out of your way so that you can focus on
  algorithm development. See below for a code example.

* Zipline comes "batteries included" as many common statistics like moving
  average and linear regression can be readily accessed from within a
  user-written algorithm.

* Input of historical data and output of performance statistics is based on
  Pandas DataFrames to integrate nicely into the existing Python eco-system.

* Statistic and machine learning libraries like matplotlib, scipy, statsmodels,
  and sklearn support development, analysis and visualization of
  state-of-the-art trading systems.

Contents
========

.. toctree::
   :maxdepth: 4

   manifesto.rst
   installation.rst
   quickstart.rst
   contributing.rst
   overview.rst
   modules.rst
   extensions.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
