<p align="center">
<a href="https://zipline.ml4trading.io">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

# Backtest your Trading Strategies

|Version Info| [![Python](https://img.shields.io/pypi/pyversions/zipline-reloaded.svg?cacheSeconds=2592000")](https://pypi.python.org/pypi/zipline-reloaded) [![Anaconda-Server Badge](https://anaconda.org/ml4t/zipline-reloaded/badges/platforms.svg)](https://anaconda.org/ml4t/zipline-reloaded) [![Release](https://img.shields.io/pypi/v/zipline-reloaded.svg?cacheSeconds=2592000)](https://pypi.org/project/zipline-reloaded/)  [![Anaconda-Server Badge](https://anaconda.org/ml4t/zipline-reloaded/badges/version.svg)](https://anaconda.org/ml4t/zipline-reloaded)|
|----|----|
|**Test** **Status** | [![CI Tests](https://github.com/stefan-jansen/zipline-reloaded/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/stefan-jansen/zipline-reloaded/actions/workflows/unit_tests.yml) [![PyPI](https://github.com/stefan-jansen/zipline-reloaded/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/stefan-jansen/zipline-reloaded/actions/workflows/build_wheels.yml) [![Anaconda](https://github.com/stefan-jansen/zipline-reloaded/actions/workflows/conda_package.yml/badge.svg)](https://github.com/stefan-jansen/zipline-reloaded/actions/workflows/conda_package.yml) [![Coverage Status](https://coveralls.io/repos/stefan-jansen/zipline-reloaded/badge.svg)](https://coveralls.io/r/stefan-jansen/zipline-reloaded)|
|**Community**|[![Discourse](https://img.shields.io/discourse/topics?server=https%3A%2F%2Fexchange.ml4trading.io%2F)](https://exchange.ml4trading.io) [![ML4T](https://img.shields.io/badge/Powered%20by-ML4Trading-blue)](https://ml4trading.io) [![Twitter](https://img.shields.io/twitter/follow/ml4trading.svg?style=social)](https://twitter.com/ml4trading)|

Zipline is a Pythonic event-driven system for backtesting, developed and used as the backtesting and live-trading engine by [crowd-sourced investment fund Quantopian](https://www.bizjournals.com/boston/news/2020/11/10/quantopian-shuts-down-cofounders-head-elsewhere.html). Since it closed late 2020, the domain that had hosted these docs expired. The library is used extensively in the book [Machine Larning for Algorithmic Trading](https://ml4trading.io)
by [Stefan Jansen](https://www.linkedin.com/in/applied-ai/) who is trying to keep the library up to date and available to his readers and the wider Python algotrading community.

- [Join our Community!](https://exchange.ml4trading.io)
- [Documentation](https://zipline.ml4trading.io)

## Features

- **Ease of Use:** Zipline tries to get out of your way so that you can focus on algorithm development. See below for a code example.
- **Batteries Included:** many common statistics like moving average and linear regression can be readily accessed from within a user-written algorithm.
- **PyData Integration:** Input of historical data and output of performance statistics are based on Pandas DataFrames to integrate nicely into the existing PyData ecosystem.
- **Statistics and Machine Learning Libraries:** You can use libraries like matplotlib, scipy, statsmodels, and scikit-klearn to support development, analysis, and visualization of state-of-the-art trading systems.

## Installation

Zipline supports Python >= 3.7 and is compatible with current versions of the relevant [NumFOCUS](https://numfocus.org/sponsored-projects?_sft_project_category=python-interface) libraries, including [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html).

If your system meets the pre-requisites described in the [installation instructions](https://zipline.ml4trading.io/install.html), you can install Zipline using pip by running:

```bash
pip install zipline-reloaded
```

Alternatively, if you are using the [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) distributions, you can use

```bash
conda install -c ml4t -c conda-forge -c ranaroussi zipline-reloaded
```

You can also [enable](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html) `conda-forge` by listing it in your `.condarc`.

In case you are installing `zipline-reloaded` alongside other packages and encounter [conflict errors](https://github.com/conda/conda/issues/9707), consider using [mamba](https://github.com/mamba-org/mamba) instead.

See the [installation](https://zipline.ml4trading.io/install.html) section of the docs for more detailed instructions.

## Quickstart

See our [getting started tutorial](https://zipline.ml4trading.io/beginner-tutorial).

The following code implements a simple dual moving average algorithm.

```python
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

You can then run this algorithm using the Zipline CLI. But first, you need to download some market data with historical prices and trading volumes:

```bash
$ zipline ingest -b quandl
$ zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark
```

This will download asset pricing data sourced from [Quandl](https://www.quandl.com/databases/WIKIP/documentation?anchor=companies), and stream it through the algorithm over the specified time range. Then, the resulting performance DataFrame is saved as `dma.pickle`, which you can load and analyze from Python.

You can find other examples in the [zipline/examples](https://github.com/stefan-jansen/zipline-reloaded/tree/main/src/zipline/examples) directory.

## Questions, suggestions, bugs?

If you find a bug or have other questions about the library, feel free to [open an issue](https://github.com/stefan-jansen/zipline/issues/new) and fill out the template.
