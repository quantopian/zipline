#!/usr/bin/env python
#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Dual Moving Average Crossover algorithm.

This algorithm buys apple once its short moving average crosses
its long moving average (indicating upwards momentum) and sells
its shares once the averages cross again (indicating downwards
momentum).

"""
import os
from zipline.api import order, record, symbol
from zipline.finance import commission, slippage

# Import exponential moving average from talib wrapper
try:
    from talib import EMA
except ImportError as exc:
    msg = (
        "Unable to import module TA-lib. Use `pip install TA-lib` to "
        "install. Note: if installation fails, you might need to install "
        "the underlying TA-lib library (more information can be found in "
        "the zipline installation documentation)."
    )
    raise ImportError(msg) from exc


def initialize(context):
    context.asset = symbol("AAPL")

    # To keep track of whether we invested in the stock or not
    context.invested = False

    # Explicitly set the commission/slippage to the "old" value until we can
    # rebuild example data.
    # github.com/quantopian/zipline/blob/master/tests/resources/
    # rebuild_example_data#L105
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())


def handle_data(context, data):
    trailing_window = data.history(context.asset, "price", 40, "1d")
    if trailing_window.isnull().values.any():
        return
    short_ema = EMA(trailing_window.values, timeperiod=20)
    long_ema = EMA(trailing_window.values, timeperiod=40)

    buy = False
    sell = False

    if (short_ema[-1] > long_ema[-1]) and not context.invested:
        order(context.asset, 100)
        context.invested = True
        buy = True
    elif (short_ema[-1] < long_ema[-1]) and context.invested:
        order(context.asset, -100)
        context.invested = False
        sell = True

    record(
        AAPL=data.current(context.asset, "price"),
        short_ema=short_ema[-1],
        long_ema=long_ema[-1],
        buy=buy,
        sell=sell,
    )


# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(context=None, results=None):
    import matplotlib.pyplot as plt
    import logging

    logging.basicConfig(
        format="[%(asctime)s-%(levelname)s][%(name)s]\n %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    log = logging.getLogger("Algorithm")

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel("Portfolio value (USD)")

    ax2 = fig.add_subplot(212)
    ax2.set_ylabel("Price (USD)")

    # If data has been record()ed, then plot it.
    # Otherwise, log the fact that no data has been recorded.
    if "AAPL" in results and "short_ema" in results and "long_ema" in results:
        results[["AAPL", "short_ema", "long_ema"]].plot(ax=ax2)

        ax2.plot(
            results.index[results.buy],
            results.loc[results.buy, "long_ema"],
            "^",
            markersize=10,
            color="m",
        )
        ax2.plot(
            results.index[results.sell],
            results.loc[results.sell, "short_ema"],
            "v",
            markersize=10,
            color="k",
        )
        plt.legend(loc=0)
        plt.gcf().set_size_inches(18, 8)
    else:
        msg = "AAPL, short_ema and long_ema data not captured using record()."
        ax2.annotate(msg, xy=(0.1, 0.5))
        log.info(msg)

    plt.show()

    if "PYTEST_CURRENT_TEST" in os.environ:
        plt.close("all")


def _test_args():
    """Extra arguments to use when zipline's automated tests run this example."""
    import pandas as pd

    return {"start": pd.Timestamp("2014-01-01"), "end": pd.Timestamp("2014-11-01")}
