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

# Import exponential moving average from talib wrapper
from zipline.transforms.ta import EMA


def initialize(context):
    context.asset = symbol('AAPL')

    # Add 2 mavg transforms, one with a long window, one with a short window.
    context.short_ema_trans = EMA(timeperiod=20)
    context.long_ema_trans = EMA(timeperiod=40)

    # To keep track of whether we invested in the stock or not
    context.invested = False


def handle_data(context, data):
    short_ema = context.short_ema_trans.handle_data(data)
    long_ema = context.long_ema_trans.handle_data(data)
    if short_ema is None or long_ema is None:
        return

    buy = False
    sell = False

    if (short_ema > long_ema).all() and not context.invested:
        order(context.asset, 100)
        context.invested = True
        buy = True
    elif (short_ema < long_ema).all() and context.invested:
        order(context.asset, -100)
        context.invested = False
        sell = True

    record(AAPL=data[context.security].price,
           short_ema=short_ema[context.asset],
           long_ema=long_ema[context.asset],
           buy=buy,
           sell=sell)


if __name__ == '__main__':
    from datetime import datetime
    import matplotlib.pyplot as plt
    import pytz
    from zipline.algorithm import TradingAlgorithm
    from zipline.api import order, record, symbol
    from zipline.utils.factory import load_from_yahoo

    start = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2014, 11, 1, 0, 0, 0, 0, pytz.utc)
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
                           end=end)

    algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
                            identifiers=['AAPL'])
    results = algo.run(data).dropna()

    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='portfolio value')
    results.portfolio_value.plot(ax=ax1)

    ax2 = fig.add_subplot(212)
    results[['AAPL', 'short_ema', 'long_ema']].plot(ax=ax2)

    ax2.plot(results.ix[results.buy].index, results.short_ema[results.buy],
             '^', markersize=10, color='m')
    ax2.plot(results.ix[results.sell].index, results.short_ema[results.sell],
             'v', markersize=10, color='k')
    plt.legend(loc=0)
    plt.gcf().set_size_inches(18, 8)
    plt.show()
