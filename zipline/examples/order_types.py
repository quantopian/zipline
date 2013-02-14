#!/usr/bin/python
#
# Copyright 2012 Quantopian, Inc.
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


######################
#  New order function
######################
# amount > 0 :: Buy/Cover
# amount < 0 :: Sell/Short
# Market order:    order(sid,amount)                      so we don't break your algos
#           or:    order(sid,amount, "market")            "market" is redundant
# Stop order:      order(sid,amount, "stop",      stop_price)
# Limit order:     order(sid,amount, "limit",     limit_price)
# StopLimit order: order(sid,amount, "stoplimit", stop_price, limit_price)
######################



import matplotlib.pyplot as plt

from zipline.algorithm import TradingAlgorithm
from zipline.transforms import MovingAverage
from zipline.utils.factory import load_from_yahoo

class OrderTypes(TradingAlgorithm):
    """Dual Moving Average Crossover algorithm.

    This algorithm buys apple once its short moving average crosses
    its long moving average (indicating upwards momentum) and sells
    its shares once the averages cross again (indicating downwards
    momentum).

    """
    def initialize(self, short_window=5, long_window=50):
        # Add 2 mavg transforms, one with a long window, one
        # with a short window.
        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           window_length=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           window_length=long_window)

        # To keep track of whether we invested in the stock or not
        self.invested = False

        self.short_mavgs = []
        self.long_mavgs = []

    def handle_data(self, data):
        short_mavg = data['AAPL'].short_mavg['price']
        long_mavg = data['AAPL'].long_mavg['price']
        if short_mavg > long_mavg and not self.invested:
            self.order('AAPL', 100, data['AAPL'].price - 1.00)
            self.invested = True
        elif short_mavg < long_mavg and self.invested:
            self.order('AAPL', -100)
            self.invested = False

        # Save mavgs for later analysis.
        self.short_mavgs.append(short_mavg)
        self.long_mavgs.append(long_mavg)


if __name__ == '__main__':
    data = load_from_yahoo(stocks=['AAPL'], indexes={})
    dma = OrderTypes()

    results = dma.run(data)

    results.portfolio_value.plot()

    data['short'] = dma.short_mavgs
    data['long'] = dma.long_mavgs
    data[['AAPL', 'short', 'long']].plot()
    plt.legend(loc=0)
    plt.show()
