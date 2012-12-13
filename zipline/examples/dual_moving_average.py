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

import matplotlib.pyplot as plt

from zipline.algorithm import TradingAlgorithm
from zipline.transforms import MovingAverage
from zipline.utils.factory import load_from_yahoo


class DualMovingAverage(TradingAlgorithm):
    """Dual Moving Average Crossover algorithm.

    This algorithm buys apple once its short moving average crosses
    its long moving average (indicating upwards momentum) and sells
    its shares once the averages cross again (indicating downwards
    momentum).

    """
    def initialize(self, short_window=200, long_window=400):
        # Add 2 mavg transforms, one with a long window, one
        # with a short window.
        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           window_length=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           window_length=long_window)

        # To keep track of whether we invested in the stock or not
        self.invested = False

        self.record_variable('short_mavg')
        self.record_variable('long_mavg')
        self.record_variable('buy')
        self.record_variable('sell')

    def handle_data(self, data):
        self.short_mavg = data['AAPL'].short_mavg['price']
        self.long_mavg = data['AAPL'].long_mavg['price']
        if self.short_mavg > self.long_mavg and not self.invested:
            self.order('AAPL', 100)
            self.invested = True
            self.buy = True
            self.sell = False
        elif self.short_mavg < self.long_mavg and self.invested:
            self.order('AAPL', -100)
            self.invested = False
            self.sell = True
            self.buy = False
        else:
            self.buy = False
            self.sell = False

if __name__ == '__main__':
    data = load_from_yahoo(stocks=['AAPL'], indexes={})
    dma = DualMovingAverage()
    results = dma.run(data)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    results.portfolio_value.plot(ax=ax1)

    # todo: should turn up in the stats automatically
    data['short'] = dma.record_var_values['short_mavg'][:-1]
    data['long'] = dma.record_var_values['long_mavg'][:-1]
    data['buy'] = dma.record_var_values['buy'][:-1]
    data['sell'] = dma.record_var_values['sell'][:-1]

    ax2 = fig.add_subplot(212)
    data[['AAPL', 'short', 'long']].plot(ax=ax2)

    ax2.plot(data.ix[data.buy].index, data.short[data.buy],
             '^', markersize=10, color='m')
    ax2.plot(data.ix[data.sell].index, data.short[data.sell],
             'v', markersize=10, color='k')
    plt.legend(loc=0)
    plt.show()
