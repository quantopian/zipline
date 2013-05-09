#!/usr/bin/python
#
# Copyright 2013 Quantopian, Inc.
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
from zipline.utils.factory import load_from_yahoo

# Import exponential moving average from talib wrapper
from zipline.transforms.ta import EMA

from datetime import datetime
import pytz


class DualEMATaLib(TradingAlgorithm):
    """Dual Moving Average Crossover algorithm.

    This algorithm buys apple once its short moving average crosses
    its long moving average (indicating upwards momentum) and sells
    its shares once the averages cross again (indicating downwards
    momentum).

    """
    def initialize(self, short_window=20, long_window=40):
        # Add 2 mavg transforms, one with a long window, one
        # with a short window.
        self.short_ema_trans = EMA('AAPL', window_length=short_window)
        self.long_ema_trans = EMA('AAPL', window_length=long_window)

        # To keep track of whether we invested in the stock or not
        self.invested = False

    def handle_data(self, data):
        self.short_ema = self.short_ema_trans.handle_data(data)
        self.long_ema = self.long_ema_trans.handle_data(data)
        if self.short_ema is None or self.long_ema is None:
            return

        self.buy = False
        self.sell = False

        if self.short_ema > self.long_ema and not self.invested:
            self.order('AAPL', 100)
            self.invested = True
            self.buy = True
        elif self.short_ema < self.long_ema and self.invested:
            self.order('AAPL', -100)
            self.invested = False
            self.sell = True

        self.record(AAPL=data['AAPL'].price,
                    short_ema=self.short_ema,
                    long_ema=self.long_ema,
                    buy=self.buy,
                    sell=self.sell)

if __name__ == '__main__':
    start = datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(1991, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
                           end=end)

    dma = DualEMATaLib()
    results = dma.run(data).dropna()

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
    plt.show()
