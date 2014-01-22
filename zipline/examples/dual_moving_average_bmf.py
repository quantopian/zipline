#!/usr/bin/env python
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
from zipline.finance import trading
from zipline.transforms import MovingAverage
from zipline.utils.factory import load_from_yahoo

from zipline.utils.tradingcalendar_bmf import BvmfTradingCalendar

from datetime import datetime
import pytz


class DualMovingAverage(TradingAlgorithm):
    SHORT = 1
    LONG = 2

    """Dual Moving Average Crossover algorithm.

    This algorithm buys apple once its short moving average crosses
    its long moving average (indicating upwards momentum) and sells
    its shares once the averages cross again (indicating downwards
    momentum).

    """
    def initialize(self, short_window=5, long_window=20):
        # Add 2 mavg transforms, one with a long window, one
        # with a short window.
        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           window_length=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           window_length=long_window)

        # To keep track of whether we invested in the stock or not

        self.invested = self.SHORT
        self.stock = 'VALE5.SA'
        self.amount = 500

    def handle_data(self, data):
        self.short_mavg = data[self.stock].short_mavg['price']
        self.long_mavg = data[self.stock].long_mavg['price']
        self.buy = False
        self.sell = False

        if self.short_mavg > self.long_mavg and self.invested == self.SHORT:
            amount = self.portfolio.positions[self.stock].amount
            trade_amount = abs(amount) + self.amount
            print 'carring position', trade_amount
            self.order(self.stock, trade_amount)

            self.invested = self.LONG
            self.buy = True
        elif self.short_mavg < self.long_mavg and self.invested == self.LONG:
            amount = self.portfolio.positions[self.stock].amount
            trade_amount = -1 * amount - self.amount
            print 'shorting position', trade_amount
            self.order(self.stock, trade_amount)

            self.invested = self.SHORT
            self.sell = True

        self.record(short_mavg=self.short_mavg,
                    long_mavg=self.long_mavg,
                    buy=self.buy,
                    sell=self.sell)

if __name__ == '__main__':
    bvsp = trading.TradingEnvironment(
        bm_symbol="^BVSP",
        exchange_tz="America/Sao_Paulo",
        tradingcalendar=BvmfTradingCalendar())

    with bvsp:
        start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
        end = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
        data = load_from_yahoo(stocks=['VALE5.SA'], indexes={}, start=start,
                               end=end)

        dma = DualMovingAverage()
        results = dma.run(data)

        br = trading.environment.benchmark_returns
        bm_returns = br[(br.index >= start) & (br.index <= end)]
        results['benchmark_returns'] = (1 + bm_returns).cumprod().values
        results['algorithm_returns'] = (1 + results.returns).cumprod()
        fig = plt.figure()
        ax1 = fig.add_subplot(211, ylabel='cumulative returns')

        results[['algorithm_returns', 'benchmark_returns']].plot(ax=ax1,
                                                                 sharex=True)

        ax2 = fig.add_subplot(212)
        data[dma.stock].plot(ax=ax2, color='r')
        results[['short_mavg', 'long_mavg']].plot(ax=ax2)

        ax2.plot(results.ix[results.buy].index,
                 results.short_mavg[results.buy],
                 '^', markersize=10, color='m')
        ax2.plot(results.ix[results.sell].index,
                 results.short_mavg[results.sell],
                 'v', markersize=10, color='k')
        plt.legend(loc=0)

        sharpe = [risk['sharpe'] for risk in dma.risk_report['one_month']]
        print("Monthly Sharpe ratios: {0}".format(sharpe))
        plt.gcf().set_size_inches(18, 8)

        plt.show()
