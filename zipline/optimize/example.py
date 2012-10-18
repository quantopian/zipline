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


# WARNING: This file is still work in progress and contains rather
# random code snippets.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from zipline.gens.mavg import MovingAverage
from zipline.algorithm import TradingAlgorithm
from zipline.gens.transform import batch_transform


class DMA(TradingAlgorithm):
    """Dual Moving Average algorithm.
    """
    def initialize(self, amount=100, short_window=20, long_window=40):
        self.amount = amount
        self.events = 0

        self.invested = {}
        for sid in self.sids:
            self.invested[sid] = False

        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           market_aware=True,
                           days=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           market_aware=True,
                           days=long_window)

    def handle_data(self, data):
        self.events += 1

        for sid in self.sids:
            # access transforms via their user-defined tag
            if (data[sid].short_mavg['price'] >
                data[sid].long_mavg['price']) \
            and not self.invested[sid]:
                self.order(sid, self.amount)
                self.invested[sid] = True
            elif (data[sid].short_mavg['price'] <
                  data[sid].long_mavg['price']) \
                and self.invested[sid]:
                self.order(sid, -self.amount)
                self.invested[sid] = False


class DualMovingAverage(TradingAlgorithm):
    """Dual Moving Average algorithm.
    """
    def initialize(self, short_window=200, long_window=400):
        self.short_mavg = []
        self.long_mavg = []

        self.invested = False

        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           market_aware=True,
                           days=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           market_aware=True,
                           days=long_window)

    def handle_data(self, data):
        self.short_mavg.append(data['AAPL'].short_mavg['price'])
        self.long_mavg.append(data['AAPL'].long_mavg['price'])

        if (data['AAPL'].short_mavg['price'] >
            data['AAPL'].long_mavg['price']) and not self.invested:
            self.order('AAPL', 100)
            self.invested = True
        elif (data['AAPL'].short_mavg['price'] <
              data['AAPL'].long_mavg['price']) and self.invested:
            self.order('AAPL', -100)
            self.invested = False


def load_close_px(indexes=None, stocks=None):
    from pandas.io.data import DataReader
    import pytz
    from collections import OrderedDict

    if indexes is None:
        indexes = {'SPX': '^GSPC'}
    if stocks is None:
        stocks = ['AAPL', 'GE', 'IBM', 'MSFT', 'XOM', 'AA', 'JNJ', 'PEP', 'KO']

    start = pd.datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = pd.datetime(2000, 1, 1, 0, 0, 0, 0, pytz.utc)

    data = OrderedDict()

    for stock in stocks:
        print stock
        stkd = DataReader(stock, 'yahoo', start, end).sort_index()
        data[stock] = stkd

    for name, ticker in indexes.iteritems():
        print name
        stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
        data[name] = stkd

    df = pd.DataFrame({key: d['Close'] for key, d in data.iteritems()})

    df.index = df.index.tz_localize(pytz.utc)

    df.save('close_px.dat')
    return df


def run((short_window, long_window)):
    data = pd.load('close_px.dat')
    #data = load_close_px()
    myalgo = DMA([0, 1],
                 amount=100,
                 short_window=short_window,
                 long_window=long_window)
    stats = myalgo.run(data)
    stats['sw'] = short_window
    stats['lw'] = long_window
    return stats


def explore_params():
    sws, lws = np.mgrid[10:20:5, 10:20:5]

    stats_all = map(run, zip(sws.flatten(), lws.flatten()))
    stats = pd.concat(stats_all)
    returns = stats.groupby(['sw', 'lw']).sum()

    plt.contourf(sws, lws, returns.returns.reshape(sws.shape))
    plt.xlabel('Short window length')
    plt.ylabel('Long window length')
    plt.savefig('DMA_contour.png')
    plt.show()


def get_opt_holdings_qp(univ_rets, track_rets):
    from cvxopt import matrix
    from cvxopt.solvers import qp
    # set up the QP for CVXOPT
    # .5 x' P x + q'x
    # P = 2 * R'R
    # q = - 2 * bmk'R
    R = univ_rets.values
    b = track_rets.values
    P = matrix(2 * np.dot(R.T, R))
    q = matrix(-2 * np.dot(R.T, b))
    result = qp(P, q)
    if result['status'] != 'optimal':
        raise Exception('optimum not reached by QP')
    return pd.Series(np.array(result['x']).ravel(), index=univ_rets.columns)


def opt_portfolio(cov, budget, min_return):
    from cvxopt import matrix
    from cvxopt.solvers import qp
    n = len(cov)
    cov = matrix(2 * cov)
    q = matrix(np.zeros(n))

    h = matrix(budget)  # G*x < h
    # coneqp
    result = qp(cov, q, h=h)
    if result['status'] != 'optimal':
        raise Exception('optimum not reached by QP')

    return pd.Series(np.array(result['x']).ravel())


def calc_te(weights, univ_rets, track_rets):
    port_rets = (univ_rets * weights).sum(1)
    return (port_rets - track_rets).std()


def plot_returns(port_returns, bmk_returns):
    plt.figure()
    cum_port = ((1 + port_returns).cumprod() - 1)
    cum_bmk = ((1 + bmk_returns).cumprod() - 1)
    # cum_port = port_returns.cumsum()
    # cum_bmk = bmk_returns.cumsum()
    cum_port.plot(label='Portfolio returns')
    cum_bmk.plot(label='Benchmark')
    plt.title('Portfolio performance')
    plt.legend(loc='best')

#print run((10, 20))

import statsmodels.api as sm


@batch_transform
def ols_transform(data, spreads):
    p0 = data.price['PEP']
    p1 = sm.add_constant(data.price['KO'])
    beta, intercept = sm.OLS(p0, p1).fit().params

    spread = (data.price['PEP'] - (beta * data.price['KO'] + intercept))[-1]

    if len(spreads) > 10:
        z_score = (spread - np.mean(spreads[-10:])) / np.std(spreads[-10:])
    else:
        z_score = np.nan

    spreads.append(spread)

    return z_score


class Pairtrade(TradingAlgorithm):
    def initialize(self):
        self.spreads = []
        self.invested = False
        self.ols_transform = ols_transform(refresh_period=10, days=10)

    def handle_data(self, data):
        zscore = self.ols_transform.handle_data(data, self.spreads)

        if zscore == np.nan:
            return

        if zscore >= 2.0 and not self.invested:
            self.order('PEP', int(100 / data['PEP'].price))
            self.order('KO', -int(100 / data['KO'].price))
        elif zscore <= -2.0 and not self.invested:
            self.order('KO', -int(100 / data['KO'].price))
            self.order('PEP', int(100 / data['PEP'].price))
        elif abs(zscore) < .5 and self.invested:
            pass


def run_pairtrade():
    data = load_close_px()
    data.save('close_px.dat')
    #data = pd.load('close_px.dat')
    myalgo = Pairtrade()
    stats = myalgo.run(data)
    return stats
