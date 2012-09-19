# WARNING: This file is still work in progress and contains rather
# random code snippets.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import cProfile
from zipline.gens.mavg import MovingAverage
from zipline.gens.cov import  CovTransform, cov
from zipline.algorithm import TradingAlgorithm
from zipline.gens.transform import BatchTransform, batch_transform

@batch_transform
def cov(data):
    return data.price.cov()

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

        self.cov = cov(sids=self.sids, refresh_period=1, days=5)

    def handle_data(self, data):
        self.events += 1

        cov = self.cov.handle_data(data)
        print cov

        for sid in self.sids:
            # access transforms via their user-defined tag
            if (data[sid].short_mavg['price'] > data[sid].long_mavg['price']) and not self.invested[sid]:
                self.order(sid, self.amount)
                self.invested[sid] = True
            elif (data[sid].short_mavg['price'] < data[sid].long_mavg['price']) and self.invested[sid]:
                self.order(sid, -self.amount)
                self.invested[sid] = False


def load_close_px(indexes=None, stocks=None):
    from pandas.io.data import DataReader
    import pytz

    if indexes is None:
        indexes = {'SPX' : '^GSPC'}
    if stocks is None:
        stocks = ['AAPL'] #, 'GE', 'IBM', 'MSFT', 'XOM', 'AA', 'JNJ', 'PEP']

    #start = pd.datetime(1990, 1, 1)
    start = pd.datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = pd.datetime(1992, 1, 1, 0, 0, 0, 0, pytz.utc) #pd.datetime.today()

    data = {}

    for stock in stocks:
        print stock
        stkd = DataReader(stock, 'yahoo', start, end).sort_index()
        data[stock] = stkd

    for name, ticker in indexes.iteritems():
        print name
        stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
        data[name] = stkd

    #df = pd.DataFrame({key: d['Close'] for key, d in data.iteritems()})
    df = pd.DataFrame({i: d['Close'] for i, d in enumerate(data.itervalues())})
    df.index = df.index.tz_localize(pytz.utc)

    return df


def run((short_window, long_window)):
    #data = pd.DataFrame.from_csv('SP500.csv')
    #data = pd.DataFrame.from_csv('aapl.csv') #load_close_px()
    data = load_close_px()
    myalgo = DMA([0, 1], amount=100, short_window=short_window, long_window=long_window)
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

#stats = run((10, 50))

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

    h = matrix(budget) # G*x < h
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

print run((10, 20))