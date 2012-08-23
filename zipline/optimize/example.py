from zipline.lines import Zipline
import pandas as pd
import pandas.io.data as dt
from pandas.io.data import DataReader

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import cProfile
from zipline.gens.mavg import MovingAverage
from zipline.optimize.algorithms import TradingAlgorithm
from datetime import timedelta

#from mpi4py_map import map

# Inherits from Algorithm base class
class DMA(TradingAlgorithm):
    """Dual Moving Average algorithm.
    """
    def __init__(self, sids, amount=100, short_window=20, long_window=40):
        self.sids = sids
        self.amount = amount
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.orders = []

        self.prices = []
        self.events = 0

        self.invested = {}
        for sid in self.sids:
            self.invested[sid] = False

        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           market_aware=False,
                           delta=timedelta(days=int(short_window)))

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           market_aware=False,
                           delta=timedelta(days=int(long_window)))

    def handle_data(self, data):
        self.events += 1

        for sid in self.sids:
            # access transforms via their user-defined tag
            if (data[sid].short_mavg['price'] > data[sid].long_mavg['price']) and not self.invested[sid]:
                self.order(sid, self.amount)
                self.invested[sid] = True
            elif (data[sid].short_mavg['price'] < data[sid].long_mavg['price']) and self.invested[sid]:
                self.order(sid, -self.amount)
                self.invested[sid] = False


class DanVWAP(TradingAlgorithm):
    """Dual Moving Average algorithm.
    """
    def __init__(self, sids, amount=100, short_window=20, long_window=40):
        self.sids = sids
        self.amount = amount
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.orders = []

        self.prices = []
        self.port = 0

        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           market_aware=False,
                           delta=timedelta(days=int(short_window)))

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           market_aware=False,
                           delta=timedelta(days=int(long_window)))

    def handle_data(self, data):
        for sid in self.sids:
            average=data[sid].vwap(5)
            price=data[sid].price

            if price>average*1.05:
                self.order(sid, self.amount)


def load_close_px(indexes=None, stocks=None):
    if indexes is None:
        indexes = {'SPX' : '^GSPC'}
    if stocks is None:
        stocks = ['AAPL', 'GE', 'IBM', 'MSFT', 'XOM', 'AA', 'JNJ', 'PEP']

    start = pd.datetime(1990, 1, 1)
    end = pd.datetime.today()

    data = {}
    for stock in stocks:
        print stock
        stkd = DataReader(stock, 'yahoo', start, end).sort_index()
        data[stock] = stkd

    for name, ticker in indexes.iteritems():
        print name
        stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
        data[name] = stkd

    df = pd.DataFrame({key: d['Close'] for key, d in data.iteritems()})

    return df


def run((short_window, long_window)):
    data = pd.DataFrame.from_csv('SP500.csv')
    myalgo = DMA([0], amount=100, short_window=short_window, long_window=long_window)
    stats = myalgo.run(data, compute_risk_metrics=False)
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
