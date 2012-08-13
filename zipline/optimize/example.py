from zipline.lines import Zipline
import pandas as pd
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import cProfile
from zipline.gens.mavg import MovingAverage
from zipline.optimize.algorithms import TradingAlgorithm
from datetime import timedelta

from mpi4py_map import map

# Inherits from Algorithm base class
class DMA(TradingAlgorithm):
    """Dual Moving Average algorithm.
    """
    def __init__(self, sid, amount=100, short_window=20, long_window=40):
        self.sids = [sid]
        self.amount = amount
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.orders = []
        self.market_entered = False
        self.prices = []
        self.events = 0

        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           market_aware=False,
                           delta=timedelta(days=int(short_window)))

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           market_aware=False,
                           delta=timedelta(days=int(long_window)))

    def handle_data(self, data):
        self.events += 1
        sid = self.sids[0]
        # access transforms via their user-defined tag
        if (data[sid].short_mavg > data[sid].long_mavg) and not self.market_entered:
            self.order(sid, 100)
            self.market_entered = True
        elif (data[sid].short_mavg < data[sid].long_mavg) and self.market_entered:
            self.order(sid, -100)
            self.market_entered = False


def run((short_window, long_window)):
    data = pd.DataFrame.from_csv('SP500.csv')
    myalgo = DMA(sid=0, amount=100, short_window=short_window, long_window=long_window)
    stats = myalgo.run(data, compute_risk_metrics=False)
    stats['sw'] = short_window
    stats['lw'] = long_window
    return stats

sws, lws = np.mgrid[50:80:5, 100:140:5]

stats_all = map(run, zip(sws.flatten(), lws.flatten()))

# for sw, lw in zip(sws.flatten(), lws.flatten()):
#     stats = run(short_window=sw, long_window=lw)
#     stats_all.append(stats)

stats = pd.concat(stats_all)
returns = stats.groupby(['sw', 'lw']).sum()
plt.contourf(sws, lws, returns.returns.reshape(sws.shape))
plt.xlabel('Short window length')
plt.ylabel('Long window length')
plt.savefig('DMA_contour.png')
plt.show()