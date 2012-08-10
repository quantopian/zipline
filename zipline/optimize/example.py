from zipline.lines import Zipline
from zipline.optimize.algorithms import DMA
import pandas as pd
import matplotlib.pyplot as plt
import cProfile

def run():
    myalgo = DMA(sid=0, amount=100)
    zp = Zipline(algorithm=myalgo, sources='S&P')
    stats = zp.run()
    print stats
    return stats


#cProfile.run('run()')

stats = run()
stats.returns.plot()
plt.show()