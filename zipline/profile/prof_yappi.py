from __future__ import division

import logging
from zipline.core.devsimulator import AddressAllocator
import zipline.finance
from zipline.optimize.factory import create_predictable_zipline
import pandas as pd
import numpy as np
import os.path

def convert_ystats(ystats):
    """Convert yappi.get_stats().func_stats object to pandas
    DataFrame.

    """
    func_names = [os.path.split(item[0])[-1] for item in ystats]
    ncall = [float(item[1]) for item in ystats]
    ttot = [float(item[2]) for item in ystats]
    tsub = [float(item[3]) for item in ystats]
    tavg = [float(item[4]) for item in ystats]
    stats = pd.DataFrame({'ncall': ncall, 'ttot': ttot, 'tsub': tsub, 'tavg': tavg}, index=func_names)

    return stats


allocator = AddressAllocator(1000)

config = {  'allocator'   :allocator,
            'sid'         :133,
            'trade_count' :5000,
            'amplitude'   :30,
            'base_price'  :50
         }

LOGGER = logging.getLogger('ZiplineLogger')

import yappi

def gen_single_stats(func, *args, **kwargs):
    """Profile func(*args, **kwargs) with yappi.

    Returns DataFrame of statistics.
    """
    yappi.start()
    func(*args, **kwargs)
    yappi.stop()
    return convert_ystats(yappi.get_stats().func_stats)

def gen_avg_stats(func, runs=1, *args, **kwargs):
    """Profile func(*args, **kwargs) with yappi. Runs multiple times at computes the average.

    Returns DataFrame of average statistics.
    """

    avg_stats = pd.concat([gen_single_stats() for i in range(runs)], keys=range(runs))
    grouped = avg_stats.groupby(level=1)

    return grouped.aggregate(np.mean)

def run_updown(fname='before_stats.csv'):
    """Profile a zipline with the UpDown tradesource (does not require
    DB access) and the buy/sell algorithm (requires no
    computation).

    Saves output statics under fname.

    Returns Dataframe of statistics.
    """
    zp, _ = create_predictable_zipline(config, simulate=False)
    stats = gen_single_stats(zp.simulate, blocking=True)
    stats.to_csv(fname)

    return stats

def calc_speedup(before='before_stats.csv', after='after_stats.csv'):
    """Calculate speed-up between two previously run and saved
    statistics under filename before and after.

    Prints DataFrame of top 30 speed-ups and top 30 slow-downs.

    """
    old = pd.DataFrame.from_csv(before)
    new = pd.DataFrame.from_csv(after)
    speed_up = old / new
    speed_up = speed_up.fillna(1)
    speed_up = speed_up.sort(column='ttot', ascending=False)
    slow_down = speed_up.sort(column='ttot', ascending=True)
    print speed_up[:30]
    print slow_down[:30]

if __name__ == '__main__':
    run_updown()
    yappi.print_stats(sort_type=yappi.SORTTYPE_TTOT)