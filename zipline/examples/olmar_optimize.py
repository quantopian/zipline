from datetime import datetime
import pytz
import numpy as np

from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo

import olmar_ext as olmar

STOCKS = ['CERN', 'DLTR', 'ROST', 'MSFT', 'SBUX']


def run_algo(eps=1, window_length=30):
    start = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_bars_from_yahoo(
        stocks=STOCKS, indexes={}, start=start, end=end)

    initialize_params = {'eps': eps,
                         'window_length': window_length}

    algo = TradingAlgorithm(initialize=olmar.initialize,
                            handle_data=olmar.handle_data,
                            initialize_params=initialize_params)

    perf = algo.run(data)  # flake8: noqa
    # Minimize negative sharpe = maximize sharpe
    return -algo.risk_report['twelve_month'][-1]['sharpe']

test_eps = np.linspace(1, 20, 5)
results_eps = map(run_algo, test_eps)
print(results_eps)

test_window_length = np.arange(10, 50, 10)
results_window_length = map(lambda x: run_algo(window_length=x),
                            test_window_length)
print(results_window_length)
