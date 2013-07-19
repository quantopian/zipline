#!/usr/bin/python

from datetime import datetime
import pytz

from zipline.algorithm import TradingAlgorithm
from zipline.utils import factory


class MyAlgo(TradingAlgorithm):
    def initialize(self):
        # To keep track of whether we invested in the stock or not
        self.invested = False

    def handle_data(self, data):
        ticker = 'SPY'

        if not self.invested:
            shares = self.portfolio.cash/data[ticker].price
            self.order(ticker, shares)
            self.invested = True

if __name__ == '__main__':
    start = datetime(2010, 7, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2013, 6, 29, 0, 0, 0, 0, pytz.utc)
    data = factory.load_from_yahoo(stocks=['SPY'], indexes={}, start=start)

    algo = MyAlgo()

    daily_stats = algo.run(data)
    risk_report = algo.risk_report
    for year_period in risk_report.year_periods:
        print('Results for ' + str(year_period.algorithm_returns.index[-1]))
        # Quiet the returns data from the output since it is a lot
        year_period.algorithm_returns = None
        year_period.benchmark_returns = None
        print(year_period)
        print
        print
