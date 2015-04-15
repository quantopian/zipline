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
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import pytz

from zipline.algorithm import TradingAlgorithm
from zipline.transforms import batch_transform
from zipline.utils.factory import load_from_yahoo
from zipline.sources.data_frame_source import DataFrameSource


@batch_transform
def ols_transform(data, sid1, sid2):
    """Computes regression coefficient (slope and intercept)
    via Ordinary Least Squares between two SIDs.
    """
    p0 = data.price[sid1]
    p1 = sm.add_constant(data.price[sid2], prepend=True)
    slope, intercept = sm.OLS(p0, p1).fit().params

    return slope, intercept


class Pairtrade(TradingAlgorithm):
    """Pairtrading relies on cointegration of two stocks.

    The expectation is that once the two stocks drifted apart
    (i.e. there is spread), they will eventually revert again. Thus,
    if we short the upward drifting stock and long the downward
    drifting stock (in short, we buy the spread) once the spread
    widened we can sell the spread with profit once they converged
    again. A nice property of this algorithm is that we enter the
    market in a neutral position.

    This specific algorithm tries to exploit the cointegration of
    Pepsi and Coca Cola by estimating the correlation between the
    two. Divergence of the spread is evaluated by z-scoring.
    """

    def initialize(self, window_length=100):
        self.spreads = []
        self.invested = 0
        self.window_length = window_length
        self.ols_transform = ols_transform(refresh_period=self.window_length,
                                           window_length=self.window_length)
        self.PEPsid = self.symbol('PEP')
        self.KOsid = self.symbol('KO')

    def handle_data(self, data):
        ######################################################
        # 1. Compute regression coefficients between PEP and KO
        params = self.ols_transform.handle_data(data, self.PEPsid, self.KOsid)
        if params is None:
            return
        intercept, slope = params

        ######################################################
        # 2. Compute spread and zscore
        zscore = self.compute_zscore(data, slope, intercept)
        self.record(zscores=zscore)

        ######################################################
        # 3. Place orders
        self.place_orders(data, zscore)

    def compute_zscore(self, data, slope, intercept):
        """1. Compute the spread given slope and intercept.
           2. zscore the spread.
        """
        spread = (data[self.PEPsid].price -
                  (slope * data[self.KOsid].price + intercept))
        self.spreads.append(spread)
        spread_wind = self.spreads[-self.window_length:]
        zscore = (spread - np.mean(spread_wind)) / np.std(spread_wind)
        return zscore

    def place_orders(self, data, zscore):
        """Buy spread if zscore is > 2, sell if zscore < .5.
        """
        if zscore >= 2.0 and not self.invested:
            self.order(self.PEPsid, int(100 / data[self.PEPsid].price))
            self.order(self.KOsid, -int(100 / data[self.KOsid].price))
            self.invested = True
        elif zscore <= -2.0 and not self.invested:
            self.order(self.PEPsid, -int(100 / data[self.PEPsid].price))
            self.order(self.KOsid, int(100 / data[self.KOsid].price))
            self.invested = True
        elif abs(zscore) < .5 and self.invested:
            self.sell_spread()
            self.invested = False

    def sell_spread(self):
        """
        decrease exposure, regardless of position long/short.
        buy for a short position, sell for a long.
        """
        ko_amount = self.portfolio.positions[self.KOsid].amount
        self.order(self.KOsid, -1 * ko_amount)
        pep_amount = self.portfolio.positions[self.PEPsid].amount
        self.order(self.PEPsid, -1 * pep_amount)

if __name__ == '__main__':
    start = datetime(2000, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_from_yahoo(stocks=['PEP', 'KO'], indexes={},
                           start=start, end=end)
    source = DataFrameSource(data)

    pairtrade = Pairtrade()
    results = pairtrade.run(source)
    data['spreads'] = np.nan

    ax1 = plt.subplot(211)
    # TODO Bugged - indices are out of bounds
    # data[[pairtrade.PEPsid, pairtrade.KOsid]].plot(ax=ax1)
    plt.ylabel('price')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(212, sharex=ax1)
    results.zscores.plot(ax=ax2, color='r')
    plt.ylabel('zscored spread')

    plt.gcf().set_size_inches(18, 8)
