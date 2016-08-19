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

import functools

import logbook
import math
import numpy as np
import numpy.linalg as la

from six import iteritems

import pandas as pd

from . import risk
from . risk import (
    alpha,
    check_entry,
    downside_risk,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
)

log = logbook.Logger('Risk Period')

choose_treasury = functools.partial(risk.choose_treasury,
                                    risk.select_treasury_duration)


class RiskMetricsPeriod(object):
    def __init__(self, start_date, end_date, returns, env,
                 benchmark_returns=None, algorithm_leverages=None):

        self.env = env
        treasury_curves = env.treasury_curves
        if treasury_curves.index[-1] >= start_date:
            mask = ((treasury_curves.index >= start_date) &
                    (treasury_curves.index <= end_date))

            self.treasury_curves = treasury_curves[mask]
        else:
            # our test is beyond the treasury curve history
            # so we'll use the last available treasury curve
            self.treasury_curves = treasury_curves[-1:]

        self.start_date = start_date
        self.end_date = end_date

        if benchmark_returns is None:
            br = env.benchmark_returns
            benchmark_returns = br[(br.index >= returns.index[0]) &
                                   (br.index <= returns.index[-1])]

        self.algorithm_returns = self.mask_returns_to_period(returns,
                                                             env)
        self.benchmark_returns = self.mask_returns_to_period(benchmark_returns,
                                                             env)
        self.algorithm_leverages = algorithm_leverages

        self.calculate_metrics()

    def calculate_metrics(self):

        self.benchmark_period_returns = \
            self.calculate_period_returns(self.benchmark_returns)

        self.algorithm_period_returns = \
            self.calculate_period_returns(self.algorithm_returns)

        if not self.algorithm_returns.index.equals(
            self.benchmark_returns.index
        ):
            message = "Mismatch between benchmark_returns ({bm_count}) and \
            algorithm_returns ({algo_count}) in range {start} : {end}"
            message = message.format(
                bm_count=len(self.benchmark_returns),
                algo_count=len(self.algorithm_returns),
                start=self.start_date,
                end=self.end_date
            )
            raise Exception(message)

        self.num_trading_days = len(self.benchmark_returns)
        self.trading_day_counts = pd.stats.moments.rolling_count(
            self.algorithm_returns, self.num_trading_days)

        self.mean_algorithm_returns = \
            self.algorithm_returns.cumsum() / self.trading_day_counts

        self.benchmark_volatility = self.calculate_volatility(
            self.benchmark_returns)
        self.algorithm_volatility = self.calculate_volatility(
            self.algorithm_returns)
        self.treasury_period_return = choose_treasury(
            self.treasury_curves,
            self.start_date,
            self.end_date,
            self.env,
        )
        self.sharpe = self.calculate_sharpe()
        # The consumer currently expects a 0.0 value for sharpe in period,
        # this differs from cumulative which was np.nan.
        # When factoring out the sharpe_ratio, the different return types
        # were collapsed into `np.nan`.
        # TODO: Either fix consumer to accept `np.nan` or make the
        # `sharpe_ratio` return type configurable.
        # In the meantime, convert nan values to 0.0
        if pd.isnull(self.sharpe):
            self.sharpe = 0.0
        self.sortino = self.calculate_sortino()
        self.information = self.calculate_information()
        self.beta, self.algorithm_covariance, self.benchmark_variance, \
            self.condition_number, self.eigen_values = self.calculate_beta()
        self.alpha = self.calculate_alpha()
        self.excess_return = self.algorithm_period_returns - \
            self.treasury_period_return
        self.max_drawdown = self.calculate_max_drawdown()
        self.max_leverage = self.calculate_max_leverage()

    def to_dict(self):
        """
        Creates a dictionary representing the state of the risk report.
        Returns a dict object of the form:
        """
        period_label = self.end_date.strftime("%Y-%m")
        rval = {
            'trading_days': self.num_trading_days,
            'benchmark_volatility': self.benchmark_volatility,
            'algo_volatility': self.algorithm_volatility,
            'treasury_period_return': self.treasury_period_return,
            'algorithm_period_return': self.algorithm_period_returns,
            'benchmark_period_return': self.benchmark_period_returns,
            'sharpe': self.sharpe,
            'sortino': self.sortino,
            'information': self.information,
            'beta': self.beta,
            'alpha': self.alpha,
            'excess_return': self.excess_return,
            'max_drawdown': self.max_drawdown,
            'max_leverage': self.max_leverage,
            'period_label': period_label
        }

        return {k: None if check_entry(k, v) else v
                for k, v in iteritems(rval)}

    def __repr__(self):
        statements = []
        metrics = [
            "algorithm_period_returns",
            "benchmark_period_returns",
            "excess_return",
            "num_trading_days",
            "benchmark_volatility",
            "algorithm_volatility",
            "sharpe",
            "sortino",
            "information",
            "algorithm_covariance",
            "benchmark_variance",
            "beta",
            "alpha",
            "max_drawdown",
            "max_leverage",
            "algorithm_returns",
            "benchmark_returns",
            "condition_number",
            "eigen_values"
        ]

        for metric in metrics:
            value = getattr(self, metric)
            statements.append("{m}:{v}".format(m=metric, v=value))

        return '\n'.join(statements)

    def mask_returns_to_period(self, daily_returns, env):
        if isinstance(daily_returns, list):
            returns = pd.Series([x.returns for x in daily_returns],
                                index=[x.date for x in daily_returns])
        else:  # otherwise we're receiving an index already
            returns = daily_returns

        trade_days = env.trading_days
        trade_day_mask = returns.index.normalize().isin(trade_days)

        mask = ((returns.index >= self.start_date) &
                (returns.index <= self.end_date) & trade_day_mask)

        returns = returns[mask]
        return returns

    def calculate_period_returns(self, returns):
        period_returns = (1. + returns).prod() - 1
        return period_returns

    def calculate_volatility(self, daily_returns):
        return np.std(daily_returns, ddof=1) * math.sqrt(self.num_trading_days)

    def calculate_sharpe(self):
        """
        http://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return sharpe_ratio(self.algorithm_volatility,
                            self.algorithm_period_returns,
                            self.treasury_period_return)

    def calculate_sortino(self):
        """
        http://en.wikipedia.org/wiki/Sortino_ratio
        """
        mar = downside_risk(self.algorithm_returns,
                            self.mean_algorithm_returns,
                            self.num_trading_days)
        # Hold on to downside risk for debugging purposes.
        self.downside_risk = mar
        return sortino_ratio(self.algorithm_period_returns,
                             self.treasury_period_return,
                             mar)

    def calculate_information(self):
        """
        http://en.wikipedia.org/wiki/Information_ratio
        """
        return information_ratio(self.algorithm_returns,
                                 self.benchmark_returns)

    def calculate_beta(self):
        """

        .. math::

            \\beta_a = \\frac{\mathrm{Cov}(r_a,r_p)}{\mathrm{Var}(r_p)}

        http://en.wikipedia.org/wiki/Beta_(finance)
        """
        # it doesn't make much sense to calculate beta for less than two days,
        # so return nan.
        if len(self.algorithm_returns) < 2:
            return np.nan, np.nan, np.nan, np.nan, []

        returns_matrix = np.vstack([self.algorithm_returns,
                                    self.benchmark_returns])
        C = np.cov(returns_matrix, ddof=1)

        # If there are missing benchmark values, then we can't calculate the
        # beta.
        if not np.isfinite(C).all():
            return np.nan, np.nan, np.nan, np.nan, []

        eigen_values = la.eigvals(C)
        condition_number = max(eigen_values) / min(eigen_values)
        algorithm_covariance = C[0][1]
        benchmark_variance = C[1][1]
        beta = algorithm_covariance / benchmark_variance

        return (
            beta,
            algorithm_covariance,
            benchmark_variance,
            condition_number,
            eigen_values
        )

    def calculate_alpha(self):
        """
        http://en.wikipedia.org/wiki/Alpha_(investment)
        """
        return alpha(self.algorithm_period_returns,
                     self.treasury_period_return,
                     self.benchmark_period_returns,
                     self.beta)

    def calculate_max_drawdown(self):
        compounded_returns = []
        cur_return = 0.0
        for r in self.algorithm_returns:
            try:
                cur_return += math.log(1.0 + r)
            # this is a guard for a single day returning -100%, if returns are
            # greater than -1.0 it will throw an error because you cannot take
            # the log of a negative number
            except ValueError:
                log.debug("{cur} return, zeroing the returns".format(
                    cur=cur_return))
                cur_return = 0.0
            compounded_returns.append(cur_return)

        cur_max = None
        max_drawdown = None
        for cur in compounded_returns:
            if cur_max is None or cur > cur_max:
                cur_max = cur

            drawdown = (cur - cur_max)
            if max_drawdown is None or drawdown < max_drawdown:
                max_drawdown = drawdown

        if max_drawdown is None:
            return 0.0

        return 1.0 - math.exp(max_drawdown)

    def calculate_max_leverage(self):
        if self.algorithm_leverages is None:
            return 0.0
        else:
            return max(self.algorithm_leverages)
