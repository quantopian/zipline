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


import logbook
import math
import numpy as np
import numpy.linalg as la

import zipline.finance.trading as trading

import pandas as pd

from . risk import (
    alpha,
    check_entry,
    choose_treasury,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
)

log = logbook.Logger('Risk Cumulative')


class RiskMetricsCumulative(object):
    """
    :Usage:
        Instantiate RiskMetricsCumulative once.
        Call update() method on each dt to update the metrics.
    """

    def __init__(self, sim_params):
        self.treasury_curves = trading.environment.treasury_curves
        self.start_date = sim_params.period_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.end_date = sim_params.period_end.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        all_trading_days = trading.environment.trading_days
        mask = ((all_trading_days >= self.start_date) &
                (all_trading_days <= self.end_date))

        self.trading_days = all_trading_days[mask]
        if sim_params.period_end not in self.trading_days:
            last_day = pd.tseries.index.DatetimeIndex(
                [sim_params.period_end]
            )
            self.trading_days = self.trading_days.append(last_day)

        self.sim_params = sim_params

        if sim_params.emission_rate == 'daily':
            self.initialize_daily_indices()
        elif sim_params.emission_rate == 'minute':
            self.initialize_minute_indices(sim_params)

        self.algorithm_returns = None
        self.benchmark_returns = None

        self.compounded_log_returns = []
        self.moving_avg = []

        self.algorithm_volatility = []
        self.benchmark_volatility = []
        self.algorithm_period_returns = []
        self.benchmark_period_returns = []

        self.algorithm_covariance = None
        self.benchmark_variance = None
        self.condition_number = None
        self.eigen_values = None

        self.sharpe = []
        self.sortino = []
        self.information = []
        self.beta = []
        self.alpha = []
        self.max_drawdown = 0
        self.current_max = -np.inf
        self.excess_returns = []
        self.daily_treasury = {}

    def initialize_minute_indices(self, sim_params):
        self.algorithm_returns_cont = pd.Series(index=pd.date_range(
            sim_params.first_open, sim_params.last_close,
            freq="Min"))
        self.benchmark_returns_cont = pd.Series(index=pd.date_range(
            sim_params.first_open, sim_params.last_close,
            freq="Min"))

    def initialize_daily_indices(self):
        self.algorithm_returns_cont = pd.Series(index=self.trading_days)
        self.benchmark_returns_cont = pd.Series(index=self.trading_days)

    @property
    def last_return_date(self):
        return self.algorithm_returns.index[-1]

    def update(self, dt, algorithm_returns, benchmark_returns):
        self.algorithm_returns_cont[dt] = algorithm_returns
        self.algorithm_returns = self.algorithm_returns_cont.valid()

        self.benchmark_returns_cont[dt] = benchmark_returns
        self.benchmark_returns = self.benchmark_returns_cont.valid()

        self.num_trading_days = len(self.algorithm_returns)

        self.update_compounded_log_returns()

        self.algorithm_period_returns.append(
            self.calculate_period_returns(self.algorithm_returns))
        self.benchmark_period_returns.append(
            self.calculate_period_returns(self.benchmark_returns))

        if not self.algorithm_returns.index.equals(
            self.benchmark_returns.index
        ):
            message = "Mismatch between benchmark_returns ({bm_count}) and \
algorithm_returns ({algo_count}) in range {start} : {end} on {dt}"
            message = message.format(
                bm_count=len(self.benchmark_returns),
                algo_count=len(self.algorithm_returns),
                start=self.start_date,
                end=self.end_date,
                dt=dt
            )
            raise Exception(message)

        self.update_current_max()
        self.benchmark_volatility.append(
            self.calculate_volatility(self.benchmark_returns))
        self.algorithm_volatility.append(
            self.calculate_volatility(self.algorithm_returns))

        # caching the treasury rates for the minutely case is a
        # big speedup, because it avoids searching the treasury
        # curves on every minute.
        treasury_end = self.algorithm_returns.index[-1].replace(
            hour=0, minute=0)
        if treasury_end not in self.daily_treasury:
            treasury_period_return = choose_treasury(
                self.treasury_curves,
                self.start_date,
                self.algorithm_returns.index[-1]
            )
            self.daily_treasury[treasury_end] =\
                treasury_period_return
        self.treasury_period_return = \
            self.daily_treasury[treasury_end]
        self.excess_returns.append(
            self.algorithm_period_returns[-1] - self.treasury_period_return)
        self.beta.append(self.calculate_beta()[0])
        self.alpha.append(self.calculate_alpha())
        self.sharpe.append(self.calculate_sharpe())
        self.sortino.append(self.calculate_sortino())
        self.information.append(self.calculate_information())
        self.max_drawdown = self.calculate_max_drawdown()

    def to_dict(self):
        """
        Creates a dictionary representing the state of the risk report.
        Returns a dict object of the form:
        """
        period_label = self.last_return_date.strftime("%Y-%m")
        rval = {
            'trading_days': len(self.algorithm_returns.valid()),
            'benchmark_volatility': self.benchmark_volatility[-1],
            'algo_volatility': self.algorithm_volatility[-1],
            'treasury_period_return': self.treasury_period_return,
            'algorithm_period_return': self.algorithm_period_returns[-1],
            'benchmark_period_return': self.benchmark_period_returns[-1],
            'beta': self.beta[-1],
            'alpha': self.alpha[-1],
            'excess_return': self.excess_returns[-1],
            'max_drawdown': self.max_drawdown,
            'period_label': period_label
        }

        rval['sharpe'] = self.sharpe[-1]
        rval['sortino'] = self.sortino[-1]
        rval['information'] = self.information[-1]

        return {k: None
                if check_entry(k, v)
                else v for k, v in rval.iteritems()}

    def __repr__(self):
        statements = []
        metrics = [
            "algorithm_period_returns",
            "benchmark_period_returns",
            "excess_returns",
            "trading_days",
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
            "algorithm_returns",
            "benchmark_returns",
            "condition_number",
            "eigen_values"
        ]

        for metric in metrics:
            value = getattr(self, metric)
            if isinstance(value, list):
                if len(value) == 0:
                    value = np.nan
                else:
                    value = value[-1]
            statements.append("{m}:{v}".format(m=metric, v=value))

        return '\n'.join(statements)

    def update_compounded_log_returns(self):
        if len(self.algorithm_returns) == 0:
            return

        try:
            compound = math.log(1 + self.algorithm_returns[
                self.algorithm_returns.last_valid_index()])
        except ValueError:
            compound = 0.0
            # BUG? Shouldn't this be set to log(1.0 + 0) ?

        if len(self.compounded_log_returns) == 0:
            self.compounded_log_returns.append(compound)
        else:
            self.compounded_log_returns.append(
                self.compounded_log_returns[-1] +
                compound
            )

    def calculate_period_returns(self, returns):
        returns = np.array(returns)
        return (1. + returns).prod() - 1

    def update_current_max(self):
        if len(self.compounded_log_returns) == 0:
            return
        if self.current_max < self.compounded_log_returns[-1]:
            self.current_max = self.compounded_log_returns[-1]

    def calculate_max_drawdown(self):
        if len(self.compounded_log_returns) == 0:
            return self.max_drawdown

        cur_drawdown = 1.0 - math.exp(
            self.compounded_log_returns[-1] -
            self.current_max)

        if self.max_drawdown < cur_drawdown:
            return cur_drawdown
        else:
            return self.max_drawdown

    def calculate_sharpe(self):
        """
        http://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return sharpe_ratio(self.algorithm_volatility[-1],
                            self.algorithm_period_returns[-1],
                            self.treasury_period_return)

    def calculate_sortino(self, mar=None):
        """
        http://en.wikipedia.org/wiki/Sortino_ratio
        """
        if mar is None:
            mar = self.treasury_period_return

        return sortino_ratio(np.array(self.algorithm_returns),
                             self.algorithm_period_returns[-1],
                             mar)

    def calculate_information(self):
        """
        http://en.wikipedia.org/wiki/Information_ratio
        """
        A = np.array
        return information_ratio(A(self.algorithm_returns),
                                 A(self.benchmark_returns))

    def calculate_alpha(self):
        """
        http://en.wikipedia.org/wiki/Alpha_(investment)
        """
        return alpha(self.algorithm_period_returns[-1],
                     self.treasury_period_return,
                     self.benchmark_period_returns[-1],
                     self.beta[-1])

    def calculate_volatility(self, daily_returns):
        return np.std(daily_returns, ddof=1) * math.sqrt(self.num_trading_days)

    def calculate_beta(self):
        """

        .. math::

            \\beta_a = \\frac{\mathrm{Cov}(r_a,r_p)}{\mathrm{Var}(r_p)}

        http://en.wikipedia.org/wiki/Beta_(finance)
        """
        # it doesn't make much sense to calculate beta for less than two days,
        # so return none.
        if len(self.algorithm_returns) < 2:
            return 0.0, 0.0, 0.0, 0.0, []

        returns_matrix = np.vstack([self.algorithm_returns,
                                    self.benchmark_returns])
        C = np.cov(returns_matrix, ddof=1)
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
