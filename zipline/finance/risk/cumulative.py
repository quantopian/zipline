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

    METRIC_NAMES = (
        'alpha',
        'beta',
        'sharpe',
        'algorithm_volatility',
        'benchmark_volatility',
        'sortino',
        'information',
    )

    def __init__(self, sim_params, returns_frequency=None):
        """
        - @returns_frequency allows for configuration of the whether
        the benchmark and algorithm returns are in units of minutes or days,
        if `None` defaults to the `emission_rate` in `sim_params`.
        """

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

        if returns_frequency is None:
            returns_frequency = self.sim_params.emission_rate

        if returns_frequency == 'daily':
            cont_index = self.get_daily_index()
        elif returns_frequency == 'minute':
            cont_index = self.get_minute_index(sim_params)

        self.algorithm_returns_cont = pd.Series(index=cont_index)
        self.benchmark_returns_cont = pd.Series(index=cont_index)

        # The returns at a given time are read and reset from the respective
        # returns container.
        self.algorithm_returns = None
        self.benchmark_returns = None

        self.compounded_log_returns = pd.Series(index=cont_index)
        self.algorithm_period_returns = pd.Series(index=cont_index)
        self.benchmark_period_returns = pd.Series(index=cont_index)
        self.excess_returns = pd.Series(index=cont_index)

        self.latest_dt = cont_index[0]

        self.metrics = pd.DataFrame(index=cont_index,
                                    columns=self.METRIC_NAMES)

        self.max_drawdown = 0
        self.current_max = -np.inf
        self.daily_treasury = {}

    def get_minute_index(self, sim_params):
        return pd.date_range(sim_params.first_open, sim_params.last_close,
                             freq="Min")

    def get_daily_index(self):
        return self.trading_days

    @property
    def last_return_date(self):
        return self.algorithm_returns.index[-1]

    def update(self, dt, algorithm_returns, benchmark_returns):
        # Keep track of latest dt for use in to_dict and other methods
        # that report current state.
        self.latest_dt = dt

        self.algorithm_returns_cont[dt] = algorithm_returns
        self.algorithm_returns = self.algorithm_returns_cont.valid()

        self.benchmark_returns_cont[dt] = benchmark_returns
        self.benchmark_returns = self.benchmark_returns_cont.valid()

        self.num_trading_days = len(self.algorithm_returns)

        self.update_compounded_log_returns()

        self.algorithm_period_returns[dt] = \
            self.calculate_period_returns(self.algorithm_returns)
        self.benchmark_period_returns[dt] = \
            self.calculate_period_returns(self.benchmark_returns)

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
        self.metrics.benchmark_volatility[dt] = \
            self.calculate_volatility(self.benchmark_returns)
        self.metrics.algorithm_volatility[dt] = \
            self.calculate_volatility(self.algorithm_returns)

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
        self.excess_returns[self.latest_dt] = (
            self.algorithm_period_returns[self.latest_dt]
            -
            self.treasury_period_return)
        self.metrics.beta[dt] = self.calculate_beta()
        self.metrics.alpha[dt] = self.calculate_alpha(dt)
        self.metrics.sharpe[dt] = self.calculate_sharpe()
        self.metrics.sortino[dt] = self.calculate_sortino()
        self.metrics.information[dt] = self.calculate_information()
        self.max_drawdown = self.calculate_max_drawdown()

    def to_dict(self):
        """
        Creates a dictionary representing the state of the risk report.
        Returns a dict object of the form:
        """
        period_label = self.last_return_date.strftime("%Y-%m")
        dt = self.latest_dt
        rval = {
            'trading_days': len(self.algorithm_returns.valid()),
            'benchmark_volatility':
            self.metrics.benchmark_volatility[dt],
            'algo_volatility':
            self.metrics.algorithm_volatility[dt],
            'treasury_period_return': self.treasury_period_return,
            'algorithm_period_return': self.algorithm_period_returns[dt],
            'benchmark_period_return': self.benchmark_period_returns[dt],
            'beta': self.metrics.beta[dt],
            'alpha': self.metrics.alpha[dt],
            'excess_return': self.excess_returns[dt],
            'max_drawdown': self.max_drawdown,
            'period_label': period_label
        }

        rval['sharpe'] = self.metrics.sharpe[dt]
        rval['sortino'] = self.metrics.sortino[dt]
        rval['information'] = self.metrics.information[dt]

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
            "beta",
            "alpha",
            "max_drawdown",
            "algorithm_returns",
            "benchmark_returns",
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

        if len(self.compounded_log_returns[:self.latest_dt]) == 0:
            self.compounded_log_returns[self.latest_dt] = compound
        else:
            self.compounded_log_returns[self.latest_dt] = \
                self.compounded_log_returns[self.latest_dt] + compound

    def calculate_period_returns(self, returns):
        return (1. + returns).prod() - 1

    def update_current_max(self):
        if len(self.compounded_log_returns) == 0:
            return
        if self.current_max < self.compounded_log_returns[self.latest_dt]:
            self.current_max = self.compounded_log_returns[self.latest_dt]

    def calculate_max_drawdown(self):
        if len(self.compounded_log_returns) == 0:
            return self.max_drawdown

        cur_drawdown = 1.0 - math.exp(
            self.compounded_log_returns[self.latest_dt] -
            self.current_max)

        if self.max_drawdown < cur_drawdown:
            return cur_drawdown
        else:
            return self.max_drawdown

    def calculate_sharpe(self):
        """
        http://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return sharpe_ratio(self.metrics.algorithm_volatility[self.latest_dt],
                            self.algorithm_period_returns[self.latest_dt],
                            self.treasury_period_return)

    def calculate_sortino(self, mar=None):
        """
        http://en.wikipedia.org/wiki/Sortino_ratio
        """
        if mar is None:
            mar = self.treasury_period_return

        return sortino_ratio(np.array(self.algorithm_returns),
                             self.algorithm_period_returns[self.latest_dt],
                             mar)

    def calculate_information(self):
        """
        http://en.wikipedia.org/wiki/Information_ratio
        """
        A = np.array
        return information_ratio(A(self.algorithm_returns),
                                 A(self.benchmark_returns))

    def calculate_alpha(self, dt):
        """
        http://en.wikipedia.org/wiki/Alpha_(investment)
        """
        return alpha(self.algorithm_period_returns[self.latest_dt],
                     self.treasury_period_return,
                     self.benchmark_period_returns[self.latest_dt],
                     self.metrics.beta[dt])

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
            return 0.0

        returns_matrix = np.vstack([self.algorithm_returns,
                                    self.benchmark_returns])
        C = np.cov(returns_matrix, ddof=1)
        algorithm_covariance = C[0][1]
        benchmark_variance = C[1][1]
        beta = algorithm_covariance / benchmark_variance

        return beta
