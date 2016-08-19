#
# Copyright 2014 Quantopian, Inc.
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

import zipline.utils.math_utils as zp_math

import pandas as pd
from pandas.tseries.tools import normalize_date

from six import iteritems

from . risk import (
    alpha,
    check_entry,
    choose_treasury,
    downside_risk,
    sharpe_ratio,
    sortino_ratio,
)

log = logbook.Logger('Risk Cumulative')


choose_treasury = functools.partial(choose_treasury, lambda *args: '10year',
                                    compound=False)


def information_ratio(algo_volatility, algorithm_return, benchmark_return):
    """
    http://en.wikipedia.org/wiki/Information_ratio

    Args:
        algorithm_returns (np.array-like):
            All returns during algorithm lifetime.
        benchmark_returns (np.array-like):
            All benchmark returns during algo lifetime.

    Returns:
        float. Information ratio.
    """
    if zp_math.tolerant_equals(algo_volatility, 0):
        return np.nan

    # The square of the annualization factor is in the volatility,
    # because the volatility is also annualized,
    # i.e. the sqrt(annual factor) is in the volatility's numerator.
    # So to have the the correct annualization factor for the
    # Sharpe value's numerator, which should be the sqrt(annual factor).
    # The square of the sqrt of the annual factor, i.e. the annual factor
    # itself, is needed in the numerator to factor out the division by
    # its square root.
    return (algorithm_return - benchmark_return) / algo_volatility


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
        'downside_risk',
        'sortino',
        'information',
    )

    def __init__(self, sim_params, env, create_first_day_stats=False):
        self.treasury_curves = env.treasury_curves
        self.start_date = sim_params.period_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.end_date = sim_params.period_end.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        self.trading_days = env.days_in_range(self.start_date, self.end_date)

        # Hold on to the trading day before the start,
        # used for index of the zero return value when forcing returns
        # on the first day.
        self.day_before_start = self.start_date - env.trading_days.freq

        last_day = normalize_date(sim_params.period_end)
        if last_day not in self.trading_days:
            last_day = pd.tseries.index.DatetimeIndex(
                [last_day]
            )
            self.trading_days = self.trading_days.append(last_day)

        self.sim_params = sim_params
        self.env = env

        self.create_first_day_stats = create_first_day_stats

        cont_index = self.trading_days

        self.cont_index = cont_index
        self.cont_len = len(self.cont_index)

        empty_cont = np.full(self.cont_len, np.nan)

        self.algorithm_returns_cont = empty_cont.copy()
        self.benchmark_returns_cont = empty_cont.copy()
        self.algorithm_cumulative_leverages_cont = empty_cont.copy()
        self.mean_returns_cont = empty_cont.copy()
        self.annualized_mean_returns_cont = empty_cont.copy()
        self.mean_benchmark_returns_cont = empty_cont.copy()
        self.annualized_mean_benchmark_returns_cont = empty_cont.copy()

        # The returns at a given time are read and reset from the respective
        # returns container.
        self.algorithm_returns = None
        self.benchmark_returns = None
        self.mean_returns = None
        self.annualized_mean_returns = None
        self.mean_benchmark_returns = None
        self.annualized_mean_benchmark_returns = None

        self.algorithm_cumulative_returns = empty_cont.copy()
        self.benchmark_cumulative_returns = empty_cont.copy()
        self.algorithm_cumulative_leverages = empty_cont.copy()
        self.excess_returns = empty_cont.copy()

        self.latest_dt_loc = 0
        self.latest_dt = cont_index[0]

        self.benchmark_volatility = empty_cont.copy()
        self.algorithm_volatility = empty_cont.copy()
        self.beta = empty_cont.copy()
        self.alpha = empty_cont.copy()
        self.sharpe = empty_cont.copy()
        self.downside_risk = empty_cont.copy()
        self.sortino = empty_cont.copy()
        self.information = empty_cont.copy()

        self.drawdowns = empty_cont.copy()
        self.max_drawdowns = empty_cont.copy()
        self.max_drawdown = 0
        self.max_leverages = empty_cont.copy()
        self.max_leverage = 0
        self.current_max = -np.inf
        self.daily_treasury = pd.Series(index=self.trading_days)
        self.treasury_period_return = np.nan

        self.num_trading_days = 0

    def update(self, dt, algorithm_returns, benchmark_returns, leverage):
        # Keep track of latest dt for use in to_dict and other methods
        # that report current state.
        self.latest_dt = dt
        dt_loc = self.cont_index.get_loc(dt)
        self.latest_dt_loc = dt_loc

        self.algorithm_returns_cont[dt_loc] = algorithm_returns
        self.algorithm_returns = self.algorithm_returns_cont[:dt_loc + 1]

        self.num_trading_days = len(self.algorithm_returns)

        if self.create_first_day_stats:
            if len(self.algorithm_returns) == 1:
                self.algorithm_returns = np.append(0.0, self.algorithm_returns)

        self.algorithm_cumulative_returns[dt_loc] = \
            self.calculate_cumulative_returns(self.algorithm_returns)

        algo_cumulative_returns_to_date = \
            self.algorithm_cumulative_returns[:dt_loc + 1]

        self.mean_returns_cont[dt_loc] = \
            algo_cumulative_returns_to_date[dt_loc] / self.num_trading_days

        self.mean_returns = self.mean_returns_cont[:dt_loc + 1]

        self.annualized_mean_returns_cont[dt_loc] = \
            self.mean_returns_cont[dt_loc] * 252

        self.annualized_mean_returns = \
            self.annualized_mean_returns_cont[:dt_loc + 1]

        if self.create_first_day_stats:
            if len(self.mean_returns) == 1:
                self.mean_returns = np.append(0.0, self.mean_returns)
                self.annualized_mean_returns = np.append(
                    0.0, self.annualized_mean_returns)

        self.benchmark_returns_cont[dt_loc] = benchmark_returns
        self.benchmark_returns = self.benchmark_returns_cont[:dt_loc + 1]

        if self.create_first_day_stats:
            if len(self.benchmark_returns) == 1:
                self.benchmark_returns = np.append(0.0, self.benchmark_returns)

        self.benchmark_cumulative_returns[dt_loc] = \
            self.calculate_cumulative_returns(self.benchmark_returns)

        benchmark_cumulative_returns_to_date = \
            self.benchmark_cumulative_returns[:dt_loc + 1]

        self.mean_benchmark_returns_cont[dt_loc] = \
            benchmark_cumulative_returns_to_date[dt_loc] / \
            self.num_trading_days

        self.mean_benchmark_returns = self.mean_benchmark_returns_cont[:dt_loc]

        self.annualized_mean_benchmark_returns_cont[dt_loc] = \
            self.mean_benchmark_returns_cont[dt_loc] * 252

        self.annualized_mean_benchmark_returns = \
            self.annualized_mean_benchmark_returns_cont[:dt_loc + 1]

        self.algorithm_cumulative_leverages_cont[dt_loc] = leverage
        self.algorithm_cumulative_leverages = \
            self.algorithm_cumulative_leverages_cont[:dt_loc + 1]

        if self.create_first_day_stats:
            if len(self.algorithm_cumulative_leverages) == 1:
                self.algorithm_cumulative_leverages = np.append(
                    0.0,
                    self.algorithm_cumulative_leverages)

        if not len(self.algorithm_returns) and len(self.benchmark_returns):
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
        self.benchmark_volatility[dt_loc] = \
            self.calculate_volatility(self.benchmark_returns)
        self.algorithm_volatility[dt_loc] = \
            self.calculate_volatility(self.algorithm_returns)

        # caching the treasury rates for the minutely case is a
        # big speedup, because it avoids searching the treasury
        # curves on every minute.
        # In both minutely and daily, the daily curve is always used.
        treasury_end = dt.replace(hour=0, minute=0)
        if np.isnan(self.daily_treasury[treasury_end]):
            treasury_period_return = choose_treasury(
                self.treasury_curves,
                self.start_date,
                treasury_end,
                self.env,
            )
            self.daily_treasury[treasury_end] = treasury_period_return
        self.treasury_period_return = self.daily_treasury[treasury_end]
        self.excess_returns[dt_loc] = (
            self.algorithm_cumulative_returns[dt_loc] -
            self.treasury_period_return)
        self.beta[dt_loc] = self.calculate_beta()
        self.alpha[dt_loc] = self.calculate_alpha()
        self.sharpe[dt_loc] = self.calculate_sharpe()
        self.downside_risk[dt_loc] = \
            self.calculate_downside_risk()
        self.sortino[dt_loc] = self.calculate_sortino()
        self.information[dt_loc] = self.calculate_information()
        self.max_drawdown = self.calculate_max_drawdown()
        self.max_drawdowns[dt_loc] = self.max_drawdown
        self.max_leverage = self.calculate_max_leverage()
        self.max_leverages[dt_loc] = self.max_leverage

    def to_dict(self):
        """
        Creates a dictionary representing the state of the risk report.
        Returns a dict object of the form:
        """
        dt = self.latest_dt
        dt_loc = self.latest_dt_loc
        period_label = dt.strftime("%Y-%m")
        rval = {
            'trading_days': self.num_trading_days,
            'benchmark_volatility':
            self.benchmark_volatility[dt_loc],
            'algo_volatility':
            self.algorithm_volatility[dt_loc],
            'treasury_period_return': self.treasury_period_return,
            # Though the two following keys say period return,
            # they would be more accurately called the cumulative return.
            # However, the keys need to stay the same, for now, for backwards
            # compatibility with existing consumers.
            'algorithm_period_return':
            self.algorithm_cumulative_returns[dt_loc],
            'benchmark_period_return':
            self.benchmark_cumulative_returns[dt_loc],
            'beta': self.beta[dt_loc],
            'alpha': self.alpha[dt_loc],
            'sharpe': self.sharpe[dt_loc],
            'sortino': self.sortino[dt_loc],
            'information': self.information[dt_loc],
            'excess_return': self.excess_returns[dt_loc],
            'max_drawdown': self.max_drawdown,
            'max_leverage': self.max_leverage,
            'period_label': period_label
        }

        return {k: (None if check_entry(k, v) else v)
                for k, v in iteritems(rval)}

    def __repr__(self):
        statements = []
        for metric in self.METRIC_NAMES:
            value = getattr(self, metric)[-1]
            if isinstance(value, list):
                if len(value) == 0:
                    value = np.nan
                else:
                    value = value[-1]
            statements.append("{m}:{v}".format(m=metric, v=value))

        return '\n'.join(statements)

    def calculate_cumulative_returns(self, returns):
        return (1. + returns).prod() - 1

    def update_current_max(self):
        if len(self.algorithm_cumulative_returns) == 0:
            return
        current_cumulative_return = \
            self.algorithm_cumulative_returns[self.latest_dt_loc]
        if self.current_max < current_cumulative_return:
            self.current_max = current_cumulative_return

    def calculate_max_drawdown(self):
        if len(self.algorithm_cumulative_returns) == 0:
            return self.max_drawdown

        # The drawdown is defined as: (high - low) / high
        # The above factors out to: 1.0 - (low / high)
        #
        # Instead of explicitly always using the low, use the current total
        # return value, and test that against the max drawdown, which will
        # exceed the previous max_drawdown iff the current return is lower than
        # the previous low in the current drawdown window.
        cur_drawdown = 1.0 - (
            (1.0 + self.algorithm_cumulative_returns[self.latest_dt_loc])
            /
            (1.0 + self.current_max))

        self.drawdowns[self.latest_dt_loc] = cur_drawdown

        if self.max_drawdown < cur_drawdown:
            return cur_drawdown
        else:
            return self.max_drawdown

    def calculate_max_leverage(self):
        # The leverage is defined as: the gross_exposure/net_liquidation
        # gross_exposure = long_exposure + abs(short_exposure)
        # net_liquidation = ending_cash + long_exposure + short_exposure
        cur_leverage = self.algorithm_cumulative_leverages_cont[
            self.latest_dt_loc]

        return max(cur_leverage, self.max_leverage)

    def calculate_sharpe(self):
        """
        http://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return sharpe_ratio(
            self.algorithm_volatility[self.latest_dt_loc],
            self.annualized_mean_returns_cont[self.latest_dt_loc],
            self.daily_treasury[self.latest_dt.date()])

    def calculate_sortino(self):
        """
        http://en.wikipedia.org/wiki/Sortino_ratio
        """
        return sortino_ratio(
            self.annualized_mean_returns_cont[self.latest_dt_loc],
            self.daily_treasury[self.latest_dt.date()],
            self.downside_risk[self.latest_dt_loc])

    def calculate_information(self):
        """
        http://en.wikipedia.org/wiki/Information_ratio
        """
        return information_ratio(
            self.algorithm_volatility[self.latest_dt_loc],
            self.annualized_mean_returns_cont[self.latest_dt_loc],
            self.annualized_mean_benchmark_returns_cont[self.latest_dt_loc])

    def calculate_alpha(self):
        """
        http://en.wikipedia.org/wiki/Alpha_(investment)
        """
        return alpha(
            self.annualized_mean_returns_cont[self.latest_dt_loc],
            self.treasury_period_return,
            self.annualized_mean_benchmark_returns_cont[self.latest_dt_loc],
            self.beta[self.latest_dt_loc])

    def calculate_volatility(self, daily_returns):
        if len(daily_returns) <= 1:
            return 0.0
        return np.std(daily_returns, ddof=1) * math.sqrt(252)

    def calculate_downside_risk(self):
        return downside_risk(self.algorithm_returns,
                             self.mean_returns,
                             252)

    def calculate_beta(self):
        """

        .. math::

            \\beta_a = \\frac{\mathrm{Cov}(r_a,r_p)}{\mathrm{Var}(r_p)}

        http://en.wikipedia.org/wiki/Beta_(finance)
        """
        # it doesn't make much sense to calculate beta for less than two
        # values, so return none.
        if len(self.algorithm_returns) < 2:
            return 0.0

        returns_matrix = np.vstack([self.algorithm_returns,
                                    self.benchmark_returns])
        C = np.cov(returns_matrix, ddof=1)
        algorithm_covariance = C[0][1]
        benchmark_variance = C[1][1]
        beta = algorithm_covariance / benchmark_variance

        return beta
