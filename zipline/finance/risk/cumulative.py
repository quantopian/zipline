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
import msgpack
import numpy as np

from zipline.finance import trading
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

    return (
        (algorithm_return - benchmark_return)
        # The square of the annualization factor is in the volatility,
        # because the volatility is also annualized,
        # i.e. the sqrt(annual factor) is in the volatility's numerator.
        # So to have the the correct annualization factor for the
        # Sharpe value's numerator, which should be the sqrt(annual factor).
        # The square of the sqrt of the annual factor, i.e. the annual factor
        # itself, is needed in the numerator to factor out the division by
        # its square root.
        / algo_volatility)


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

    def __init__(self, sim_params,
                 returns_frequency=None,
                 create_first_day_stats=False):
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

        self.trading_days = trading.environment.days_in_range(
            self.start_date,
            self.end_date)

        # Hold on to the trading day before the start,
        # used for index of the zero return value when forcing returns
        # on the first day.
        self.day_before_start = self.start_date - \
            trading.environment.trading_days.freq

        last_day = normalize_date(sim_params.period_end)
        if last_day not in self.trading_days:
            last_day = pd.tseries.index.DatetimeIndex(
                [last_day]
            )
            self.trading_days = self.trading_days.append(last_day)

        self.sim_params = sim_params

        self.create_first_day_stats = create_first_day_stats

        if returns_frequency is None:
            returns_frequency = self.sim_params.emission_rate

        self.returns_frequency = returns_frequency

        if returns_frequency == 'daily':
            cont_index = self.get_daily_index()
        elif returns_frequency == 'minute':
            cont_index = self.get_minute_index(sim_params)

        self.cont_index = cont_index

        self.algorithm_returns_cont = pd.Series(index=cont_index)
        self.benchmark_returns_cont = pd.Series(index=cont_index)
        self.mean_returns_cont = pd.Series(index=cont_index)
        self.annualized_mean_returns_cont = pd.Series(index=cont_index)
        self.mean_benchmark_returns_cont = pd.Series(index=cont_index)
        self.annualized_mean_benchmark_returns_cont = pd.Series(
            index=cont_index)

        # The returns at a given time are read and reset from the respective
        # returns container.
        self.algorithm_returns = None
        self.benchmark_returns = None
        self.mean_returns = None
        self.annualized_mean_returns = None
        self.mean_benchmark_returns = None
        self.annualized_mean_benchmark_returns = None

        self.algorithm_cumulative_returns = pd.Series(index=cont_index)
        self.benchmark_cumulative_returns = pd.Series(index=cont_index)
        self.excess_returns = pd.Series(index=cont_index)

        self.latest_dt = cont_index[0]

        self.metrics = pd.DataFrame(index=cont_index,
                                    columns=self.METRIC_NAMES)

        self.drawdowns = pd.Series(index=cont_index)
        self.max_drawdowns = pd.Series(index=cont_index)
        self.max_drawdown = 0
        self.current_max = -np.inf
        self.daily_treasury = pd.Series(index=self.trading_days)
        self.treasury_period_return = np.nan

        self.num_trading_days = 0

    def get_minute_index(self, sim_params):
        """
        Stitches together multiple days worth of business minutes into
        one continous index.
        """
        trading_minutes = None
        for day in self.trading_days:
            minutes_for_day = trading.environment.market_minutes_for_day(day)
            if trading_minutes is None:
                # Create container for all minutes on first iteration
                trading_minutes = minutes_for_day
            else:
                trading_minutes = trading_minutes + minutes_for_day
        return trading_minutes

    def get_daily_index(self):
        return self.trading_days

    def update(self, dt, algorithm_returns, benchmark_returns):
        # Keep track of latest dt for use in to_dict and other methods
        # that report current state.
        self.latest_dt = dt

        self.algorithm_returns_cont[dt] = algorithm_returns
        self.algorithm_returns = self.algorithm_returns_cont[:dt]

        self.num_trading_days = len(self.algorithm_returns)

        if self.create_first_day_stats:
            if len(self.algorithm_returns) == 1:
                self.algorithm_returns = pd.Series(
                    {self.day_before_start: 0.0}).append(
                    self.algorithm_returns)

        self.algorithm_cumulative_returns[dt] = \
            self.calculate_cumulative_returns(self.algorithm_returns)

        algo_cumulative_returns_to_date = \
            self.algorithm_cumulative_returns[:dt]

        self.mean_returns_cont[dt] = \
            algo_cumulative_returns_to_date[dt] / self.num_trading_days

        self.mean_returns = self.mean_returns_cont[:dt]

        self.annualized_mean_returns_cont[dt] = \
            self.mean_returns_cont[dt] * 252

        self.annualized_mean_returns = self.annualized_mean_returns_cont[:dt]

        if self.create_first_day_stats:
            if len(self.mean_returns) == 1:
                self.mean_returns = pd.Series(
                    {self.day_before_start: 0.0}).append(self.mean_returns)
                self.annualized_mean_returns = pd.Series(
                    {self.day_before_start: 0.0}).append(
                    self.annualized_mean_returns)

        self.benchmark_returns_cont[dt] = benchmark_returns
        self.benchmark_returns = self.benchmark_returns_cont[:dt]

        if self.create_first_day_stats:
            if len(self.benchmark_returns) == 1:
                self.benchmark_returns = pd.Series(
                    {self.day_before_start: 0.0}).append(
                    self.benchmark_returns)

        self.benchmark_cumulative_returns[dt] = \
            self.calculate_cumulative_returns(self.benchmark_returns)

        benchmark_cumulative_returns_to_date = \
            self.benchmark_cumulative_returns[:dt]

        self.mean_benchmark_returns_cont[dt] = \
            benchmark_cumulative_returns_to_date[dt] / self.num_trading_days

        self.mean_benchmark_returns = self.mean_benchmark_returns_cont[:dt]

        self.annualized_mean_benchmark_returns_cont[dt] = \
            self.mean_benchmark_returns_cont[dt] * 252

        self.annualized_mean_benchmark_returns = \
            self.annualized_mean_benchmark_returns_cont[:dt]

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
        # In both minutely and daily, the daily curve is always used.
        treasury_end = dt.replace(hour=0, minute=0)
        if np.isnan(self.daily_treasury[treasury_end]):
            treasury_period_return = choose_treasury(
                self.treasury_curves,
                self.start_date,
                treasury_end
            )
            self.daily_treasury[treasury_end] = treasury_period_return
        self.treasury_period_return = self.daily_treasury[treasury_end]
        self.excess_returns[self.latest_dt] = (
            self.algorithm_cumulative_returns[self.latest_dt]
            -
            self.treasury_period_return)
        self.metrics.beta[dt] = self.calculate_beta()
        self.metrics.alpha[dt] = self.calculate_alpha()
        self.metrics.sharpe[dt] = self.calculate_sharpe()
        self.metrics.downside_risk[dt] = self.calculate_downside_risk()
        self.metrics.sortino[dt] = self.calculate_sortino()
        self.metrics.information[dt] = self.calculate_information()
        self.max_drawdown = self.calculate_max_drawdown()
        self.max_drawdowns[dt] = self.max_drawdown

    def to_dict(self):
        """
        Creates a dictionary representing the state of the risk report.
        Returns a dict object of the form:
        """
        dt = self.latest_dt
        period_label = dt.strftime("%Y-%m")
        rval = {
            'trading_days': self.num_trading_days,
            'benchmark_volatility': self.metrics.benchmark_volatility[dt],
            'algo_volatility': self.metrics.algorithm_volatility[dt],
            'treasury_period_return': self.treasury_period_return,
            # Though the two following keys say period return,
            # they would be more accurately called the cumulative return.
            # However, the keys need to stay the same, for now, for backwards
            # compatibility with existing consumers.
            'algorithm_period_return': self.algorithm_cumulative_returns[dt],
            'benchmark_period_return': self.benchmark_cumulative_returns[dt],
            'beta': self.metrics.beta[dt],
            'alpha': self.metrics.alpha[dt],
            'sharpe': self.metrics.sharpe[dt],
            'sortino': self.metrics.sortino[dt],
            'information': self.metrics.information[dt],
            'excess_return': self.excess_returns[dt],
            'max_drawdown': self.max_drawdown,
            'period_label': period_label
        }

        return {k: (None if check_entry(k, v) else v)
                for k, v in iteritems(rval)}

    def _get_state(self):
        """
        Uses msgpack to create a serialized version of the object.
        """

        # Go through and call any custom serialization methods we've added
        state_dict = {}
        for k, v in self.__dict__.iteritems():
            if (not k.startswith('_')) and (not k == 'treasury_curves'):
                state_dict[k] = v
        
        return 'RiskMetricsCumulative', state_dict

    def _set_state(self, saved_state):
        self.__dict__.update(saved_state)

        # This are big and we don't need to serialize them
        # pop them back in now
        self.treasury_curves = trading.environment.treasury_curves

    def __repr__(self):
        statements = []
        for metric in self.METRIC_NAMES:
            value = getattr(self.metrics, metric)[-1]
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
            self.algorithm_cumulative_returns[self.latest_dt]
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
            (1.0 + self.algorithm_cumulative_returns[self.latest_dt])
            /
            (1.0 + self.current_max))

        self.drawdowns[self.latest_dt] = cur_drawdown

        if self.max_drawdown < cur_drawdown:
            return cur_drawdown
        else:
            return self.max_drawdown

    def calculate_sharpe(self):
        """
        http://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return sharpe_ratio(self.metrics.algorithm_volatility[self.latest_dt],
                            self.annualized_mean_returns[self.latest_dt],
                            self.daily_treasury[self.latest_dt.date()])

    def calculate_sortino(self):
        """
        http://en.wikipedia.org/wiki/Sortino_ratio
        """
        return sortino_ratio(self.annualized_mean_returns[self.latest_dt],
                             self.daily_treasury[self.latest_dt.date()],
                             self.metrics.downside_risk[self.latest_dt])

    def calculate_information(self):
        """
        http://en.wikipedia.org/wiki/Information_ratio
        """
        return information_ratio(
            self.metrics.algorithm_volatility[self.latest_dt],
            self.annualized_mean_returns[self.latest_dt],
            self.annualized_mean_benchmark_returns[self.latest_dt])

    def calculate_alpha(self):
        """
        http://en.wikipedia.org/wiki/Alpha_(investment)
        """
        return alpha(self.annualized_mean_returns[self.latest_dt],
                     self.treasury_period_return,
                     self.annualized_mean_benchmark_returns[self.latest_dt],
                     self.metrics.beta[self.latest_dt])

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
        # it doesn't make much sense to calculate beta for less than two days,
        # so return none.
        if len(self.annualized_mean_returns) < 2:
            return 0.0

        returns_matrix = np.vstack([self.algorithm_returns,
                                    self.benchmark_returns])
        C = np.cov(returns_matrix, ddof=1)
        algorithm_covariance = C[0][1]
        benchmark_variance = C[1][1]
        beta = algorithm_covariance / benchmark_variance

        return beta
