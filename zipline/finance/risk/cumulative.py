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
import numpy as np

import pandas as pd
from pandas.tseries.tools import normalize_date

from six import iteritems

from . risk import (
    check_entry,
    choose_treasury
)

from empyrical import (
    alpha_beta_aligned,
    annual_volatility,
    cum_returns,
    downside_risk,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

log = logbook.Logger('Risk Cumulative')


choose_treasury = functools.partial(choose_treasury, lambda *args: '10year',
                                    compound=False)


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

    def __init__(self, sim_params, treasury_curves, trading_calendar,
                 create_first_day_stats=False):
        self.treasury_curves = treasury_curves
        self.trading_calendar = trading_calendar
        self.start_session = sim_params.start_session
        self.end_session = sim_params.end_session

        self.sessions = trading_calendar.sessions_in_range(
            self.start_session, self.end_session
        )

        # Hold on to the trading day before the start,
        # used for index of the zero return value when forcing returns
        # on the first day.
        self.day_before_start = self.start_session - self.sessions.freq

        last_day = normalize_date(sim_params.end_session)
        if last_day not in self.sessions:
            last_day = pd.tseries.index.DatetimeIndex(
                [last_day]
            )
            self.sessions = self.sessions.append(last_day)

        self.sim_params = sim_params

        self.create_first_day_stats = create_first_day_stats

        cont_index = self.sessions

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
        self.daily_treasury = pd.Series(index=self.sessions)
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

        self.algorithm_cumulative_returns[dt_loc] = cum_returns(
            self.algorithm_returns
        )[-1]

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

        self.benchmark_cumulative_returns[dt_loc] = cum_returns(
            self.benchmark_returns
        )[-1]

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
                start=self.start_session,
                end=self.end_session,
                dt=dt
            )
            raise Exception(message)

        self.update_current_max()
        self.benchmark_volatility[dt_loc] = annual_volatility(
            self.benchmark_returns
        )
        self.algorithm_volatility[dt_loc] = annual_volatility(
            self.algorithm_returns
        )

        # caching the treasury rates for the minutely case is a
        # big speedup, because it avoids searching the treasury
        # curves on every minute.
        # In both minutely and daily, the daily curve is always used.
        treasury_end = dt.replace(hour=0, minute=0)
        if np.isnan(self.daily_treasury[treasury_end]):
            treasury_period_return = choose_treasury(
                self.treasury_curves,
                self.start_session,
                treasury_end,
                self.trading_calendar,
            )
            self.daily_treasury[treasury_end] = treasury_period_return
        self.treasury_period_return = self.daily_treasury[treasury_end]
        self.excess_returns[dt_loc] = (
            self.algorithm_cumulative_returns[dt_loc] -
            self.treasury_period_return)

        self.alpha[dt_loc], self.beta[dt_loc] = alpha_beta_aligned(
            self.algorithm_returns,
            self.benchmark_returns,
        )
        self.sharpe[dt_loc] = sharpe_ratio(
            self.algorithm_returns,
        )
        self.downside_risk[dt_loc] = downside_risk(
            self.algorithm_returns
        )
        self.sortino[dt_loc] = sortino_ratio(
            self.algorithm_returns,
            _downside_risk=self.downside_risk[dt_loc]
        )
        self.information[dt_loc] = information_ratio(
            self.algorithm_returns,
            self.benchmark_returns,
        )
        self.max_drawdown = max_drawdown(
            self.algorithm_returns
        )
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

    def update_current_max(self):
        if len(self.algorithm_cumulative_returns) == 0:
            return
        current_cumulative_return = \
            self.algorithm_cumulative_returns[self.latest_dt_loc]
        if self.current_max < current_cumulative_return:
            self.current_max = current_cumulative_return

    def calculate_max_leverage(self):
        # The leverage is defined as: the gross_exposure/net_liquidation
        # gross_exposure = long_exposure + abs(short_exposure)
        # net_liquidation = ending_cash + long_exposure + short_exposure
        cur_leverage = self.algorithm_cumulative_leverages_cont[
            self.latest_dt_loc]

        return max(cur_leverage, self.max_leverage)
