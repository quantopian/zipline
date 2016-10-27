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

"""

Risk Report
===========

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | trading_days    | The number of trading days between self.start_date |
    |                 | and self.end_date                                  |
    +-----------------+----------------------------------------------------+
    | benchmark_volat\| The volatility of the benchmark between            |
    | ility           | self.start_date and self.end_date.                 |
    +-----------------+----------------------------------------------------+
    | algo_volatility | The volatility of the algo between self.start_date |
    |                 | and self.end_date.                                 |
    +-----------------+----------------------------------------------------+
    | treasury_period\| The return of treasuries over the period. Treasury |
    | _return         | maturity is chosen to match the duration of the    |
    |                 | test period.                                       |
    +-----------------+----------------------------------------------------+
    | sharpe          | The sharpe ratio based on the _algorithm_ (rather  |
    |                 | than the static portfolio) returns.                |
    +-----------------+----------------------------------------------------+
    | information     | The information ratio based on the _algorithm_     |
    |                 | (rather than the static portfolio) returns.        |
    +-----------------+----------------------------------------------------+
    | beta            | The _algorithm_ beta to the benchmark.             |
    +-----------------+----------------------------------------------------+
    | alpha           | The _algorithm_ alpha to the benchmark.            |
    +-----------------+----------------------------------------------------+
    | excess_return   | The excess return of the algorithm over the        |
    |                 | treasuries.                                        |
    +-----------------+----------------------------------------------------+
    | max_drawdown    | The largest relative peak to relative trough move  |
    |                 | for the portfolio returns between self.start_date  |
    |                 | and self.end_date.                                 |
    +-----------------+----------------------------------------------------+
    | max_leverage    | The largest gross leverage between self.start_date |
    |                 | and self.end_date                                  |
    +-----------------+----------------------------------------------------+

"""

import logbook
import datetime
from dateutil.relativedelta import relativedelta

from . period import RiskMetricsPeriod

log = logbook.Logger('Risk Report')


class RiskReport(object):
    def __init__(self, algorithm_returns, sim_params, trading_calendar,
                 treasury_curves, benchmark_returns,
                 algorithm_leverages=None):
        """
        algorithm_returns needs to be a list of daily_return objects
        sorted in date ascending order

        account needs to be a list of account objects sorted in date
        ascending order
        """

        self.algorithm_returns = algorithm_returns
        self.sim_params = sim_params
        self.trading_calendar = trading_calendar
        self.treasury_curves = treasury_curves
        self.benchmark_returns = benchmark_returns
        self.algorithm_leverages = algorithm_leverages

        if len(self.algorithm_returns) == 0:
            start_session = self.sim_params.start_session
            end_session = self.sim_params.end_session
        else:
            start_session = self.algorithm_returns.index[0]
            end_session = self.algorithm_returns.index[-1]

        self.month_periods = self.periods_in_range(
            1, start_session, end_session
        )
        self.three_month_periods = self.periods_in_range(
            3, start_session, end_session
        )
        self.six_month_periods = self.periods_in_range(
            6, start_session, end_session
        )
        self.year_periods = self.periods_in_range(
            12, start_session, end_session
        )

    def to_dict(self):
        """
        RiskMetrics are calculated for rolling windows in four lengths::
            - 1_month
            - 3_month
            - 6_month
            - 12_month

        The return value of this function is a dictionary keyed by the above
        list of durations. The value of each entry is a list of RiskMetric
        dicts of the same duration as denoted by the top_level key.

        See :py:meth:`RiskMetrics.to_dict` for the detailed list of fields
        provided for each period.
        """
        return {
            'one_month': [x.to_dict() for x in self.month_periods],
            'three_month': [x.to_dict() for x in self.three_month_periods],
            'six_month': [x.to_dict() for x in self.six_month_periods],
            'twelve_month': [x.to_dict() for x in self.year_periods],
        }

    def periods_in_range(self, months_per, start_session, end_session):
        one_day = datetime.timedelta(days=1)
        ends = []
        cur_start = start_session.replace(day=1)

        # in edge cases (all sids filtered out, start/end are adjacent)
        # a test will not generate any returns data
        if len(self.algorithm_returns) == 0:
            return ends

        # ensure that we have an end at the end of a calendar month, in case
        # the return series ends mid-month...
        the_end = end_session.replace(day=1) + relativedelta(months=1) - \
            one_day
        while True:
            cur_end = cur_start + relativedelta(months=months_per) - one_day
            if cur_end > the_end:
                break
            cur_period_metrics = RiskMetricsPeriod(
                start_session=cur_start,
                end_session=cur_end,
                returns=self.algorithm_returns,
                benchmark_returns=self.benchmark_returns,
                trading_calendar=self.trading_calendar,
                treasury_curves=self.treasury_curves,
                algorithm_leverages=self.algorithm_leverages,
            )

            ends.append(cur_period_metrics)
            cur_start = cur_start + relativedelta(months=1)

        return ends
