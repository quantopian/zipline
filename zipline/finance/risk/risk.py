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
import math
import numpy as np

import zipline.utils.math_utils as zp_math

log = logbook.Logger('Risk')


TREASURY_DURATIONS = [
    '1month', '3month', '6month',
    '1year', '2year', '3year', '5year',
    '7year', '10year', '30year'
]


# check if a field in rval is nan, and replace it with
# None.
def check_entry(key, value):
    if key != 'period_label':
        return np.isnan(value) or np.isinf(value)
    else:
        return False


############################
# Risk Metric Calculations #
############################


def sharpe_ratio(algorithm_volatility, algorithm_return, treasury_return):
    """
    http://en.wikipedia.org/wiki/Sharpe_ratio

    Args:
        algorithm_volatility (float): Algorithm volatility.
        algorithm_return (float): Algorithm return percentage.
        treasury_return (float): Treasury return percentage.

    Returns:
        float. The Sharpe ratio.
    """
    if zp_math.tolerant_equals(algorithm_volatility, 0):
        return np.nan

    return (algorithm_return - treasury_return) / algorithm_volatility


def downside_risk(algorithm_returns, mean_returns, normalization_factor):
    rets = algorithm_returns.round(8)
    mar = mean_returns.round(8)
    mask = rets < mar
    downside_diff = rets[mask] - mar[mask]
    if len(downside_diff) <= 1:
        return 0.0
    return np.std(downside_diff, ddof=1) * math.sqrt(normalization_factor)


def sortino_ratio(algorithm_period_return, treasury_period_return, mar):
    """
    http://en.wikipedia.org/wiki/Sortino_ratio

    Args:
        algorithm_returns (np.array-like):
            Returns from algorithm lifetime.
        algorithm_period_return (float):
            Algorithm return percentage from latest period.
        mar (float): Minimum acceptable return.

    Returns:
        float. The Sortino ratio.
    """
    if zp_math.tolerant_equals(mar, 0):
        return 0.0

    return (algorithm_period_return - treasury_period_return) / mar


def information_ratio(algorithm_returns, benchmark_returns):
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
    relative_returns = algorithm_returns - benchmark_returns

    relative_deviation = relative_returns.std(ddof=1)

    if zp_math.tolerant_equals(relative_deviation, 0) or \
       np.isnan(relative_deviation):
        return 0.0

    return np.mean(relative_returns) / relative_deviation


def alpha(algorithm_period_return, treasury_period_return,
          benchmark_period_returns, beta):
    """
    http://en.wikipedia.org/wiki/Alpha_(investment)

    Args:
        algorithm_period_return (float):
            Return percentage from algorithm period.
        treasury_period_return (float):
            Return percentage for treasury period.
        benchmark_period_return (float):
            Return percentage for benchmark period.
        beta (float):
            beta value for the same period as all other values

    Returns:
        float. The alpha of the algorithm.
    """
    return algorithm_period_return - \
        (treasury_period_return + beta *
         (benchmark_period_returns - treasury_period_return))

###########################
# End Risk Metric Section #
###########################


def get_treasury_rate(treasury_curves, treasury_duration, day):
    rate = None

    curve = treasury_curves.ix[day]
    # 1month note data begins in 8/2001,
    # so we can use 3month instead.
    idx = TREASURY_DURATIONS.index(treasury_duration)
    for duration in TREASURY_DURATIONS[idx:]:
        rate = curve[duration]
        if rate is not None:
            break

    return rate


def search_day_distance(end_date, dt, env):
    tdd = env.trading_day_distance(dt, end_date)
    if tdd is None:
        return None
    assert tdd >= 0
    return tdd


def select_treasury_duration(start_date, end_date):
    td = end_date - start_date
    if td.days <= 31:
        treasury_duration = '1month'
    elif td.days <= 93:
        treasury_duration = '3month'
    elif td.days <= 186:
        treasury_duration = '6month'
    elif td.days <= 366:
        treasury_duration = '1year'
    elif td.days <= 365 * 2 + 1:
        treasury_duration = '2year'
    elif td.days <= 365 * 3 + 1:
        treasury_duration = '3year'
    elif td.days <= 365 * 5 + 2:
        treasury_duration = '5year'
    elif td.days <= 365 * 7 + 2:
        treasury_duration = '7year'
    elif td.days <= 365 * 10 + 2:
        treasury_duration = '10year'
    else:
        treasury_duration = '30year'

    return treasury_duration


def choose_treasury(select_treasury, treasury_curves, start_date, end_date,
                    env, compound=True):
    """
    Find the latest known interest rate for a given duration within a date
    range.

    If we find one but it's more than a trading day ago from the date we're
    looking for, then we log a warning
    """
    treasury_duration = select_treasury(start_date, end_date)
    end_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    search_day = None

    if end_day in treasury_curves.index:
        rate = get_treasury_rate(treasury_curves,
                                 treasury_duration,
                                 end_day)
        if rate is not None:
            search_day = end_day

    if not search_day:
        # in case end date is not a trading day or there is no treasury
        # data, search for the previous day with an interest rate.
        search_days = treasury_curves.index

        # Find rightmost value less than or equal to end_day
        i = search_days.searchsorted(end_day)
        for prev_day in search_days[i - 1::-1]:
            rate = get_treasury_rate(treasury_curves,
                                     treasury_duration,
                                     prev_day)
            if rate is not None:
                search_day = prev_day
                search_dist = search_day_distance(end_date, prev_day, env)
                break

        if search_day:
            if (search_dist is None or search_dist > 1) and \
                    search_days[0] <= end_day <= search_days[-1]:
                message = "No rate within 1 trading day of end date = \
{dt} and term = {term}. Using {search_day}. Check that date doesn't exceed \
treasury history range."
                message = message.format(dt=end_date,
                                         term=treasury_duration,
                                         search_day=search_day)
                log.warn(message)

    if search_day:
        td = end_date - start_date
        if compound:
            return rate * (td.days + 1) / 365
        else:
            return rate

    message = "No rate for end date = {dt} and term = {term}. Check \
that date doesn't exceed treasury history range."
    message = message.format(
        dt=end_date,
        term=treasury_duration
    )
    raise Exception(message)
