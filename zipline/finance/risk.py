#
# Copyright 2018 Quantopian, Inc.
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


import datetime
from dateutil.relativedelta import relativedelta
from functools import partial

import logbook
import numpy as np
import pandas as pd

from six import iteritems

from empyrical import (
    alpha_beta_aligned,
    annual_volatility,
    cum_returns,
    downside_risk,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio
)

log = logbook.Logger('risk')


def risk_metric_period(
        start_session,
        end_session,
        algorithm_returns,
        benchmark_returns,
        algorithm_leverages,
):
    """
    Creates a dictionary representing the state of the risk report.

    Parameters
    ----------
    start_session : pd.Timestamp
        Start of period (inclusive) to produce metrics on
    end_session : pd.Timestamp
        End of period (inclusive) to produce metrics on
    algorithm_returns : pd.Series(pd.Timestamp -> float)
        Series of algorithm returns as of the end of each session
    benchmark_returns : pd.Series(pd.Timestamp -> float)
        Series of benchmark returns as of the end of each session
    algorithm_leverages : pd.Series(pd.Timestamp -> float)
        Series of algorithm leverages as of the end of each session


    Returns
    -------
    risk_metric : dict[str -> scalar]
        Dict of metrics that with fields like:
            {
                'algorithm_period_return': 0.0,
                'benchmark_period_return': 0.0,
                'treasury_period_return': 0,
                'excess_return': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'period_label': '1970-01',
                'trading_days': 0,
                'algo_volatility': 0.0,
                'benchmark_volatility': 0.0,
                'max_drawdown': 0.0,
                'max_leverage': 0.0,
            }
    """

    algorithm_returns = algorithm_returns[
        (algorithm_returns.index >= start_session) &
        (algorithm_returns.index <= end_session)
    ]

    # Benchmark needs to be masked to the same dates as the algo returns
    benchmark_returns = benchmark_returns[
        (benchmark_returns.index >= start_session) &
        (benchmark_returns.index <= end_session)
    ]

    assert algorithm_returns.index.equals(benchmark_returns.index), \
        ("Mismatch between benchmark_returns ({bm_count}) and "
         "algorithm_returns ({algo_count}) in range {start} : {end}").format(
            bm_count=len(benchmark_returns),
            algo_count=len(algorithm_returns),
            start=start_session,
            end=end_session,
        )

    benchmark_period_returns = cum_returns(benchmark_returns).iloc[-1]
    algorithm_period_returns = cum_returns(algorithm_returns).iloc[-1]

    alpha, beta = alpha_beta_aligned(
        algorithm_returns.values,
        benchmark_returns.values,
    )

    sharpe = sharpe_ratio(algorithm_returns)

    # The consumer currently expects a 0.0 value for sharpe in period,
    # this differs from cumulative which was np.nan.
    # When factoring out the sharpe_ratio, the different return types
    # were collapsed into `np.nan`.
    # TODO: Either fix consumer to accept `np.nan` or make the
    # `sharpe_ratio` return type configurable.
    # In the meantime, convert nan values to 0.0
    if pd.isnull(sharpe):
        sharpe = 0.0

    sortino = sortino_ratio(
        algorithm_returns.values,
        _downside_risk=downside_risk(algorithm_returns.values),
    )

    rval = {
        'algorithm_period_return': algorithm_period_returns,
        'benchmark_period_return': benchmark_period_returns,
        'treasury_period_return': 0,
        'excess_return': algorithm_period_returns,
        'alpha': alpha,
        'beta': beta,
        'sharpe': sharpe,
        'sortino': sortino,
        'period_label': end_session.strftime("%Y-%m"),
        'trading_days': len(benchmark_returns),
        'algo_volatility': annual_volatility(algorithm_returns),
        'benchmark_volatility': annual_volatility(benchmark_returns),
        'max_drawdown': max_drawdown(algorithm_returns.values),
        'max_leverage': algorithm_leverages.max(),
    }

    # check if a field in rval is nan or inf, and replace it with None except
    # period_label which is always a str
    return {
        k: None if k != 'period_label' and (np.isnan(v) or np.isinf(v)) else v
        for k, v in iteritems(rval)
    }


def _periods_in_range(
        months,
        end_session,
        end_date,
        algorithm_returns,
        benchmark_returns,
        algorithm_leverages,
        months_per,
):
    if months.size < months_per:
        return []

    end_date = end_date.tz_convert(None)
    result = []
    for period_timestamp in months:
        period = period_timestamp.to_period(freq='%dM' % months_per)
        if period.end_time > end_date:
            break
        result.append(risk_metric_period(
            start_session=period.start_time,
            end_session=min(period.end_time, end_session),
            algorithm_returns=algorithm_returns,
            benchmark_returns=benchmark_returns,
            algorithm_leverages=algorithm_leverages,
        ))

    return result


def risk_report(
        algorithm_returns,
        benchmark_returns,
        algorithm_leverages,
):
    """
    A metric that reports an end of simulation scalar or time series
    computed from the algorithm returns.

    Parameters
    ----------
    algorithm_returns : pd.Series(pd.Timestamp -> float)
        Series of algorithm returns as of the end of each session
    benchmark_returns : pd.Series(pd.Timestamp -> float)
        Series of benchmark returns as of the end of each session
    algorithm_leverages : pd.Series(pd.Timestamp -> float)
        Series of algorithm leverages as of the end of each session

    risk metrics are calculated for rolling windows in four lengths::
        - one_month
        - three_month
        - six_month
        - twelve_month

    Returns
    -------
    risk_metrics : dict[str -> list[dict]]
        Dictionary keyed by the above list of durations. The value of each
        entry is a list of risk metric dicts of the same duration as denoted
        by the top_level key.

    See :py:meth:`risk_metric_period` for the detailed list of fields
    provided for each period.
    """

    start_session = algorithm_returns.index[0]
    end_session = algorithm_returns.index[-1]

    end = end_session.replace(day=1) + relativedelta(months=1)
    months = pd.date_range(
        start=start_session,
        # Ensure we have at least one month
        end=end - datetime.timedelta(days=1),
        freq='M',
        tz='utc',
    )

    periods_in_range = partial(
        _periods_in_range,
        months=months,
        end_session=end_session.tz_convert(None),
        end_date=end,
        algorithm_returns=algorithm_returns,
        benchmark_returns=benchmark_returns,
        algorithm_leverages=algorithm_leverages,
    )

    month_periods = periods_in_range(months_per=1)
    three_month_periods = periods_in_range(months_per=3)
    six_month_periods = periods_in_range(months_per=6)
    year_periods = periods_in_range(months_per=12)

    return {
        'one_month': month_periods,
        'three_month': three_month_periods,
        'six_month': six_month_periods,
        'twelve_month': year_periods,
    }
