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
from functools import partial
import operator as op

from dateutil.relativedelta import relativedelta
import empyrical as ep
import numpy as np
import pandas as pd

from zipline.utils.exploding_object import NamedExplodingObject
from zipline.finance._finance_ext import minute_annual_volatility


class SimpleLedgerField:
    """Emit the current value of a ledger field every bar or every session.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """

    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit(".", 1)[-1]
        else:
            self._packet_field = packet_field

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        packet["minute_perf"][self._packet_field] = self._get_ledger_field(
            ledger,
        )

    def end_of_session(self, packet, ledger, session, session_ix, data_portal):
        packet["daily_perf"][self._packet_field] = self._get_ledger_field(
            ledger,
        )


class DailyLedgerField:
    """Like :class:`~zipline.finance.metrics.metric.SimpleLedgerField` but
    also puts the current value in the ``cumulative_perf`` section.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """

    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit(".", 1)[-1]
        else:
            self._packet_field = packet_field

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        field = self._packet_field
        packet["cumulative_perf"][field] = packet["minute_perf"][
            field
        ] = self._get_ledger_field(ledger)

    def end_of_session(self, packet, ledger, session, session_ix, data_portal):
        field = self._packet_field
        packet["cumulative_perf"][field] = packet["daily_perf"][
            field
        ] = self._get_ledger_field(ledger)


class StartOfPeriodLedgerField:
    """Keep track of the value of a ledger field at the start of the period.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """

    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit(".", 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        self._start_of_simulation = self._get_ledger_field(ledger)

    def start_of_session(self, ledger, session, data_portal):
        self._previous_day = self._get_ledger_field(ledger)

    def _end_of_period(self, sub_field, packet, ledger):
        packet_field = self._packet_field
        packet["cumulative_perf"][packet_field] = self._start_of_simulation
        packet[sub_field][packet_field] = self._previous_day

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        self._end_of_period("minute_perf", packet, ledger)

    def end_of_session(self, packet, ledger, session, session_ix, data_portal):
        self._end_of_period("daily_perf", packet, ledger)


class Returns:
    """Tracks the daily and cumulative returns of the algorithm."""

    def _end_of_period(field, packet, ledger, dt, session_ix, data_portal):
        packet[field]["returns"] = ledger.todays_returns
        packet["cumulative_perf"]["returns"] = ledger.portfolio.returns
        packet["cumulative_risk_metrics"][
            "algorithm_period_return"
        ] = ledger.portfolio.returns

    end_of_bar = partial(_end_of_period, "minute_perf")
    end_of_session = partial(_end_of_period, "daily_perf")


class BenchmarkReturnsAndVolatility:
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        daily_returns_series = benchmark_source.daily_returns(
            sessions[0],
            sessions[-1],
        )
        self._daily_returns = daily_returns_array = daily_returns_series.values
        self._daily_cumulative_returns = np.cumprod(1 + daily_returns_array) - 1
        self._daily_annual_volatility = (
            daily_returns_series.expanding(2).std(ddof=1) * np.sqrt(252)
        ).values

        if emission_rate == "daily":
            self._minute_cumulative_returns = NamedExplodingObject(
                "self._minute_cumulative_returns",
                "does not exist in daily emission rate",
            )
            self._minute_annual_volatility = NamedExplodingObject(
                "self._minute_annual_volatility",
                "does not exist in daily emission rate",
            )
        else:
            open_ = trading_calendar.session_open(sessions[0])
            close = trading_calendar.session_close(sessions[-1])
            returns = benchmark_source.get_range(open_, close)
            self._minute_cumulative_returns = (1 + returns).cumprod() - 1
            self._minute_annual_volatility = pd.Series(
                minute_annual_volatility(
                    returns.index.normalize().view("int64"),
                    returns.values,
                    daily_returns_array,
                ),
                index=returns.index,
            )

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        r = self._minute_cumulative_returns[dt]
        if np.isnan(r):
            r = None
        packet["cumulative_risk_metrics"]["benchmark_period_return"] = r

        v = self._minute_annual_volatility[dt]
        if np.isnan(v):
            v = None
        packet["cumulative_risk_metrics"]["benchmark_volatility"] = v

    def end_of_session(self, packet, ledger, session, session_ix, data_portal):
        r = self._daily_cumulative_returns[session_ix]
        if np.isnan(r):
            r = None
        packet["cumulative_risk_metrics"]["benchmark_period_return"] = r

        v = self._daily_annual_volatility[session_ix]
        if np.isnan(v):
            v = None
        packet["cumulative_risk_metrics"]["benchmark_volatility"] = v


class PNL:
    """Tracks daily and cumulative PNL."""

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        self._previous_pnl = 0.0

    def start_of_session(self, ledger, session, data_portal):
        self._previous_pnl = ledger.portfolio.pnl

    def _end_of_period(self, field, packet, ledger):
        pnl = ledger.portfolio.pnl
        packet[field]["pnl"] = pnl - self._previous_pnl
        packet["cumulative_perf"]["pnl"] = ledger.portfolio.pnl

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        self._end_of_period("minute_perf", packet, ledger)

    def end_of_session(self, packet, ledger, session, session_ix, data_portal):
        self._end_of_period("daily_perf", packet, ledger)


class CashFlow:
    """Tracks daily and cumulative cash flow.

    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        self._previous_cash_flow = 0.0

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        cash_flow = ledger.portfolio.cash_flow
        packet["minute_perf"]["capital_used"] = cash_flow - self._previous_cash_flow
        packet["cumulative_perf"]["capital_used"] = cash_flow

    def end_of_session(self, packet, ledger, session, session_ix, data_portal):
        cash_flow = ledger.portfolio.cash_flow
        packet["daily_perf"]["capital_used"] = cash_flow - self._previous_cash_flow
        packet["cumulative_perf"]["capital_used"] = cash_flow
        self._previous_cash_flow = cash_flow


class Orders:
    """Tracks daily orders."""

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        packet["minute_perf"]["orders"] = ledger.orders(dt)

    def end_of_session(self, packet, ledger, dt, session_ix, data_portal):
        packet["daily_perf"]["orders"] = ledger.orders()


class Transactions:
    """Tracks daily transactions."""

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        packet["minute_perf"]["transactions"] = ledger.transactions(dt)

    def end_of_session(self, packet, ledger, dt, session_ix, data_portal):
        packet["daily_perf"]["transactions"] = ledger.transactions()


class Positions:
    """Tracks daily positions."""

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        packet["minute_perf"]["positions"] = ledger.positions(dt)

    def end_of_session(self, packet, ledger, dt, session_ix, data_portal):
        packet["daily_perf"]["positions"] = ledger.positions()


class ReturnsStatistic:
    """A metric that reports an end of simulation scalar or time series
    computed from the algorithm returns.

    Parameters
    ----------
    function : callable
        The function to call on the daily returns.
    field_name : str, optional
        The name of the field. If not provided, it will be
        ``function.__name__``.
    """

    def __init__(self, function, field_name=None):
        if field_name is None:
            field_name = function.__name__

        self._function = function
        self._field_name = field_name

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        res = self._function(ledger.daily_returns_array[: session_ix + 1])
        if not np.isfinite(res):
            res = None
        packet["cumulative_risk_metrics"][self._field_name] = res

    end_of_session = end_of_bar


class AlphaBeta:
    """End of simulation alpha and beta to the benchmark."""

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        self._daily_returns_array = benchmark_source.daily_returns(
            sessions[0],
            sessions[-1],
        ).values

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        risk = packet["cumulative_risk_metrics"]

        alpha, beta = ep.alpha_beta_aligned(
            ledger.daily_returns_array[: session_ix + 1],
            self._daily_returns_array[: session_ix + 1],
        )
        if not np.isfinite(alpha):
            alpha = None
        if np.isnan(beta):
            beta = None

        risk["alpha"] = alpha
        risk["beta"] = beta

    end_of_session = end_of_bar


class MaxLeverage:
    """Tracks the maximum account leverage."""

    def start_of_simulation(self, *args):
        self._max_leverage = 0.0

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        self._max_leverage = max(self._max_leverage, ledger.account.leverage)
        packet["cumulative_risk_metrics"]["max_leverage"] = self._max_leverage

    end_of_session = end_of_bar


class NumTradingDays:
    """Report the number of trading days."""

    def start_of_simulation(self, *args):
        self._num_trading_days = 0

    def start_of_session(self, *args):
        self._num_trading_days += 1

    def end_of_bar(self, packet, ledger, dt, session_ix, data_portal):
        packet["cumulative_risk_metrics"]["trading_days"] = self._num_trading_days

    end_of_session = end_of_bar


class _ConstantCumulativeRiskMetric:
    """A metric which does not change, ever.

    Notes
    -----
    This exists to maintain the existing structure of the perf packets. We
    should kill this as soon as possible.
    """

    def __init__(self, field, value):
        self._field = field
        self._value = value

    def end_of_bar(self, packet, *args):
        packet["cumulative_risk_metrics"][self._field] = self._value

    def end_of_session(self, packet, *args):
        packet["cumulative_risk_metrics"][self._field] = self._value


class PeriodLabel:
    """Backwards compat, please kill me."""

    def start_of_session(self, ledger, session, data_portal):
        self._label = session.strftime("%Y-%m")

    def end_of_bar(self, packet, *args):
        packet["cumulative_risk_metrics"]["period_label"] = self._label

    end_of_session = end_of_bar


class _ClassicRiskMetrics:
    """Produces original risk packet."""

    def start_of_simulation(
        self, ledger, emission_rate, trading_calendar, sessions, benchmark_source
    ):
        self._leverages = np.full_like(sessions, np.nan, dtype="float64")

    def end_of_session(self, packet, ledger, dt, session_ix, data_portal):
        self._leverages[session_ix] = ledger.account.leverage

    @classmethod
    def risk_metric_period(
        cls,
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
        risk_metric : dict[str, any]
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
            (algorithm_returns.index >= start_session)
            & (algorithm_returns.index <= end_session)
        ]

        # Benchmark needs to be masked to the same dates as the algo returns
        benchmark_ret_tzinfo = benchmark_returns.index.tzinfo
        benchmark_returns = benchmark_returns[
            (benchmark_returns.index >= start_session.tz_localize(benchmark_ret_tzinfo))
            & (
                benchmark_returns.index
                <= algorithm_returns.index[-1].tz_localize(benchmark_ret_tzinfo)
            )
        ]
        benchmark_period_returns = ep.cum_returns(benchmark_returns).iloc[-1]
        algorithm_period_returns = ep.cum_returns(algorithm_returns).iloc[-1]

        alpha, beta = ep.alpha_beta_aligned(
            algorithm_returns.values,
            benchmark_returns.values,
        )
        benchmark_volatility = ep.annual_volatility(benchmark_returns)

        sharpe = ep.sharpe_ratio(algorithm_returns)

        # The consumer currently expects a 0.0 value for sharpe in period,
        # this differs from cumulative which was np.nan.
        # When factoring out the sharpe_ratio, the different return types
        # were collapsed into `np.nan`.
        # TODO: Either fix consumer to accept `np.nan` or make the
        # `sharpe_ratio` return type configurable.
        # In the meantime, convert nan values to 0.0
        if pd.isnull(sharpe):
            sharpe = 0.0

        sortino = ep.sortino_ratio(
            algorithm_returns.values,
            _downside_risk=ep.downside_risk(algorithm_returns.values),
        )

        rval = {
            "algorithm_period_return": algorithm_period_returns,
            "benchmark_period_return": benchmark_period_returns,
            "treasury_period_return": 0,
            "excess_return": algorithm_period_returns,
            "alpha": alpha,
            "beta": beta,
            "sharpe": sharpe,
            "sortino": sortino,
            "period_label": end_session.strftime("%Y-%m"),
            "trading_days": len(benchmark_returns),
            "algo_volatility": ep.annual_volatility(algorithm_returns),
            "benchmark_volatility": benchmark_volatility,
            "max_drawdown": ep.max_drawdown(algorithm_returns.values),
            "max_leverage": algorithm_leverages.max(),
        }

        # check if a field in rval is nan or inf, and replace it with None
        # except period_label which is always a str
        return {
            k: (None if k != "period_label" and not np.isfinite(v) else v)
            for k, v in rval.items()
        }

    @classmethod
    def _periods_in_range(
        cls,
        months,
        end_session,
        end_date,
        algorithm_returns,
        benchmark_returns,
        algorithm_leverages,
        months_per,
    ):
        if months.size < months_per:
            return

        tzinfo = end_date.tzinfo
        end_date = end_date
        for period_timestamp in months:
            period = period_timestamp.tz_localize(None).to_period(
                freq="%dM" % months_per
            )
            if period.end_time > end_date:
                break

            yield cls.risk_metric_period(
                start_session=period.start_time.tz_localize(tzinfo),
                end_session=min(period.end_time, end_session).tz_localize(tzinfo),
                algorithm_returns=algorithm_returns,
                benchmark_returns=benchmark_returns,
                algorithm_leverages=algorithm_leverages,
            )

    @classmethod
    def risk_report(cls, algorithm_returns, benchmark_returns, algorithm_leverages):
        start_session = algorithm_returns.index[0]
        end_session = algorithm_returns.index[-1]

        end = end_session.replace(day=1) + relativedelta(months=1)
        months = pd.date_range(
            start=start_session,
            # Ensure we have at least one month
            end=end - datetime.timedelta(days=1),
            freq="M",
            tz="utc",
        )

        periods_in_range = partial(
            cls._periods_in_range,
            months=months,
            end_session=end_session,
            end_date=end,
            algorithm_returns=algorithm_returns,
            benchmark_returns=benchmark_returns,
            algorithm_leverages=algorithm_leverages,
        )

        return {
            "one_month": list(periods_in_range(months_per=1)),
            "three_month": list(periods_in_range(months_per=3)),
            "six_month": list(periods_in_range(months_per=6)),
            "twelve_month": list(periods_in_range(months_per=12)),
        }

    def end_of_simulation(
        self, packet, ledger, trading_calendar, sessions, data_portal, benchmark_source
    ):
        packet.update(
            self.risk_report(
                algorithm_returns=ledger.daily_returns_series,
                benchmark_returns=benchmark_source.daily_returns(
                    sessions[0],
                    sessions[-1],
                ),
                algorithm_leverages=self._leverages,
            )
        )
