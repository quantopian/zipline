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

from math import sqrt
import operator as op

import empyrical
import numpy as np
import pandas as pd

from zipline.utils.exploding_object import NamedExplodingObject
from ._metric import minute_annual_volatility


class SimpleLedgerField(object):
    """Emit the current value of a ledger field every day.

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
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf'][self._packet_field] = self._get_ledger_field(
            ledger,
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf'][self._packet_field] = self._get_ledger_field(
            ledger,
        )


class DailyLedgerField(object):
    """Keep a daily record of a field of the ledger object.

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
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._daily_value = pd.Series(np.nan, index=sessions)

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        field = self._packet_field
        packet['cumulative_perf'][field] = packet['minute_perf'][field] = (
            self._get_ledger_field(ledger)
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        self._daily_value[session] = value = self._get_ledger_field(ledger)

        field = self._packet_field
        packet['cumulative_perf'][field] = packet['daily_perf'][field] = (
            value
        )

    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet[self._packet_field] = self._daily_value.tolist()


class StartOfPeriodLedgerField(object):
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
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._start_of_simulation = self._get_ledger_field(ledger)

    def start_of_session(self, ledger, session, data_portal):
        self._previous_day = self._get_ledger_field(ledger)

    def _end_of_period(self, sub_field, packet, ledger):
        packet_field = self._packet_field
        packet['cumulative_perf'][packet_field] = self._start_of_simulation
        packet[sub_field][packet_field] = self._previous_day

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        self._end_of_period('minute_perf', packet, ledger)

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        self._end_of_period('daily_perf', packet, ledger)


class ReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the algorithm.
    """
    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._previous_total_returns = 0

        self._day_ix = 0
        self._daily_sum = 0
        self._todays_prod = 1
        self._annualization_factor = sqrt(252)

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        current_total_returns = ledger.portfolio.returns
        todays_returns = (
            (current_total_returns + 1) /
            (self._previous_total_returns + 1) -
            1
        )

        packet['minute_perf']['returns'] = todays_returns
        packet['cumulative_perf']['returns'] = current_total_returns

        if self._day_ix < 1:
            variance = np.nan
        else:
            self._todays_prod *= 1 + todays_returns

            intermediate_sum = self._daily_sum + self._todays_prod - 1
            mean = intermediate_sum / (self._day_ix + 1)

            demeaned_old = ledger.daily_returns[:dt.normalize()] - mean
            variance = (
                (self._todays_prod - 1 - mean) ** 2 +
                demeaned_old.dot(demeaned_old)
            )

        packet['cumulative_risk_metrics']['algorithm_volatility'] = (
            sqrt(variance) * self._annualization_factor
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf']['returns'] = dr = ledger.daily_returns[session]
        packet['cumulative_perf']['returns'] = r = ledger.portfolio.returns
        self._previous_total_returns = r

        self._day_ix += 1
        self._daily_sum += dr
        self._todays_prod = 1

        mean = self._daily_sum / self._day_ix
        variance = ((ledger.daily_returns[:session] - mean) ** 2).sum()
        packet['cumulative_risk_metrics']['algorithm_volatility'] = (
            sqrt(variance) * self._annualization_factor
        )

    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet['cumulative_algorithm_returns'] = (
            (1 + ledger.daily_returns).prod() - 1
        )
        packet['daily_algorithm_returns'] = ledger.daily_returns.tolist()


class BenchmarkReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """
    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._daily_returns = benchmark_source.daily_returns(
            sessions[0],
            sessions[-1],
        )
        self._daily_cumulative_returns = (
            (1 + self._daily_returns).cumprod() - 1
        )
        self._daily_annual_volatility = (
            self._daily_returns.expanding(2).std(ddof=1) * np.sqrt(252)
        )

        if emission_rate == 'daily':
            self._minute_returns = NamedExplodingObject(
                'self._minute_returns',
                'does not exist in daily emission rate',
            )
            self._minute_cumulative_returns = NamedExplodingObject(
                'self._minute_cumulative_returns',
                'does not exist in daily emission rate',
            )
            self._minute_annual_volatility = NamedExplodingObject(
                'self._minute_annual_volatility',
                'does not exist in daily emission rate',
            )
        else:
            open_ = trading_calendar.session_open(sessions[0])
            close = trading_calendar.session_close(sessions[-1])
            returns = benchmark_source.get_range(open_, close)
            self._minute_returns = returns.groupby(pd.TimeGrouper('D')).apply(
                lambda g: (g + 1).cumprod() - 1,
            )
            self._minute_cumulative_returns = (1 + returns).cumprod() - 1
            self._minute_annual_volatility = pd.Series(
                minute_annual_volatility(
                    returns.index.normalize().view('int64'),
                    returns.values,
                    self._daily_returns.values,
                ),
                index=returns.index,
            )

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['benchmark_returns'] = self._minute_returns[dt]
        packet['cumulative_perf']['benchmark_returns'] = (
            self._minute_cumulative_returns[dt]
        )
        packet['cumulative_risk_metrics']['benchmark_volatility'] = (
            self._minute_annual_volatility[dt]
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf']['benchmark_returns'] = (
            self._daily_returns[session]
        )
        packet['cumulative_perf']['benchmark_returns'] = (
            self._daily_cumulative_returns[session]
        )
        packet['cumulative_risk_metrics']['benchmark_volatility'] = (
            self._daily_annual_volatility[session]
        )

    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet['cumulative_benchmark_returns'] = (
            self._daily_cumulative_returns.iloc[-1]
        )
        packet['daily_benchmark_returns'] = self._daily_returns.tolist()


class PNL(object):
    """Tracks daily and total PNL.
    """
    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        # We start the index at -1 because we want to point the previous day.
        # -1 will wrap around and point to the *last* day; however, we
        # initialize the whole series to 0 so this will give us the results
        # we want without an explicit check.
        self._pnl_index = -1
        self._pnl = pd.Series(0, index=sessions)

    def _compute_pnl_in_period(self, ledger):
        return ledger.portfolio.pnl - self._pnl[self._pnl_index]

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['pnl'] = self._compute_pnl_in_period(ledger)
        packet['cumulative_perf']['pnl'] = ledger.portfolio.pnl

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf']['pnl'] = self._compute_pnl_in_period(ledger)
        packet['cumulative_perf']['pnl'] = pnl = ledger.portfolio.pnl
        self._pnl_index += 1
        self._pnl[self._pnl_index] = pnl

    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet['total_pnl'] = ledger.portfolio.pnl
        packet['daily_pnl'] = self._pnl.tolist()


class CashFlow(object):
    """Tracks daily and cumulative cash flow.

    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """
    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._previous_cash_flow = 0.0

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        cash_flow = ledger.portfolio.cash_flow
        packet['minute_perf']['capital_used'] = (
            cash_flow - self._previous_cash_flow
        )
        packet['cumulative_perf']['capital_used'] = cash_flow

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        cash_flow = ledger.portfolio.cash_flow
        packet['daily_perf']['capital_used'] = (
            cash_flow - self._previous_cash_flow
        )
        packet['cumulative_perf']['capital_used'] = cash_flow
        self._previous_cash_flow = cash_flow


class Orders(object):
    """Tracks daily orders.
    """
    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['orders'] = ledger.orders(dt)

    def end_of_session(self,
                       packet,
                       ledger,
                       dt,
                       data_portal):
        packet['daily_perf']['orders'] = ledger.orders()


class Transactions(object):
    """Tracks daily transactions.
    """
    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['transactions'] = ledger.transactions(dt)

    def end_of_session(self,
                       packet,
                       ledger,
                       dt,
                       data_portal):
        packet['daily_perf']['transactions'] = ledger.transactions()


class Positions(object):
    """Tracks daily positions.
    """
    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['positions'] = ledger.positions(dt)

    def end_of_session(self,
                       packet,
                       ledger,
                       dt,
                       data_portal):
        packet['daily_perf']['positions'] = ledger.positions()


class ReturnsStatistic(object):
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

    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet[self._field_name] = self._function(ledger.daily_returns)


class AlphaBeta(object):
    """End of simulation alpha and beta to the benchmark.
    """
    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet['alpha'], packet['beta'] = empyrical.alpha_beta_aligned(
            ledger.daily_returns,
            benchmark_source.daily_returns(sessions[0], sessions[-1]),
        )


class MaxLeverage(object):
    """Tracks the maximum account leverage.
    """
    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._max_leverage = 0.0

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        self._max_leverage = max(self._max_leverage, ledger.account.leverage)

    end_of_session = end_of_bar

    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet['max_leverage'] = self._max_leverage


class NumTradingDays(object):
    """Report the number of trading days.
    """
    def end_of_simulation(self, packet, ledger, benchmark_source, sessions):
        packet['num_trading_days'] = len(sessions)
