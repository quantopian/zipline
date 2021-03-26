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
import empyrical

from zipline.utils.deprecate import deprecated

from .core import (
    metrics_sets,
    register,
    unregister,
    load,
)
from .metric import (
    AlphaBeta,
    BenchmarkReturnsAndVolatility,
    CashFlow,
    DailyLedgerField,
    MaxLeverage,
    NumTradingDays,
    Orders,
    PeriodLabel,
    PNL,
    Returns,
    ReturnsStatistic,
    SimpleLedgerField,
    StartOfPeriodLedgerField,
    Transactions,
    _ConstantCumulativeRiskMetric,
    _ClassicRiskMetrics,
)
from .tracker import MetricsTracker


__all__ = ["MetricsTracker", "unregister", "metrics_sets", "load"]


register("none", set)


@register("default")
def default_metrics():
    return {
        Returns(),
        ReturnsStatistic(empyrical.annual_volatility, "algo_volatility"),
        BenchmarkReturnsAndVolatility(),
        PNL(),
        CashFlow(),
        Orders(),
        Transactions(),
        SimpleLedgerField("positions"),
        StartOfPeriodLedgerField(
            "portfolio.positions_exposure",
            "starting_exposure",
        ),
        DailyLedgerField(
            "portfolio.positions_exposure",
            "ending_exposure",
        ),
        StartOfPeriodLedgerField("portfolio.positions_value", "starting_value"),
        DailyLedgerField("portfolio.positions_value", "ending_value"),
        StartOfPeriodLedgerField("portfolio.cash", "starting_cash"),
        DailyLedgerField("portfolio.cash", "ending_cash"),
        DailyLedgerField("portfolio.portfolio_value"),
        DailyLedgerField("position_tracker.stats.longs_count"),
        DailyLedgerField("position_tracker.stats.shorts_count"),
        DailyLedgerField("position_tracker.stats.long_value"),
        DailyLedgerField("position_tracker.stats.short_value"),
        DailyLedgerField("position_tracker.stats.long_exposure"),
        DailyLedgerField("position_tracker.stats.short_exposure"),
        DailyLedgerField("account.gross_leverage"),
        DailyLedgerField("account.net_leverage"),
        AlphaBeta(),
        ReturnsStatistic(empyrical.sharpe_ratio, "sharpe"),
        ReturnsStatistic(empyrical.sortino_ratio, "sortino"),
        ReturnsStatistic(empyrical.max_drawdown),
        MaxLeverage(),
        # Please kill these!
        _ConstantCumulativeRiskMetric("excess_return", 0.0),
        _ConstantCumulativeRiskMetric("treasury_period_return", 0.0),
        NumTradingDays(),
        PeriodLabel(),
    }


@register("classic")
@deprecated(
    "The original risk packet has been deprecated and will be removed in a "
    'future release. Please use "default" metrics instead.'
)
def classic_metrics():
    metrics = default_metrics()
    metrics.add(_ClassicRiskMetrics())
    return metrics
