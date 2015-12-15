import datetime
import pytz

import nose.tools as nt
import pandas.util.testing as tm
import pandas as pd

from zipline.finance.blotter import Blotter, Order
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance.performance.period import PerformancePeriod
from zipline.finance.performance.position import Position
from zipline.finance.performance.tracker import PerformanceTracker
from zipline.finance.performance.position_tracker import PositionTracker
from zipline.finance.risk.cumulative import RiskMetricsCumulative
from zipline.finance.risk.period import RiskMetricsPeriod
from zipline.finance.risk.report import RiskReport
from zipline.finance.slippage import (
    FixedSlippage,
    VolumeShareSlippage
)
from zipline.finance.transaction import Transaction
from zipline.protocol import Account
from zipline.protocol import Portfolio
from zipline.protocol import Position as ProtocolPosition

from zipline.finance.trading import SimulationParameters, TradingEnvironment

from zipline.utils import factory


def stringify_cases(cases, func=None):
    # get better test case names
    results = []
    if func is None:
        def func(case):
            return case[0].__name__
    for case in cases:
        new_case = list(case)
        key = func(case)
        new_case.insert(0, key)
        results.append(new_case)
    return results

cases_env = TradingEnvironment()
sim_params_daily = SimulationParameters(
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    10000,
    emission_rate='daily',
    env=cases_env)
sim_params_minute = SimulationParameters(
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    10000,
    emission_rate='minute',
    env=cases_env)
returns = factory.create_returns_from_list(
    [1.0], sim_params_daily)


def object_serialization_cases(skip_daily=False):
    # Wrapped in a function to recreate DI objects.
    cases = [
        (Blotter, (), {}, 'repr'),
        (Order, (datetime.datetime(2013, 6, 19), 8554, 100), {}, 'dict'),
        (PerShare, (), {}, 'dict'),
        (PerTrade, (), {}, 'dict'),
        (PerDollar, (), {}, 'dict'),
        (PerformancePeriod,
            (10000, cases_env.asset_finder),
            {'position_tracker': PositionTracker(cases_env.asset_finder)},
            'to_dict'),
        (Position, (8554,), {}, 'dict'),
        (PositionTracker, (cases_env.asset_finder,), {}, 'dict'),
        (PerformanceTracker, (sim_params_minute, cases_env), {}, 'to_dict'),
        (RiskMetricsCumulative, (sim_params_minute, cases_env), {}, 'to_dict'),
        (RiskMetricsPeriod,
            (returns.index[0], returns.index[0], returns, cases_env),
         {}, 'to_dict'),
        (RiskReport, (returns, sim_params_minute, cases_env), {}, 'to_dict'),
        (FixedSlippage, (), {}, 'dict'),
        (Transaction,
            (8554, 10, datetime.datetime(2013, 6, 19), 100, "0000"), {},
            'dict'),
        (VolumeShareSlippage, (), {}, 'dict'),
        (Account, (), {}, 'dict'),
        (Portfolio, (), {}, 'dict'),
        (ProtocolPosition, (8554,), {}, 'dict')
    ]

    if not skip_daily:
        cases.extend([
            (PerformanceTracker,
             (sim_params_daily, cases_env), {}, 'to_dict'),
            (RiskMetricsCumulative,
             (sim_params_daily, cases_env), {}, 'to_dict'),
            (RiskReport,
             (returns, sim_params_daily, cases_env), {}, 'to_dict'),
        ])

    return stringify_cases(cases)


def assert_dict_equal(d1, d2):
    # check keys
    nt.assert_is_instance(d1, dict)
    nt.assert_is_instance(d2, dict)
    nt.assert_set_equal(set(d1.keys()), set(d2.keys()))
    for k in d1:
        v1 = d1[k]
        v2 = d2[k]

        asserter = nt.assert_equal
        if isinstance(v1, pd.DataFrame):
            asserter = tm.assert_frame_equal
        if isinstance(v1, pd.Series):
            asserter = tm.assert_series_equal

        try:
            asserter(v1, v2)
        except AssertionError:
            raise AssertionError('{k} is not equal'.format(k=k))
