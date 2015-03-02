#
# Copyright 2015 Quantopian, Inc.
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
import pytz

from nose_parameterized import parameterized
from unittest import TestCase

from zipline.finance.blotter import Blotter, Order
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance.performance.period import PerformancePeriod
from zipline.finance.performance.position import Position
from zipline.finance.performance.tracker import PerformanceTracker
from zipline.finance.risk.cumulative import RiskMetricsCumulative
from zipline.finance.risk.period import RiskMetricsPeriod
from zipline.finance.risk.report import RiskReport
from zipline.finance.slippage import (
    FixedSlippage,
    Transaction,
    VolumeShareSlippage
)
from zipline.protocol import Account
from zipline.protocol import Portfolio
from zipline.protocol import Position as ProtocolPosition


from zipline.finance.trading import SimulationParameters

from zipline.utils import factory

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class SerializationTestCase(TestCase):

    @parameterized.expand([
        (Order, (datetime.datetime(2013, 6, 19), 8554, 100)),
        (PerShare, ()),
        (PerTrade, ()),
        (PerDollar, ()),
        (Position, (8554,)),
        (FixedSlippage, ()),
        (Transaction, (8554, 10, datetime.datetime(2013, 6, 19), 100, "0000")),
        (VolumeShareSlippage, ()),
        (Account, ()),
        (Portfolio, ()),
        (ProtocolPosition, (8554,))
    ])
    def test_object_serialization(self, cls, initargs):

        obj = cls(*initargs)
        state = obj.__getstate__()
        if hasattr(obj, '__getinitargs__'):
            initargs = obj.__getinitargs__()
        else:
            initargs = None

        obj2 = cls.__new__(cls)
        if initargs is not None:
            obj2.__init__(*initargs)
        obj2.__setstate__(state)

        self.assertEqual(obj.__dict__, obj2.__dict__)

    # Need special handling to compare equality for some objects

    def test_perf_period_serialization(self):

        obj = PerformancePeriod(10000)
        state = obj.__getstate__()

        obj2 = PerformancePeriod.__new__(PerformancePeriod)
        obj2.__setstate__(state)

        self.assertEqual(obj.to_dict(), obj2.to_dict())

    def test_blotter_serialization(self):

        obj = Blotter()
        state = obj.__getstate__()
        initargs = obj.__getinitargs__()

        obj2 = Blotter.__new__(Blotter)
        obj2.__init__(*initargs)
        obj2.__setstate__(state)

        self.assertEqual(obj.__repr__(), obj2.__repr__())

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_perf_tracker_serialization(self, emission_rate):

        sim_params = SimulationParameters(
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            10000,
            emission_rate=emission_rate)

        obj = PerformanceTracker(sim_params)
        state = obj.__getstate__()

        obj2 = PerformanceTracker.__new__(PerformanceTracker)
        obj2.__setstate__(state)

        self.assertEqual(obj.to_dict(), obj2.to_dict())

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_risk_metrics_cumulative_serialization(self, emission_rate):
        sim_params = SimulationParameters(
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            10000,
            emission_rate=emission_rate)

        obj = RiskMetricsCumulative(sim_params)
        state = obj.__getstate__()

        obj2 = RiskMetricsCumulative.__new__(RiskMetricsCumulative)
        obj2.__setstate__(state)

        self.assertEqual(obj.to_dict(), obj2.to_dict())

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_risk_metrics_period_serialization(self, emission_rate):

        sim_params = SimulationParameters(
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            10000,
            emission_rate=emission_rate)

        returns = factory.create_returns_from_list(
            [1.0], sim_params)
        obj = RiskMetricsPeriod(returns.index[0], returns.index[0], returns)
        state = obj.__getstate__()

        obj2 = RiskMetricsPeriod.__new__(RiskMetricsPeriod)
        obj2.__setstate__(state)

        self.assertEqual(obj.to_dict(), obj2.to_dict())

    @parameterized.expand([
        ('daily',),
        ('minute',),
    ])
    def test_risk_report_serialization(self, emission_rate):

        sim_params = SimulationParameters(
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
            10000,
            emission_rate=emission_rate)

        returns = factory.create_returns_from_list(
            [1.0], sim_params)
        obj = RiskReport(returns, sim_params)
        state = obj.__getstate__()

        obj2 = RiskReport.__new__(RiskReport)
        obj2.__setstate__(state)

        self.assertEqual(obj.to_dict(), obj2.to_dict())
