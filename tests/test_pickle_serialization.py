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
import pickle
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

sim_params_daily = SimulationParameters(
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    10000,
    emission_rate='daily')
sim_params_minute = SimulationParameters(
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    datetime.datetime(2013, 6, 19, tzinfo=pytz.UTC),
    10000,
    emission_rate='minute')
returns = factory.create_returns_from_list(
    [1.0], sim_params_daily)


class PickleSerializationTestCase(TestCase):

    @parameterized.expand([
        (Blotter, (), 'repr'),
        (Order, (datetime.datetime(2013, 6, 19), 8554, 100), 'dict'),
        (PerShare, (), 'dict'),
        (PerTrade, (), 'dict'),
        (PerDollar, (), 'dict'),
        (PerformancePeriod, (10000,), 'to_dict'),
        (Position, (8554,), 'dict'),
        (PerformanceTracker, (sim_params_daily,), 'to_dict'),
        (PerformanceTracker, (sim_params_minute,), 'to_dict'),
        (RiskMetricsCumulative, (sim_params_daily,), 'to_dict'),
        (RiskMetricsCumulative, (sim_params_minute,), 'to_dict'),
        (RiskMetricsPeriod,
            (returns.index[0], returns.index[0], returns), 'to_dict'),
        (RiskReport, (returns, sim_params_daily), 'to_dict'),
        (RiskReport, (returns, sim_params_minute), 'to_dict'),
        (FixedSlippage, (), 'dict'),
        (Transaction,
            (8554, 10, datetime.datetime(2013, 6, 19), 100, "0000"), 'dict'),
        (VolumeShareSlippage, (), 'dict'),
        (Account, (), 'dict'),
        (Portfolio, (), 'dict'),
        (ProtocolPosition, (8554,), 'dict')
    ])
    def test_object_serialization(self,
                                  cls,
                                  initargs,
                                  comparison_method='dict'):

        obj = cls(*initargs)
        state = pickle.dumps(obj)

        obj2 = pickle.loads(state)

        if comparison_method == 'repr':
            self.assertEqual(obj.__repr__(), obj2.__repr__())
        elif comparison_method == 'to_dict':
            self.assertEqual(obj.to_dict(), obj2.to_dict())
        else:
            self.assertEqual(obj.__dict__, obj2.__dict__)
