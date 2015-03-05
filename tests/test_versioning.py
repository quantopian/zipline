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
import os
import pandas
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

base_state_dir = 'tests/resources/saved_state_archive'

BASE_STATE_DIR = os.path.join(
    os.path.dirname(__file__),
    'resources',
    'saved_state_archive')


class VersioningTestCase(TestCase):

    def load_state_from_disk(self, cls):
        state_dir = cls.__module__ + '.' + cls.__name__

        full_dir = BASE_STATE_DIR + '/' + state_dir

        state_files = \
            [f for f in os.listdir(full_dir) if 'State_Version_' in f]

        for f_name in state_files:
            f = open(full_dir + '/' + f_name, 'r')
            yield pickle.load(f)

    # Only test versioning in minutely mode right now
    @parameterized.expand([
        (Blotter, (), 'repr'),
        (Order, (datetime.datetime(2013, 6, 19), 8554, 100), 'dict'),
        (PerShare, (), 'dict'),
        (PerTrade, (), 'dict'),
        (PerDollar, (), 'dict'),
        (PerformancePeriod, (10000,), 'to_dict'),
        (Position, (8554,), 'dict'),
        (PerformanceTracker, (sim_params_minute,), 'to_dict'),
        (RiskMetricsCumulative, (sim_params_minute,), 'to_dict'),
        (RiskMetricsPeriod,
            (returns.index[0], returns.index[0], returns), 'to_dict'),
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

        # The state generated under one version of pandas may not be
        # compatible with another. To ensure that tests pass under the travis
        # pandas version matrix, we only run versioning tests under the
        # current version of pandas. This will need to be updated once we
        # change the pandas version on prod.
        if pandas.__version__ != '0.12.0':
            return

        # Make reference object
        obj = cls(*initargs)

        # Fetch state
        state_versions = self.load_state_from_disk(cls)

        for version in state_versions:

            # For each version inflate a new object and ensure that it
            # matches the original.

            newargs = version['newargs']
            initargs = version['initargs']
            state = version['obj_state']

            if newargs is not None:
                obj2 = cls.__new__(cls, *newargs)
            else:
                obj2 = cls.__new__(cls)
            if initargs is not None:
                obj2.__init__(*initargs)
            obj2.__setstate__(state)

            # The ObjectId generated on instantiation of Order will
            # not be the same as the one loaded from saved state.
            if cls == Order:
                obj.__dict__['id'] = obj2.__dict__['id']

            if comparison_method == 'repr':
                self.assertEqual(obj.__repr__(), obj2.__repr__())
            elif comparison_method == 'to_dict':
                self.assertEqual(obj.to_dict(), obj2.to_dict())
            else:
                self.assertEqual(obj.__dict__, obj2.__dict__)
