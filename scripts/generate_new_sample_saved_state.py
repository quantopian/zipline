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
import logbook
import os
import pickle
import pytz

import sys
sys.path.insert(0, '.')  # noqa

from zipline.finance.blotter import Blotter, Order
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance.performance.period import PerformancePeriod
from zipline.finance.performance.position import Position
from zipline.finance.performance.position_tracker import PositionTracker
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
from zipline.utils.serialization_utils import VERSION_LABEL

base_state_dir = 'tests/resources/saved_state_archive'
if not os.path.exists(base_state_dir):
    os.makedirs(base_state_dir)


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


argument_list = [
    (Blotter, ()),
    (Order, (datetime.datetime(2013, 6, 19), 8554, 100)),
    (PerShare, ()),
    (PerTrade, ()),
    (PerDollar, ()),
    (PerformancePeriod, (10000,)),
    (Position, (8554,)),
    (PositionTracker, ()),
    (PerformanceTracker, (sim_params_minute,)),
    (RiskMetricsCumulative, (sim_params_minute,)),
    (RiskMetricsPeriod, (returns.index[0], returns.index[0], returns)),
    (RiskReport, (returns, sim_params_minute)),
    (FixedSlippage, ()),
    (Transaction, (8554, 10, datetime.datetime(2013, 6, 19), 100, "0000")),
    (VolumeShareSlippage, ()),
    (Account, ()),
    (Portfolio, ()),
    (ProtocolPosition, (8554,))
]


def write_state_to_disk(cls, state, emission_rate=None):
    state_dir = cls.__module__ + '.' + cls.__name__

    full_dir = base_state_dir + '/' + state_dir

    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    if emission_rate is not None:
        name = 'State_Version_' + emission_rate + \
            str(state['obj_state'][VERSION_LABEL])
    else:
        name = 'State_Version_' + str(state['obj_state'][VERSION_LABEL])

    full_path = full_dir + '/' + name

    f = open(full_path, 'w')

    pickle.dump(state, f)

    f.close()


def generate_object_state(cls, initargs):

    obj = cls(*initargs)
    state = obj.__getstate__()
    if hasattr(obj, '__getinitargs__'):
        initargs = obj.__getinitargs__()
    else:
        initargs = None
    if hasattr(obj, '__getnewargs__'):
        newargs = obj.__getnewargs__()
    else:
        newargs = None

    on_disk_state = {
        'obj_state': state,
        'initargs': initargs,
        'newargs': newargs
    }

    write_state_to_disk(cls, on_disk_state)


if __name__ == "__main__":
    logbook.StderrHandler().push_application()

    for args in argument_list:
        generate_object_state(*args)
