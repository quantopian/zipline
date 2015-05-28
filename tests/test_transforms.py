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
from datetime import timedelta
from functools import wraps
from itertools import product
from nose_parameterized import parameterized
import operator
import random
from six import itervalues
from six.moves import map
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from zipline.algorithm import TradingAlgorithm
import zipline.utils.factory as factory
from zipline.api import add_transform, get_datetime


def handle_data_wrapper(f):
    @wraps(f)
    def wrapper(context, data):
        dt = get_datetime()
        if dt.date() != context.current_date:
            context.warmup -= 1
            context.mins_for_days.append(1)
            context.current_date = dt.date()
        else:
            context.mins_for_days[-1] += 1

        hist = context.history(2, '1d', 'close_price')
        for n in (1, 2, 3):
            if n in data:
                if data[n].dt == dt:
                    context.vol_bars[n].append(data[n].volume)
                else:
                    context.vol_bars[n].append(0)

                context.price_bars[n].append(data[n].price)
            else:
                context.price_bars[n].append(np.nan)
                context.vol_bars[n].append(0)

            context.last_close_prices[n] = hist[n][0]

        if context.warmup < 0:
            return f(context, data)

    return wrapper


def initialize_with(test_case, tfm_name, days):
    def initalize(context):
        context.test_case = test_case
        context.days = days
        context.mins_for_days = []
        context.price_bars = (None, [np.nan], [np.nan], [np.nan])
        context.vol_bars = (None, [np.nan], [np.nan], [np.nan])
        if context.days:
            context.warmup = days + 1
        else:
            context.warmup = 2

        context.current_date = None

        context.last_close_prices = [np.nan, np.nan, np.nan, np.nan]
        add_transform(tfm_name, days)

    return initalize


def windows_with_frequencies(*args):
    args = args or (None,)
    return product(('daily', 'minute'), args)


def with_algo(f):
    name = f.__name__
    if not name.startswith('test_'):
        raise ValueError('This must decorate a test case')

    tfm_name = name[len('test_'):]

    @wraps(f)
    def wrapper(self, data_frequency, days=None):
        sim_params, source = self.sim_and_source[data_frequency]

        algo = TradingAlgorithm(
            initialize=initialize_with(self, tfm_name, days),
            handle_data=handle_data_wrapper(f),
            sim_params=sim_params,
            identifiers=[1, 2, 3]
        )
        algo.run(source)

    return wrapper


class TransformTestCase(TestCase):
    """
    Tests the simple transforms by running them through a zipline.
    """
    @classmethod
    def setUpClass(cls):
        random.seed(0)
        cls.sids = (1, 2, 3)

        minute_sim_ps = factory.create_simulation_parameters(
            num_days=3,
            sids=cls.sids,
            data_frequency='minute',
            emission_rate='minute',
        )
        daily_sim_ps = factory.create_simulation_parameters(
            num_days=30,
            sids=cls.sids,
            data_frequency='daily',
            emission_rate='daily',
        )
        cls.sim_and_source = {
            'minute': (minute_sim_ps, factory.create_minutely_trade_source(
                cls.sids,
                sim_params=minute_sim_ps,
            )),
            'daily': (daily_sim_ps, factory.create_trade_source(
                cls.sids,
                trade_time_increment=timedelta(days=1),
                sim_params=daily_sim_ps,
            )),
        }

    def tearDown(self):
        """
        Each test consumes a source, we need to rewind it.
        """
        for _, source in itervalues(self.sim_and_source):
            source.rewind()

    @parameterized.expand(windows_with_frequencies(1, 2, 3, 4))
    @with_algo
    def test_mavg(context, data):
        """
        Tests the mavg transform by manually keeping track of the prices
        in a naiive way and asserting that our mean is the same.
        """
        mins = sum(context.mins_for_days[-context.days:])

        for sid in data:
            assert_allclose(
                data[sid].mavg(context.days),
                np.mean(context.price_bars[sid][-mins:]),
            )

    @parameterized.expand(windows_with_frequencies(2, 3, 4))
    @with_algo
    def test_stddev(context, data):
        """
        Tests the stddev transform by manually keeping track of the prices
        in a naiive way and asserting that our stddev is the same.
        This accounts for the corrected ddof.
        """
        mins = sum(context.mins_for_days[-context.days:])

        for sid in data:
            assert_allclose(
                data[sid].stddev(context.days),
                np.std(context.price_bars[sid][-mins:], ddof=1),
            )

    @parameterized.expand(windows_with_frequencies(2, 3, 4))
    @with_algo
    def test_vwap(context, data):
        """
        Tests the vwap transform by manually keeping track of the prices
        and volumes in a naiive way and asserting that our hand-rolled vwap is
        the same
        """
        mins = sum(context.mins_for_days[-context.days:])
        for sid in data:
            prices = context.price_bars[sid][-mins:]
            vols = context.vol_bars[sid][-mins:]
            manual_vwap = sum(
                map(operator.mul, np.nan_to_num(np.array(prices)), vols),
            ) / sum(vols)

            assert_allclose(
                data[sid].vwap(context.days),
                manual_vwap,
            )

    @parameterized.expand(windows_with_frequencies())
    @with_algo
    def test_returns(context, data):
        for sid in data:
            last_close = context.last_close_prices[sid]
            returns = (data[sid].price - last_close) / last_close

            assert_allclose(
                data[sid].returns(),
                returns,
            )
