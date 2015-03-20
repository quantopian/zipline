#
# Copyright 2013 Quantopian, Inc.
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

import pytz
import numpy as np
import pandas as pd

from datetime import timedelta, datetime
from unittest import TestCase, skip

from zipline.utils.test_utils import setup_logger

import zipline.utils.factory as factory

from zipline.test_algorithms import TALIBAlgorithm

import talib
import zipline.transforms.ta as ta


class TestTALIB(TestCase):
    def setUp(self):
        setup_logger(self)
        sim_params = factory.create_simulation_parameters(
            start=datetime(1990, 1, 1, tzinfo=pytz.utc),
            end=datetime(1990, 3, 30, tzinfo=pytz.utc))
        self.source, self.panel = \
            factory.create_test_panel_ohlc_source(sim_params)

    @skip
    def test_talib_with_default_params(self):
        BLACKLIST = ['make_transform', 'BatchTransform',
                     # TODO: Figure out why MAVP generates a KeyError
                     'MAVP']
        names = [name for name in dir(ta)
                 if name[0].isupper() and name not in BLACKLIST]

        for name in names:
            print(name)
            zipline_transform = getattr(ta, name)(sid=0)
            talib_fn = getattr(talib.abstract, name)

            start = datetime(1990, 1, 1, tzinfo=pytz.utc)
            end = start + timedelta(days=zipline_transform.lookback + 10)
            sim_params = factory.create_simulation_parameters(
                start=start, end=end)
            source, panel = \
                factory.create_test_panel_ohlc_source(sim_params)

            algo = TALIBAlgorithm(talib=zipline_transform)
            algo.run(source)

            zipline_result = np.array(
                algo.talib_results[zipline_transform][-1])

            talib_data = dict()
            data = zipline_transform.window
            # TODO: Figure out if we are clobbering the tests by this
            # protection against empty windows
            if not data:
                continue
            for key in ['open', 'high', 'low', 'volume']:
                if key in data:
                    talib_data[key] = data[key][0].values
            talib_data['close'] = data['price'][0].values
            expected_result = talib_fn(talib_data)

            if isinstance(expected_result, list):
                expected_result = np.array([e[-1] for e in expected_result])
            else:
                expected_result = np.array(expected_result[-1])
            if not (np.all(np.isnan(zipline_result)) and
                    np.all(np.isnan(expected_result))):
                self.assertTrue(np.allclose(zipline_result, expected_result))
            else:
                print('--- NAN')

            # reset generator so next iteration has data
            # self.source, self.panel = \
                # factory.create_test_panel_ohlc_source(self.sim_params)

    def test_multiple_talib_with_args(self):
        zipline_transforms = [ta.MA(timeperiod=10),
                              ta.MA(timeperiod=25)]
        talib_fn = talib.abstract.MA
        algo = TALIBAlgorithm(talib=zipline_transforms)
        algo.run(self.source)
        # Test if computed values match those computed by pandas rolling mean.
        sid = 0
        talib_values = np.array([x[sid] for x in
                                 algo.talib_results[zipline_transforms[0]]])
        np.testing.assert_array_equal(talib_values,
                                      pd.rolling_mean(self.panel[0]['price'],
                                                      10).values)
        talib_values = np.array([x[sid] for x in
                                 algo.talib_results[zipline_transforms[1]]])
        np.testing.assert_array_equal(talib_values,
                                      pd.rolling_mean(self.panel[0]['price'],
                                                      25).values)
        for t in zipline_transforms:
            talib_result = np.array(algo.talib_results[t][-1])
            talib_data = dict()
            data = t.window
            # TODO: Figure out if we are clobbering the tests by this
            # protection against empty windows
            if not data:
                continue
            for key in ['open', 'high', 'low', 'volume']:
                if key in data:
                    talib_data[key] = data[key][0].values
            talib_data['close'] = data['price'][0].values
            expected_result = talib_fn(talib_data, **t.call_kwargs)[-1]
            np.testing.assert_allclose(talib_result, expected_result)

    def test_talib_with_minute_data(self):

        ma_one_day_minutes = ta.MA(timeperiod=10, bars='minute')

        # Assert that the BatchTransform window length is enough to cover
        # the amount of minutes in the timeperiod.

        # Here, 10 minutes only needs a window length of 1.
        self.assertEquals(1, ma_one_day_minutes.window_length)

        # With minutes greater than the 390, i.e. one trading day, we should
        # have a window_length of two days.
        ma_two_day_minutes = ta.MA(timeperiod=490, bars='minute')
        self.assertEquals(2, ma_two_day_minutes.window_length)

        # TODO: Ensure that the lookback into the datapanel is returning
        # expected results.
        # Requires supplying minute instead of day data to the unit test.
        # When adding test data, should add more minute events than the
        # timeperiod to ensure that lookback is behaving properly.
