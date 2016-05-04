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
from functools import partial
import tarfile

import matplotlib
from nose_parameterized import parameterized
import pandas as pd

from zipline import examples, run_algorithm
from zipline.data.bundles import register, unregister
from zipline.testing import test_resource_path
from zipline.testing.fixtures import WithTmpDir, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.cache import dataframe_cache

# Otherwise the next line sometimes complains about being run too late.
_multiprocess_can_split_ = False

matplotlib.use('Agg')


class ExamplesTests(WithTmpDir, ZiplineTestCase):
    # some columns contain values with unique ids that will not be the same
    cols_to_check = [
        'algo_volatility',
        'algorithm_period_return',
        'alpha',
        'benchmark_period_return',
        'benchmark_volatility',
        'beta',
        'capital_used',
        'ending_cash',
        'ending_exposure',
        'ending_value',
        'excess_return',
        'gross_leverage',
        'long_exposure',
        'long_value',
        'longs_count',
        'max_drawdown',
        'max_leverage',
        'net_leverage',
        'period_close',
        'period_label',
        'period_open',
        'pnl',
        'portfolio_value',
        'positions',
        'returns',
        'short_exposure',
        'short_value',
        'shorts_count',
        'sortino',
        'starting_cash',
        'starting_exposure',
        'starting_value',
        'trading_days',
        'treasury_period_return',
    ]

    @classmethod
    def init_class_fixtures(cls):
        super(ExamplesTests, cls).init_class_fixtures()

        register('test', lambda *args: None)
        cls.add_class_callback(partial(unregister, 'test'))

        with tarfile.open(test_resource_path('example_data.tar.gz')) as tar:
            tar.extractall(cls.tmpdir.path)

        cls.expected_perf = dataframe_cache(
            cls.tmpdir.getpath(
                'example_data/expected_perf/%s' %
                pd.__version__.replace('.', '-'),
            ),
            serialization='pickle',
        )

    @parameterized.expand(e for e in dir(examples) if not e.startswith('_'))
    def test_example(self, example):
        mod = getattr(examples, example)
        actual_perf = run_algorithm(
            handle_data=mod.handle_data,
            initialize=mod.initialize,
            before_trading_start=getattr(mod, 'before_trading_start', None),
            analyze=getattr(mod, 'analyze', None),
            bundle='test',
            environ={
                'ZIPLINE_ROOT': self.tmpdir.getpath('example_data/root'),
            },
            capital_base=1e7,
            **mod._test_args()
        )
        assert_equal(
            actual_perf[self.cols_to_check],
            self.expected_perf[example][self.cols_to_check],
            # There is a difference in the datetime columns in pandas
            # 0.16 and 0.17 because in 16 they are object and in 17 they are
            # datetime[ns, UTC]. We will just ignore the dtypes for now.
            check_dtype=False,
        )
