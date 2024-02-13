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
from operator import itemgetter
import tarfile

import matplotlib
import pandas as pd

from zipline import examples
from zipline.data.bundles import register, unregister
from zipline.testing import test_resource_path, parameter_space
from zipline.testing.fixtures import (
    WithTmpDir,
    ZiplineTestCase,
    read_checked_in_benchmark_data,
)
from zipline.testing.predicates import assert_equal
from zipline.utils.cache import dataframe_cache


# Otherwise the next line sometimes complains about being run too late.
_multiprocess_can_split_ = False

matplotlib.use('Agg')

EXAMPLE_MODULES = examples.load_example_modules()


class ExamplesTests(WithTmpDir, ZiplineTestCase):
    # some columns contain values with unique ids that will not be the same

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

        cls.no_benchmark_expected_perf = {
            example_name: cls._no_benchmark_expectations_applied(
                expected_perf.copy()
            )
            for example_name, expected_perf in cls.expected_perf.items()
        }

    @staticmethod
    def _no_benchmark_expectations_applied(expected_perf):
        # With no benchmark, expect zero results for these metrics:
        expected_perf[['alpha', 'beta']] = None
        for col in ['benchmark_period_return', 'benchmark_volatility']:
            expected_perf.loc[
                ~pd.isnull(expected_perf[col]),
                col,
            ] = 0.0
        return expected_perf

    @parameter_space(
        example_name=sorted(EXAMPLE_MODULES),
        benchmark_returns=[read_checked_in_benchmark_data(), None]
    )
    def test_example(self, example_name, benchmark_returns):
        actual_perf = examples.run_example(
            EXAMPLE_MODULES,
            example_name,
            # This should match the invocation in
            # zipline/tests/resources/rebuild_example_data
            environ={
                'ZIPLINE_ROOT': self.tmpdir.getpath('example_data/root'),
            },
            benchmark_returns=benchmark_returns,
        )
        if benchmark_returns is not None:
            expected_perf = self.expected_perf[example_name]
        else:
            expected_perf = self.no_benchmark_expected_perf[example_name]

        # Exclude positions column as the positions do not always have the
        # same order
        columns = [column for column in examples._cols_to_check
                   if column != 'positions']
        assert_equal(
            actual_perf[columns],
            expected_perf[columns],
            # There is a difference in the datetime columns in pandas
            # 0.16 and 0.17 because in 16 they are object and in 17 they are
            # datetime[ns, UTC]. We will just ignore the dtypes for now.
            check_dtype=False,
        )
        # Sort positions by SID before comparing
        assert_equal(
            expected_perf['positions'].apply(sorted, key=itemgetter('sid')),
            actual_perf['positions'].apply(sorted, key=itemgetter('sid')),
        )
