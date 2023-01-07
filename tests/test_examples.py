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
import pytest
import warnings
from functools import partial
from itertools import combinations
from operator import itemgetter
import tarfile
from os import listdir
from os.path import dirname, join, realpath
import matplotlib
import pandas as pd
from zipline import examples
from zipline.data.bundles import register, unregister
from zipline.testing.fixtures import read_checked_in_benchmark_data
from zipline.testing.predicates import assert_equal
from zipline.utils.cache import dataframe_cache

TEST_RESOURCE_PATH = join(
    dirname(realpath(__file__)), "resources"  # zipline_repo/tests
)

PANDAS_VERSION = pd.__version__.replace(".", "-")

matplotlib.use("Agg")

EXAMPLE_MODULES = examples.load_example_modules()


def _no_benchmark_expectations_applied(expected_perf):
    # With no benchmark, expect zero results for these metrics:
    expected_perf[["alpha", "beta"]] = None
    for col in ["benchmark_period_return", "benchmark_volatility"]:
        expected_perf.loc[
            ~pd.isnull(expected_perf[col]),
            col,
        ] = 0.0
    return expected_perf


def _stored_pd_data(skip_vers=["0-18-1", "0-19-2", "0-22-0", "1-1-3", "1-2-3"]):
    with tarfile.open(join(TEST_RESOURCE_PATH, "example_data.tar.gz")) as tar:
        pd_versions = {
            n.split("/")[2]
            for n in tar.getnames()
            if "example_data/expected_perf/" in n
        }
        pd_versions = list(pd_versions)
    return sorted(list(filter(lambda x: x not in skip_vers, pd_versions)))


STORED_DATA_VERSIONS = _stored_pd_data()
COMBINED_DATA_VERSIONS = list(combinations(STORED_DATA_VERSIONS, 2))


@pytest.fixture(scope="class")
def _setup_class(request, tmpdir_factory):
    request.cls.tmp_path = tmpdir_factory.mktemp("tmp")
    request.cls.tmpdir = str(request.cls.tmp_path)
    register("test", lambda *args: None)

    with tarfile.open(join(TEST_RESOURCE_PATH, "example_data.tar.gz")) as tar:
        tar.extractall(request.cls.tmpdir)

    request.cls.expected_perf_dirs = listdir(
        join(
            str(request.cls.tmp_path),
            "example_data",
            "expected_perf",
        )
    )

    if PANDAS_VERSION not in request.cls.expected_perf_dirs:
        warnings.warn(
            "No data stored matches the current version of pandas. "
            "Consider including the data using rebuild_example_data",
        )

    yield
    partial(unregister, "test")


@pytest.fixture(scope="function")
def _df_cache(_setup_class, request):
    request.cls.expected_perf = dataframe_cache(
        join(
            str(request.cls.tmp_path),
            "example_data",
            f"expected_perf/{request.param}",
        ),
        serialization="pickle",
    )

    request.cls.no_benchmark_expected_perf = {
        example_name: _no_benchmark_expectations_applied(expected_perf.copy())
        for example_name, expected_perf in request.cls.expected_perf.items()
    }


@pytest.mark.usefixtures("_setup_class", "_df_cache")
class TestsExamplesTests:

    # some columns contain values with unique ids that will not be the same

    @pytest.mark.filterwarnings("ignore: Matplotlib is currently using agg")
    @pytest.mark.parametrize(
        "benchmark_returns", [read_checked_in_benchmark_data(), None]
    )
    @pytest.mark.parametrize("example_name", sorted(EXAMPLE_MODULES))
    @pytest.mark.parametrize("_df_cache", STORED_DATA_VERSIONS, indirect=True)
    def test_example(self, example_name, benchmark_returns):
        actual_perf = examples.run_example(
            EXAMPLE_MODULES,
            example_name,
            # This should match the invocation in
            # zipline/tests/resources/rebuild_example_data
            environ={
                "ZIPLINE_ROOT": join(self.tmpdir, "example_data", "root"),
            },
            benchmark_returns=benchmark_returns,
        )
        if benchmark_returns is not None:
            expected_perf = self.expected_perf[example_name]
        else:
            expected_perf = self.no_benchmark_expected_perf[example_name]

        # Exclude positions column as the positions do not always have the
        # same order
        columns = [
            column for column in examples._cols_to_check if column != "positions"
        ]
        assert_equal(
            actual_perf[columns],
            expected_perf[columns],
            # There is a difference in the datetime columns in pandas
            # 0.16 and 0.17 because in 16 they are object and in 17 they are
            # datetime[ns, UTC]. We will just ignore the dtypes for now.
            # check_dtype=False,
        )
        # Sort positions by SID before comparing
        assert_equal(
            expected_perf["positions"].apply(sorted, key=itemgetter("sid")),
            actual_perf["positions"].apply(sorted, key=itemgetter("sid")),
        )


@pytest.mark.usefixtures("_setup_class")
class TestsStoredDataCheck:
    def expected_perf(self, pd_version):
        return dataframe_cache(
            join(
                str(self.tmp_path),
                "example_data",
                f"expected_perf/{pd_version}",
            ),
            serialization="pickle",
        )

    @pytest.mark.parametrize(
        "benchmark_returns", [read_checked_in_benchmark_data(), None]
    )
    @pytest.mark.parametrize("example_name", sorted(EXAMPLE_MODULES))
    @pytest.mark.parametrize("pd_versions", COMBINED_DATA_VERSIONS, ids=str)
    def test_compare_stored_data(self, example_name, benchmark_returns, pd_versions):

        if benchmark_returns is not None:
            expected_perf_a = self.expected_perf(pd_versions[0])[example_name]
            expected_perf_b = self.expected_perf(pd_versions[1])[example_name]
        else:
            expected_perf_a = {
                example_name: _no_benchmark_expectations_applied(expected_perf.copy())
                for example_name, expected_perf in self.expected_perf(
                    pd_versions[0]
                ).items()
            }[example_name]
            expected_perf_b = {
                example_name: _no_benchmark_expectations_applied(expected_perf.copy())
                for example_name, expected_perf in self.expected_perf(
                    pd_versions[1]
                ).items()
            }[example_name]

        # Exclude positions column as the positions do not always have the
        # same order
        columns = [
            column for column in examples._cols_to_check if column != "positions"
        ]

        assert_equal(
            expected_perf_a[columns],
            expected_perf_b[columns],
        )
        # Sort positions by SID before comparing
        assert_equal(
            expected_perf_a["positions"].apply(sorted, key=itemgetter("sid")),
            expected_perf_b["positions"].apply(sorted, key=itemgetter("sid")),
        )
