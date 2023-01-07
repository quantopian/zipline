import re
from collections import namedtuple

import pytest

from zipline.finance.metrics.core import _make_metrics_set_core
from zipline.testing.predicates import assert_equal
from zipline.utils.compat import mappingproxy


@pytest.fixture(scope="function")
def metrics():
    MetricsCoreSet = namedtuple(
        "MetricsCoreSet",
        [
            "metrics_sets",
            "register",
            "unregister",
            "load",
        ],
    )
    metrics_set_core = MetricsCoreSet(*_make_metrics_set_core())
    # make sure this starts empty
    assert metrics_set_core.metrics_sets == mappingproxy({})
    yield metrics_set_core


@pytest.mark.usefixtures("metrics")
class TestMetricsSetCore:
    def test_load_not_registered(self, metrics):
        msg = "no metrics set registered as 'ayy-lmao', options are: []"
        with pytest.raises(ValueError, match=re.escape(msg)):
            metrics.load("ayy-lmao")

        # register in reverse order to test the sorting of the options
        metrics.register("c", set)
        metrics.register("b", set)
        metrics.register("a", set)

        msg = "no metrics set registered as 'ayy-lmao', options are: " "['a', 'b', 'c']"
        with pytest.raises(ValueError, match=re.escape(msg)):
            metrics.load("ayy-lmao")

    def test_register_decorator(self, metrics):
        ayy_lmao_set = set()

        @metrics.register("ayy-lmao")
        def ayy_lmao():
            return ayy_lmao_set

        expected_metrics_sets = mappingproxy({"ayy-lmao": ayy_lmao})
        assert metrics.metrics_sets == expected_metrics_sets
        assert metrics.load("ayy-lmao") is ayy_lmao_set

        msg = "metrics set 'ayy-lmao' is already registered"
        with pytest.raises(ValueError, match=msg):

            @metrics.register("ayy-lmao")
            def other():  # pragma: no cover
                raise AssertionError("dead")

        # ensure that the failed registration didn't break the previously
        # registered set
        assert metrics.metrics_sets == expected_metrics_sets
        assert metrics.load("ayy-lmao") is ayy_lmao_set

        metrics.unregister("ayy-lmao")
        assert metrics.metrics_sets == mappingproxy({})

        msg = "no metrics set registered as 'ayy-lmao', options are: []"
        with pytest.raises(ValueError, match=re.escape(msg)):
            metrics.load("ayy-lmao")

        msg = "metrics set 'ayy-lmao' was not already registered"
        with pytest.raises(ValueError, match=msg):
            metrics.unregister("ayy-lmao")

    def test_register_non_decorator(self, metrics):
        ayy_lmao_set = set()

        def ayy_lmao():
            return ayy_lmao_set

        metrics.register("ayy-lmao", ayy_lmao)

        expected_metrics_sets = mappingproxy({"ayy-lmao": ayy_lmao})
        assert metrics.metrics_sets == expected_metrics_sets
        assert metrics.load("ayy-lmao") is ayy_lmao_set

        def other():  # pragma: no cover
            raise AssertionError("dead")

        msg = "metrics set 'ayy-lmao' is already registered"
        with pytest.raises(ValueError, match=msg):
            metrics.register("ayy-lmao", other)

        # ensure that the failed registration didn't break the previously
        # registered set
        assert metrics.metrics_sets == expected_metrics_sets
        assert metrics.load("ayy-lmao") is ayy_lmao_set

        metrics.unregister("ayy-lmao")
        assert_equal(metrics.metrics_sets, mappingproxy({}))

        msg = "no metrics set registered as 'ayy-lmao', options are: []"
        with pytest.raises(ValueError, match=re.escape(msg)):
            metrics.load("ayy-lmao")

        msg = "metrics set 'ayy-lmao' was not already registered"
        with pytest.raises(ValueError, match=msg):
            metrics.unregister("ayy-lmao")
