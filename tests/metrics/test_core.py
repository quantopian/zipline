from zipline.finance.metrics.core import _make_metrics_set_core
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
)
from zipline.utils.compat import mappingproxy


class MetricsSetCoreTestCase(ZiplineTestCase):
    def init_instance_fixtures(self):
        super(MetricsSetCoreTestCase, self).init_instance_fixtures()

        self.metrics_sets, self.register, self.unregister, self.load = (
            _make_metrics_set_core()
        )

        # make sure this starts empty
        assert_equal(self.metrics_sets, mappingproxy({}))

    def test_load_not_registered(self):
        msg = "no metrics set registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

        # register in reverse order to test the sorting of the options
        self.register('c', set)
        self.register('b', set)
        self.register('a', set)

        msg = (
            "no metrics set registered as 'ayy-lmao', options are: "
            "['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

    def test_register_decorator(self):
        ayy_lmao_set = set()

        @self.register('ayy-lmao')
        def ayy_lmao():
            return ayy_lmao_set

        expected_metrics_sets = mappingproxy({'ayy-lmao': ayy_lmao})
        assert_equal(self.metrics_sets, expected_metrics_sets)
        assert_is(self.load('ayy-lmao'), ayy_lmao_set)

        msg = "metrics set 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            @self.register('ayy-lmao')
            def other():  # pragma: no cover
                raise AssertionError('dead')

        # ensure that the failed registration didn't break the previously
        # registered set
        assert_equal(self.metrics_sets, expected_metrics_sets)
        assert_is(self.load('ayy-lmao'), ayy_lmao_set)

        self.unregister('ayy-lmao')
        assert_equal(self.metrics_sets, mappingproxy({}))

        msg = "no metrics set registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

        msg = "metrics set 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.unregister('ayy-lmao')

    def test_register_non_decorator(self):
        ayy_lmao_set = set()

        def ayy_lmao():
            return ayy_lmao_set

        self.register('ayy-lmao', ayy_lmao)

        expected_metrics_sets = mappingproxy({'ayy-lmao': ayy_lmao})
        assert_equal(self.metrics_sets, expected_metrics_sets)
        assert_is(self.load('ayy-lmao'), ayy_lmao_set)

        def other():  # pragma: no cover
            raise AssertionError('dead')

        msg = "metrics set 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            self.register('ayy-lmao', other)

        # ensure that the failed registration didn't break the previously
        # registered set
        assert_equal(self.metrics_sets, expected_metrics_sets)
        assert_is(self.load('ayy-lmao'), ayy_lmao_set)

        self.unregister('ayy-lmao')
        assert_equal(self.metrics_sets, mappingproxy({}))

        msg = "no metrics set registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

        msg = "metrics set 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.unregister('ayy-lmao')
