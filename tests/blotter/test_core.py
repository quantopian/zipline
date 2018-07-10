from zipline.finance.blotter import SimulatedBlotter
from zipline.finance.blotter.core import BlotterClassDispatcher
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
    assert_true,
)
from zipline.utils.compat import mappingproxy


class BlotterCoreTestCase(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(BlotterCoreTestCase, self).init_instance_fixtures()
        self.bcd = BlotterClassDispatcher(classes={})
        assert_equal(self.bcd.blotter_factories, mappingproxy({}))

    def test_load_not_registered(self):
        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.bcd.load('ayy-lmao')

        # register in reverse order to test the sorting of the options
        self.bcd.register('c', SimulatedBlotter)
        self.bcd.register('b', SimulatedBlotter)
        self.bcd.register('a', SimulatedBlotter)

        msg = (
            "no blotter class registered as 'ayy-lmao', options are: "
            "['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            self.bcd.load('ayy-lmao')

    def test_register_decorator(self):

        @self.bcd.register('ayy-lmao')
        class ProperDummyBlotter(SimulatedBlotter):
            pass

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(self.bcd.blotter_factories, expected_blotters)
        assert_is(self.bcd.load('ayy-lmao'), ProperDummyBlotter)
        assert_true(self.bcd.class_exists('ayy-lmao'))

        msg = "blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            @self.bcd.register('ayy-lmao')
            class Fake(object):
                pass

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            @self.bcd.register('something-different')
            class ImproperDummyBlotter(object):
                pass

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(self.bcd.blotter_factories, expected_blotters)
        assert_is(self.bcd.load('ayy-lmao'), ProperDummyBlotter)

        self.bcd.unregister('ayy-lmao')
        assert_equal(self.bcd.blotter_factories, mappingproxy({}))

        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.bcd.load('ayy-lmao')

        msg = "blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.bcd.unregister('ayy-lmao')

    def test_register_non_decorator(self):

        class ProperDummyBlotter(SimulatedBlotter):
            pass

        self.bcd.register('ayy-lmao', ProperDummyBlotter)

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(self.bcd.blotter_factories, expected_blotters)
        assert_is(self.bcd.load('ayy-lmao'), ProperDummyBlotter)
        assert_true(self.bcd.class_exists('ayy-lmao'))

        msg = "blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            class Fake(object):
                pass

            self.bcd.register('ayy-lmao', Fake)

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            class ImproperDummyBlotter(object):
                pass

            self.bcd.register('something-different', ImproperDummyBlotter)

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(self.bcd.blotter_factories, expected_blotters)
        assert_is(self.bcd.load('ayy-lmao'), ProperDummyBlotter)

        self.bcd.unregister('ayy-lmao')
        assert_equal(self.bcd.blotter_factories, mappingproxy({}))

        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.bcd.load('ayy-lmao')

        msg = "blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.bcd.unregister('ayy-lmao')
