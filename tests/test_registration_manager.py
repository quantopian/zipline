from zipline.extensions import RegistrationManager
from zipline.finance.blotter import SimulatedBlotter, Blotter
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
    assert_true,
)
from zipline.utils.compat import mappingproxy


class RegistrationManagerTestCase(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(RegistrationManagerTestCase, self).init_instance_fixtures()
        self.rm = RegistrationManager(Blotter)
        assert_equal(self.rm.get_registered_classes(), mappingproxy({}))

    def test_load_not_registered(self):
        msg = (
            "no Blotter class registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            self.rm.load('ayy-lmao')

        # register in reverse order to test the sorting of the options
        self.rm.register('c', SimulatedBlotter)
        self.rm.register('b', SimulatedBlotter)
        self.rm.register('a', SimulatedBlotter)

        msg = (
            "no Blotter class registered under name 'ayy-lmao', options are: "
            "['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            self.rm.load('ayy-lmao')

    def test_register_decorator(self):

        @self.rm.register('ayy-lmao')
        class ProperDummyBlotter(SimulatedBlotter):
            pass

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(self.rm.get_registered_classes(), expected_blotters)
        assert_is(self.rm.load('ayy-lmao'), ProperDummyBlotter)
        assert_true(self.rm.class_exists('ayy-lmao'))

        msg = "Blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            @self.rm.register('ayy-lmao')
            class Fake(object):
                pass

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            @self.rm.register('something-different')
            class ImproperDummyBlotter(object):
                pass

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(self.rm.get_registered_classes(), expected_blotters)
        assert_is(self.rm.load('ayy-lmao'), ProperDummyBlotter)

        self.rm.unregister('ayy-lmao')
        assert_equal(self.rm.get_registered_classes(), mappingproxy({}))

        msg = (
            "no Blotter class registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            self.rm.load('ayy-lmao')

        msg = "Blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.rm.unregister('ayy-lmao')

    def test_register_non_decorator(self):

        class ProperDummyBlotter(SimulatedBlotter):
            pass

        self.rm.register('ayy-lmao', ProperDummyBlotter)

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(self.rm.get_registered_classes(), expected_blotters)
        assert_is(self.rm.load('ayy-lmao'), ProperDummyBlotter)
        assert_true(self.rm.class_exists('ayy-lmao'))

        msg = "Blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            class Fake(object):
                pass

            self.rm.register('ayy-lmao', Fake)

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            class ImproperDummyBlotter(object):
                pass

            self.rm.register('something-different', ImproperDummyBlotter)

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(self.rm.get_registered_classes(), expected_blotters)
        assert_is(self.rm.load('ayy-lmao'), ProperDummyBlotter)

        self.rm.unregister('ayy-lmao')
        assert_equal(self.rm.get_registered_classes(), mappingproxy({}))

        msg = (
            "no Blotter class registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            self.rm.load('ayy-lmao')

        msg = "Blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.rm.unregister('ayy-lmao')
