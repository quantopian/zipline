from zipline.extensions import Registry
from zipline.finance.blotter import SimulationBlotter, Blotter
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
    assert_true,
)


class RegistrationManagerTestCase(ZiplineTestCase):

    def test_load_not_registered(self):
        rm = Registry(Blotter)
        assert_equal(rm.get_registered_classes(), {})

        msg = (
            "no Blotter class registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        # register in reverse order to test the sorting of the options
        rm.register('c', SimulationBlotter)
        rm.register('b', SimulationBlotter)
        rm.register('a', SimulationBlotter)

        msg = (
            "no Blotter class registered under name 'ayy-lmao', options are: "
            "['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

    def test_register_decorator(self):
        rm = Registry(Blotter)
        assert_equal(rm.get_registered_classes(), {})

        @rm.register('ayy-lmao')
        class ProperDummyBlotter(SimulationBlotter):
            pass

        expected_blotters = {'ayy-lmao': ProperDummyBlotter}
        assert_equal(rm.get_registered_classes(), expected_blotters)
        assert_is(rm.load('ayy-lmao'), ProperDummyBlotter)
        assert_true(rm.class_registered('ayy-lmao'))

        msg = "Blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            @rm.register('ayy-lmao')
            class Fake(object):
                pass

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            @rm.register('something-different')
            class ImproperDummyBlotter(object):
                pass

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(rm.get_registered_classes(), expected_blotters)
        assert_is(rm.load('ayy-lmao'), ProperDummyBlotter)

        rm.unregister('ayy-lmao')
        assert_equal(rm.get_registered_classes(), {})

        msg = (
            "no Blotter class registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        msg = "Blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')

    def test_register_non_decorator(self):
        rm = Registry(Blotter)
        assert_equal(rm.get_registered_classes(), {})

        class ProperDummyBlotter(SimulationBlotter):
            pass

        rm.register('ayy-lmao', ProperDummyBlotter)

        expected_blotters = {'ayy-lmao': ProperDummyBlotter}
        assert_equal(rm.get_registered_classes(), expected_blotters)
        assert_is(rm.load('ayy-lmao'), ProperDummyBlotter)
        assert_true(rm.class_registered('ayy-lmao'))

        class Fake(object):
            pass

        msg = "Blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            rm.register('ayy-lmao', Fake)

        class ImproperDummyBlotter(object):
            pass

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            rm.register('something-different', ImproperDummyBlotter)

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(rm.get_registered_classes(), expected_blotters)
        assert_is(rm.load('ayy-lmao'), ProperDummyBlotter)

        rm.unregister('ayy-lmao')
        assert_equal(rm.get_registered_classes(), {})

        msg = (
            "no Blotter class registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        msg = "Blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')
