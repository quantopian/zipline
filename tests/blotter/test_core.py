from zipline.finance.blotter import SimulatedBlotter
from zipline.finance.blotter.blotter_utils import BlotterClassDispatcher
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
)
from zipline.utils.compat import mappingproxy

bcd = BlotterClassDispatcher(blotter_factories={})


class BlotterCoreTestCase(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(BlotterCoreTestCase, self).init_instance_fixtures()
        bcd.clear_blotter_classes()
        assert_equal(bcd.blotter_factories, mappingproxy({}))

    def test_load_not_registered(self):
        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            bcd.get_blotter_class('ayy-lmao')

        # register in reverse order to test the sorting of the options
        bcd.register_blotter_class('c', SimulatedBlotter)
        bcd.register_blotter_class('b', SimulatedBlotter)
        bcd.register_blotter_class('a', SimulatedBlotter)

        msg = (
            "no blotter class registered as 'ayy-lmao', options are: "
            "['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            bcd.get_blotter_class('ayy-lmao')

    def test_register_decorator(self):

        @bcd.register_blotter_class('ayy-lmao')
        class ProperDummyBlotter(SimulatedBlotter):
            pass

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(bcd.blotter_factories, expected_blotters)
        assert_is(bcd.get_blotter_class('ayy-lmao'), ProperDummyBlotter)

        msg = "blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            @bcd.register_blotter_class('ayy-lmao')
            class Fake(object):
                pass

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            @bcd.register_blotter_class('something-different')
            class ImproperDummyBlotter(object):
                pass

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(bcd.blotter_factories, expected_blotters)
        assert_is(bcd.get_blotter_class('ayy-lmao'), ProperDummyBlotter)

        bcd.unregister_blotter_class('ayy-lmao')
        assert_equal(bcd.blotter_factories, mappingproxy({}))

        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            bcd.get_blotter_class('ayy-lmao')

        msg = "blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            bcd.unregister_blotter_class('ayy-lmao')

    def test_register_non_decorator(self):

        class ProperDummyBlotter(SimulatedBlotter):
            pass

        bcd.register_blotter_class('ayy-lmao', ProperDummyBlotter)

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(bcd.blotter_factories, expected_blotters)
        assert_is(bcd.get_blotter_class('ayy-lmao'), ProperDummyBlotter)

        msg = "blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            class Fake(object):
                pass

            bcd.register_blotter_class('ayy-lmao', Fake)

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            class ImproperDummyBlotter(object):
                pass

            bcd.register_blotter_class('something-different',
                                       ImproperDummyBlotter)

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(bcd.blotter_factories, expected_blotters)
        assert_is(bcd.get_blotter_class('ayy-lmao'), ProperDummyBlotter)

        bcd.unregister_blotter_class('ayy-lmao')
        assert_equal(bcd.blotter_factories, mappingproxy({}))

        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            bcd.get_blotter_class('ayy-lmao')

        msg = "blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            bcd.unregister_blotter_class('ayy-lmao')
