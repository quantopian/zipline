from zipline.extensions import Registry
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
    assert_true,
)


class FakeInterface():
    pass


class RegistrationManagerTestCase(ZiplineTestCase):

    def test_load_not_registered(self):
        rm = Registry(FakeInterface)
        assert_equal(rm.get_registered_classes(), {})

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        # register in reverse order to test the sorting of the options
        rm.register('c', FakeInterface)
        rm.register('b', FakeInterface)
        rm.register('a', FakeInterface)

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: ['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

    def test_register_decorator(self):
        rm = Registry(FakeInterface)
        assert_equal(rm.get_registered_classes(), {})

        @rm.register('ayy-lmao')
        class ProperDummyInterface(FakeInterface):
            pass

        expected_classes = {'ayy-lmao': ProperDummyInterface}
        assert_equal(rm.get_registered_classes(), expected_classes)
        assert_is(rm.load('ayy-lmao'), ProperDummyInterface)
        assert_true(
            rm.class_registered('ayy-lmao'),
            "Class ProperDummyInterface wasn't properly registered under"
            "name 'ayy-lmao'"
        )

        m = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, m):
            @rm.register('ayy-lmao')
            class Fake(object):
                pass

        # ensure that the failed registration didn't break the previously
        # registered interface class
        assert_equal(rm.get_registered_classes(), expected_classes)
        assert_is(rm.load('ayy-lmao'), ProperDummyInterface)

        rm.unregister('ayy-lmao')
        assert_equal(rm.get_registered_classes(), {})

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')

    def test_register_non_decorator(self):
        rm = Registry(FakeInterface)
        assert_equal(rm.get_registered_classes(), {})

        class ProperDummyInterface(FakeInterface):
            pass

        rm.register('ayy-lmao', ProperDummyInterface)

        expected_classes = {'ayy-lmao': ProperDummyInterface}
        assert_equal(rm.get_registered_classes(), expected_classes)
        assert_is(rm.load('ayy-lmao'), ProperDummyInterface)
        assert_true(
            rm.class_registered('ayy-lmao'),
            "Class ProperDummyInterface wasn't properly registered under"
            "name 'ayy-lmao'"
        )

        class Fake(object):
            pass

        m = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, m):
            rm.register('ayy-lmao', Fake)

        class ImproperDummyInterface(object):
            pass

        # ensure that the failed registration didn't break the previously
        # registered interface class
        assert_equal(rm.get_registered_classes(), expected_classes)
        assert_is(rm.load('ayy-lmao'), ProperDummyInterface)

        rm.unregister('ayy-lmao')
        assert_equal(rm.get_registered_classes(), {})

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')
