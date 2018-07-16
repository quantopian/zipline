from zipline.extensions import Registry
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import assert_raises_str, assert_true


class FakeInterface(object):
    pass


class RegistrationManagerTestCase(ZiplineTestCase):

    def test_load_not_registered(self):
        rm = Registry(FakeInterface)

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

        @rm.register('ayy-lmao')
        class ProperDummyInterface(FakeInterface):
            pass

        def check_registered():
            assert_true(
                rm.is_registered('ayy-lmao'),
                "Class ProperDummyInterface wasn't properly registered under"
                "name 'ayy-lmao'"
            )
            self.assertIsInstance(rm.load('ayy-lmao'), ProperDummyInterface)

        # Check that we successfully registered.
        check_registered()

        # Try and fail to register with the same key again.
        m = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, m):
            @rm.register('ayy-lmao')
            class Fake(object):
                pass

        # check that the failed registration didn't break the previous
        # registration
        check_registered()

        # Unregister the key and assert that the key is now gone.
        rm.unregister('ayy-lmao')

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

        class ProperDummyInterface(FakeInterface):
            pass

        rm.register('ayy-lmao', ProperDummyInterface)

        def check_registered():
            assert_true(
                rm.is_registered('ayy-lmao'),
                "Class ProperDummyInterface wasn't properly registered under"
                "name 'ayy-lmao'"
            )
            self.assertIsInstance(rm.load('ayy-lmao'), ProperDummyInterface)

        # Check that we successfully registered.
        check_registered()

        class Fake(object):
            pass

        # Try and fail to register with the same key again.
        m = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, m):
            rm.register('ayy-lmao', Fake)

        # check that the failed registration didn't break the previous
        # registration
        check_registered()

        rm.unregister('ayy-lmao')

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: []"
        )
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')
