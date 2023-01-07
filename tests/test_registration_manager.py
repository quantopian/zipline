from zipline.extensions import Registry
import pytest
import re


class FakeInterface:
    pass


class TestRegistrationManager:
    def test_load_not_registered(self):
        rm = Registry(FakeInterface)

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao',"
            " options are: []"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            rm.load("ayy-lmao")

        # register in reverse order to test the sorting of the options
        rm.register("c", FakeInterface)
        rm.register("b", FakeInterface)
        rm.register("a", FakeInterface)

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: ['a', 'b', 'c']"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            rm.load("ayy-lmao")

    def test_register_decorator(self):
        rm = Registry(FakeInterface)

        @rm.register("ayy-lmao")
        class ProperDummyInterface(FakeInterface):
            pass

        def check_registered():
            assert rm.is_registered(
                "ayy-lmao"
            ), "Class ProperDummyInterface wasn't properly registered under \n name 'ayy-lmao'"

            assert isinstance(rm.load("ayy-lmao"), ProperDummyInterface)

        # Check that we successfully registered.
        check_registered()

        # Try and fail to register with the same key again.
        msg = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with pytest.raises(ValueError, match=msg):

            @rm.register("ayy-lmao")
            class Fake:
                pass

        # assert excinfo.value.args == msg
        # check that the failed registration didn't break the previous
        # registration
        check_registered()

        # Unregister the key and assert that the key is now gone.
        rm.unregister("ayy-lmao")

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: []"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            rm.load("ayy-lmao")

        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with pytest.raises(ValueError, match=msg):
            rm.unregister("ayy-lmao")

    def test_register_non_decorator(self):
        rm = Registry(FakeInterface)

        class ProperDummyInterface(FakeInterface):
            pass

        rm.register("ayy-lmao", ProperDummyInterface)

        def check_registered():
            assert rm.is_registered(
                "ayy-lmao"
            ), "Class ProperDummyInterface wasn't properly registered under name 'ayy-lmao'"
            assert isinstance(rm.load("ayy-lmao"), ProperDummyInterface)

        # Check that we successfully registered.
        check_registered()

        class Fake:
            pass

        # Try and fail to register with the same key again.
        msg = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with pytest.raises(ValueError, match=msg):
            rm.register("ayy-lmao", Fake)

        # check that the failed registration didn't break the previous
        # registration
        check_registered()

        rm.unregister("ayy-lmao")

        msg = (
            "no FakeInterface factory registered under name 'ayy-lmao', "
            "options are: []"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            rm.load("ayy-lmao")

        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with pytest.raises(ValueError, match=msg):
            rm.unregister("ayy-lmao")
