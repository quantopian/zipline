from zipline.finance.blotter import SimulatedBlotter
from zipline.finance.blotter.blotter import Blotter
from zipline.finance.blotter.core import _make_blotters_core
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_raises_str,
)
from zipline.utils.compat import mappingproxy


class BlotterCoreTestCase(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(BlotterCoreTestCase, self).init_instance_fixtures()

        self.blotters, self.register, self.unregister, self.load = (
            _make_blotters_core()
        )

        # make sure this starts empty
        assert_equal(self.blotters, mappingproxy({}))

    def test_load_not_registered(self):
        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

        # register in reverse order to test the sorting of the options
        self.register('c', SimulatedBlotter)
        self.register('b', SimulatedBlotter)
        self.register('a', SimulatedBlotter)

        msg = (
            "no blotter class registered as 'ayy-lmao', options are: "
            "['a', 'b', 'c']"
        )
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

    def test_register_decorator(self):

        @self.register('ayy-lmao')
        class ProperDummyBlotter(Blotter):

            def order(self, asset, amount, style, order_id=None):
                pass

            def cancel(self, order_id, relay_status=True):
                pass

            def cancel_all_orders_for_asset(self, asset, warn=False,
                                            relay_status=True):
                pass

            def execute_cancel_policy(self, event):
                pass

            def reject(self, order_id, reason=''):
                pass

            def hold(self, order_id, reason=''):
                pass

            def process_splits(self, splits):
                pass

            def get_transactions(self, bar_data):
                pass

            def prune_orders(self, closed_orders):
                pass

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(self.blotters, expected_blotters)
        assert_is(self.load('ayy-lmao'), ProperDummyBlotter)

        msg = "blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            @self.register('ayy-lmao')
            class Fake(object):
                pass

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            @self.register('something-different')
            class ImproperDummyBlotter(object):
                pass

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(self.blotters, expected_blotters)
        assert_is(self.load('ayy-lmao'), ProperDummyBlotter)

        self.unregister('ayy-lmao')
        assert_equal(self.blotters, mappingproxy({}))

        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

        msg = "blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.unregister('ayy-lmao')

    def test_register_non_decorator(self):

        class ProperDummyBlotter(Blotter):

            def order(self, asset, amount, style, order_id=None):
                pass

            def cancel(self, order_id, relay_status=True):
                pass

            def cancel_all_orders_for_asset(self, asset, warn=False,
                                            relay_status=True):
                pass

            def execute_cancel_policy(self, event):
                pass

            def reject(self, order_id, reason=''):
                pass

            def hold(self, order_id, reason=''):
                pass

            def process_splits(self, splits):
                pass

            def get_transactions(self, bar_data):
                pass

            def prune_orders(self, closed_orders):
                pass

        self.register('ayy-lmao', ProperDummyBlotter)

        expected_blotters = mappingproxy({'ayy-lmao': ProperDummyBlotter})
        assert_equal(self.blotters, expected_blotters)
        assert_is(self.load('ayy-lmao'), ProperDummyBlotter)

        msg = "blotter class 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, msg):
            class Fake(object):
                pass

            self.register('ayy-lmao', Fake)

        msg = "The class specified is not a subclass of Blotter"
        with assert_raises_str(TypeError, msg):
            class ImproperDummyBlotter(object):
                pass

            self.register('something-different', ImproperDummyBlotter)

        # ensure that the failed registration didn't break the previously
        # registered blotter
        assert_equal(self.blotters, expected_blotters)
        assert_is(self.load('ayy-lmao'), ProperDummyBlotter)

        self.unregister('ayy-lmao')
        assert_equal(self.blotters, mappingproxy({}))

        msg = "no blotter class registered as 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            self.load('ayy-lmao')

        msg = "blotter class 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            self.unregister('ayy-lmao')
