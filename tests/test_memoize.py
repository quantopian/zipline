"""
Tests for zipline.utils.memoize.
"""
from unittest import TestCase

from zipline.utils.memoize import remember_last


class TestRememberLast(TestCase):

    def test_remember_last(self):

        # Store the count in a list so we can mutate it from inside `func`.
        call_count = [0]

        @remember_last
        def func(x):
            call_count[0] += 1
            return x

        self.assertEqual((func(1), call_count[0]), (1, 1))

        # Calling again with the same argument should just re-use the old
        # value, which means func shouldn't get called again.
        self.assertEqual((func(1), call_count[0]), (1, 1))
        self.assertEqual((func(1), call_count[0]), (1, 1))

        # Calling with a new value should increment the counter.
        self.assertEqual((func(2), call_count[0]), (2, 2))
        self.assertEqual((func(2), call_count[0]), (2, 2))

        # Calling the old value should still increment the counter.
        self.assertEqual((func(1), call_count[0]), (1, 3))
        self.assertEqual((func(1), call_count[0]), (1, 3))
