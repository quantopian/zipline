"""
Tests for zipline.utils.memoize.
"""
from collections import defaultdict
import gc
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

    def test_remember_last_method(self):
        call_count = defaultdict(int)

        class clz(object):
            @remember_last
            def func(self, x):
                call_count[(self, x)] += 1
                return x

        inst1 = clz()
        self.assertEqual((inst1.func(1), call_count), (1, {(inst1, 1): 1}))

        # Calling again with the same argument should just re-use the old
        # value, which means func shouldn't get called again.
        self.assertEqual((inst1.func(1), call_count), (1, {(inst1, 1): 1}))

        # Calling with a new value should increment the counter.
        self.assertEqual((inst1.func(2), call_count), (2, {(inst1, 1): 1,
                                                           (inst1, 2): 1}))
        self.assertEqual((inst1.func(2), call_count), (2, {(inst1, 1): 1,
                                                           (inst1, 2): 1}))

        # Calling the old value should still increment the counter.
        self.assertEqual((inst1.func(1), call_count), (1, {(inst1, 1): 2,
                                                           (inst1, 2): 1}))
        self.assertEqual((inst1.func(1), call_count), (1, {(inst1, 1): 2,
                                                           (inst1, 2): 1}))

        inst2 = clz()
        self.assertEqual((inst2.func(1), call_count),
                         (1, {(inst1, 1): 2, (inst1, 2): 1,
                              (inst2, 1): 1}))
        self.assertEqual((inst2.func(1), call_count),
                         (1, {(inst1, 1): 2, (inst1, 2): 1,
                              (inst2, 1): 1}))

        self.assertEqual((inst2.func(2), call_count),
                         (2, {(inst1, 1): 2, (inst1, 2): 1,
                              (inst2, 1): 1, (inst2, 2): 1}))
        self.assertEqual((inst2.func(2), call_count),
                         (2, {(inst1, 1): 2, (inst1, 2): 1,
                              (inst2, 1): 1, (inst2, 2): 1}))

        self.assertEqual((inst2.func(1), call_count),
                         (1, {(inst1, 1): 2, (inst1, 2): 1,
                              (inst2, 1): 2, (inst2, 2): 1}))
        self.assertEqual((inst2.func(1), call_count),
                         (1, {(inst1, 1): 2, (inst1, 2): 1,
                              (inst2, 1): 2, (inst2, 2): 1}))

        # Remove the above references to the instances and ensure that
        # remember_last has not made its own.
        del inst1, inst2
        call_count.clear()
        while gc.collect():
            pass

        self.assertFalse([inst for inst in gc.get_objects()
                          if type(inst) == clz])
