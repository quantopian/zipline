from copy import copy, deepcopy
from pickle import loads, dumps
from unittest import TestCase
from weakref import ref

from zipline.utils.sentinel import sentinel


class SentinelTestCase(TestCase):
    def tearDown(self):
        sentinel._cache.clear()  # don't pollute cache.

    def test_name(self):
        self.assertEqual(sentinel('a').__name__, 'a')

    def test_doc(self):
        self.assertEqual(sentinel('a', 'b').__doc__, 'b')

    def test_doc_differentiates(self):
        self.assertIsNot(sentinel('a', 'b'), sentinel('a', 'c'))

    def test_memo(self):
        self.assertIs(sentinel('a'), sentinel('a'))

    def test_copy(self):
        a = sentinel('a')
        self.assertIs(copy(a), a)

    def test_deepcopy(self):
        a = sentinel('a')
        self.assertIs(deepcopy(a), a)

    def test_repr(self):
        self.assertEqual(
            repr(sentinel('a')),
            "sentinel('a')",
        )

    def test_new(self):
        with self.assertRaises(TypeError):
            type(sentinel('a'))()

    def test_pickle_roundtrip(self):
        a = sentinel('a')
        self.assertIs(loads(dumps(a)), a)

    def test_weakreferencable(self):
        ref(sentinel('a'))
