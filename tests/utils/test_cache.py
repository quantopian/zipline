from unittest import TestCase

from pandas import Timestamp, Timedelta

from zipline.utils.cache import CachedObject, Expired, ExpiringCache


class CachedObjectTestCase(TestCase):

    def test_cached_object(self):
        expiry = Timestamp('2014')
        before = expiry - Timedelta('1 minute')
        after = expiry + Timedelta('1 minute')

        obj = CachedObject(1, expiry)

        self.assertEqual(obj.unwrap(before), 1)
        self.assertEqual(obj.unwrap(expiry), 1)  # Unwrap on expiry is allowed.
        with self.assertRaises(Expired) as e:
            obj.unwrap(after)
        self.assertEqual(e.exception.args, (expiry,))


class ExpiringCacheTestCase(TestCase):

    def test_expiring_cache(self):
        expiry_1 = Timestamp('2014')
        before_1 = expiry_1 - Timedelta('1 minute')
        after_1 = expiry_1 + Timedelta('1 minute')

        expiry_2 = Timestamp('2015')
        after_2 = expiry_1 + Timedelta('1 minute')

        expiry_3 = Timestamp('2016')

        cache = ExpiringCache()

        cache.set('foo', 1, expiry_1)
        cache.set('bar', 2, expiry_2)

        self.assertEqual(cache.get('foo', before_1), 1)
        # Unwrap on expiry is allowed.
        self.assertEqual(cache.get('foo', expiry_1), 1)

        with self.assertRaises(KeyError) as e:
            self.assertEqual(cache.get('foo', after_1))
        self.assertEqual(e.exception.args, ('foo',))

        # Should raise same KeyError after deletion.
        with self.assertRaises(KeyError) as e:
            self.assertEqual(cache.get('foo', before_1))
        self.assertEqual(e.exception.args, ('foo',))

        # Second value should still exist.
        self.assertEqual(cache.get('bar', after_2), 2)

        # Should raise similar KeyError on non-existent key.
        with self.assertRaises(KeyError) as e:
            self.assertEqual(cache.get('baz', expiry_3))
        self.assertEqual(e.exception.args, ('baz',))
