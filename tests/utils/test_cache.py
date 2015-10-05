from unittest import TestCase

from pandas import Timestamp, Timedelta

from zipline.utils.cache import CachedObject, Expired


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
