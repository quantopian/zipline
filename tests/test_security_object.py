from unittest import TestCase
from zipline.assets._securities import Security


class TestSecurityRichCmp(TestCase):
    def test_lt(self):
        self.assertTrue(Security(3) < Security(4))
        self.assertFalse(Security(4) < Security(4))
        self.assertFalse(Security(5) < Security(4))

    def test_le(self):
        self.assertTrue(Security(3) <= Security(4))
        self.assertTrue(Security(4) <= Security(4))
        self.assertFalse(Security(5) <= Security(4))

    def test_eq(self):
        self.assertFalse(Security(3) == Security(4))
        self.assertTrue(Security(4) == Security(4))
        self.assertFalse(Security(5) == Security(4))

    def test_ge(self):
        self.assertFalse(Security(3) >= Security(4))
        self.assertTrue(Security(4) >= Security(4))
        self.assertTrue(Security(5) >= Security(4))

    def test_gt(self):
        self.assertFalse(Security(3) > Security(4))
        self.assertFalse(Security(4) > Security(4))
        self.assertTrue(Security(5) > Security(4))
