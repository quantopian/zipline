import pandas as pd
from zipline.utils.cache import CachedObject, Expired, ExpiringCache
import pytest


class TestCachedObject:
    def test_cached_object(self):
        expiry = pd.Timestamp("2014")
        before = expiry - pd.Timedelta("1 minute")
        after = expiry + pd.Timedelta("1 minute")

        obj = CachedObject(1, expiry)

        assert obj.unwrap(before) == 1
        assert obj.unwrap(expiry) == 1  # Unwrap on expiry is allowed.
        with pytest.raises(Expired, match=str(expiry)):
            obj.unwrap(after)

    @pytest.mark.parametrize(
        "date",
        [pd.Timestamp.min, pd.Timestamp.now(), pd.Timestamp.max],
        ids=["minTime", "nowTime", "maxTime"],
    )
    def test_expired(self, date):
        always_expired = CachedObject.expired()
        with pytest.raises(Expired):
            always_expired.unwrap(date)


class TestExpiringCache:
    def test_expiring_cache(self):
        expiry_1 = pd.Timestamp("2014")
        before_1 = expiry_1 - pd.Timedelta("1 minute")
        after_1 = expiry_1 + pd.Timedelta("1 minute")

        expiry_2 = pd.Timestamp("2015")
        after_2 = expiry_1 + pd.Timedelta("1 minute")

        expiry_3 = pd.Timestamp("2016")

        cache = ExpiringCache()

        cache.set("foo", 1, expiry_1)
        cache.set("bar", 2, expiry_2)

        assert cache.get("foo", before_1) == 1  # Unwrap on expiry is allowed.
        assert cache.get("foo", expiry_1) == 1

        with pytest.raises(KeyError, match="foo"):
            cache.get("foo", after_1)

        # Should raise same KeyError after deletion.
        with pytest.raises(KeyError, match="foo"):
            cache.get("foo", before_1)

        # Second value should still exist.
        assert cache.get("bar", after_2) == 2

        # Should raise similar KeyError on non-existent key.
        with pytest.raises(KeyError, match="baz"):
            cache.get("baz", expiry_3)
