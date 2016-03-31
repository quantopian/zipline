"""
Caching utilities for zipline
"""
from collections import namedtuple


class Expired(Exception):
    pass


class CachedObject(namedtuple("_CachedObject", "value expires")):
    """
    A simple struct for maintaining a cached object with an expiration date.

    Parameters
    ----------
    value : object
        The object to cache.
    expires : datetime-like
        Expiration date of `value`. The cache is considered invalid for dates
        **strictly greater** than `expires`.

    Methods
    -------
    get(self, dt)
        Get the cached object.

    Usage
    -----
    >>> from pandas import Timestamp, Timedelta
    >>> expires = Timestamp('2014', tz='UTC')
    >>> obj = CachedObject(1, expires)
    >>> obj.unwrap(expires - Timedelta('1 minute'))
    1
    >>> obj.unwrap(expires)
    1
    >>> obj.unwrap(expires + Timedelta('1 minute'))
    Traceback (most recent call last):
        ...
    Expired: 2014-01-01 00:00:00+00:00
    """

    def unwrap(self, dt):
        """
        Get the cached value.

        Returns
        -------
        value : object
            The cached value.

        Raises
        ------
        Expired
            Raised when `dt` is greater than self.expires.
        """
        if dt > self.expires:
            raise Expired(self.expires)
        return self.value


class ExpiringCache(object):
    """
    A cache of multiple CachedObjects, which returns the wrapped the value
    or raises and deletes the CachedObject if the value has expired.

    Parameters
    ----------
    cache : dict-like
        An instance of a dict-like object which needs to support at least:
        `__del__`, `__getitem__`, `__setitem__`
        If `None`, than a dict is used as a default.

    Methods
    -------
    get(self, key, dt)
        Get the value of a cached object for the given `key` at `dt`, if the
        CachedObject has expired then the object is removed from the cache,
        and `KeyError` is raised.

    set(self, key, value, expiration_dt)
        Add a new `value` to the cache at `dt` wrapped in a CachedObject which
        expires at `expiration_dt`.

    Usage
    -----
    >>> from pandas import Timestamp, Timedelta
    >>> expires = Timestamp('2014', tz='UTC')
    >>> value = 1
    >>> cache = ExpiringCache()
    >>> cache.set('foo', value, expires)
    >>> cache.get('foo', expires - Timedelta('1 minute'))
    1
    >>> cache.get('foo', expires + Timedelta('1 minute'))
    Traceback (most recent call last):
        ...
    KeyError: 'foo'
    """

    def __init__(self, cache=None):
        if cache is not None:
            self._cache = cache
        else:
            self._cache = {}

    def get(self, key, dt):
        try:
            return self._cache[key].unwrap(dt)
        except Expired:
            del self._cache[key]
            raise KeyError(key)

    def set(self, key, value, expiration_dt):
        self._cache[key] = CachedObject(value, expiration_dt)
