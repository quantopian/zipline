"""
Tools for memoization of function results.
"""
from zipline.utils.compat import lru_cache
from weakref import WeakKeyDictionary


class lazyval(object):
    """Decorator that marks that an attribute of an instance should not be
    computed until needed, and that the value should be memoized.

    Example
    -------

    >>> from zipline.utils.memoize import lazyval
    >>> class C(object):
    ...     def __init__(self):
    ...         self.count = 0
    ...     @lazyval
    ...     def val(self):
    ...         self.count += 1
    ...         return "val"
    ...
    >>> c = C()
    >>> c.count
    0
    >>> c.val, c.count
    ('val', 1)
    >>> c.val, c.count
    ('val', 1)
    >>> c.val = 'not_val'
    Traceback (most recent call last):
    ...
    AttributeError: Can't set read-only attribute.
    >>> c.val
    'val'
    """
    def __init__(self, get):
        self._get = get
        self._cache = WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return self._cache[instance]
        except KeyError:
            self._cache[instance] = val = self._get(instance)
            return val

    def __set__(self, instance, value):
        raise AttributeError("Can't set read-only attribute.")

    def __delitem__(self, instance):
        del self._cache[instance]


class classlazyval(lazyval):
    """ Decorator that marks that an attribute of a class should not be
    computed until needed, and that the value should be memoized.

    Example
    -------

    >>> from zipline.utils.memoize import classlazyval
    >>> class C(object):
    ...     count = 0
    ...     @classlazyval
    ...     def val(cls):
    ...         cls.count += 1
    ...         return "val"
    ...
    >>> C.count
    0
    >>> C.val, C.count
    ('val', 1)
    >>> C.val, C.count
    ('val', 1)
    """
    # We don't reassign the name on the class to implement the caching because
    # then we would need to use a metaclass to track the name of the
    # descriptor.
    def __get__(self, instance, owner):
        return super(classlazyval, self).__get__(owner, owner)


remember_last = lru_cache(1)
