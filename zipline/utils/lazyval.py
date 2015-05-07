"""
An immutable, lazily loaded value descriptor.
"""


from weakref import WeakKeyDictionary


class lazyval(object):
    """
    Decorator that marks that an attribute should not be computed until
    needed, and that the value should be memoized.

    Example
    -------

    >>> from zipline.utils.lazyval import lazyval
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
