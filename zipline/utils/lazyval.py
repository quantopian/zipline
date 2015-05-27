"""
An immutable, lazily loaded value descriptor.
"""


from weakref import WeakKeyDictionary


class lazyval(object):
    """
    Decorator that marks that an attribute should not be computed until
    needed, and that the value should be memoized.

    This works like:

      >>> class C(object):
      ...     @property
      ...     @lru_cache(1)
      ...     def f(self):
      ...         ...
      ...

    However, `f` will only implement the `__get__` method.
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
