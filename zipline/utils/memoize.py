"""
Tools for memoization of function results.
"""
from functools import wraps
from six import iteritems
from weakref import WeakKeyDictionary


class lazyval(object):
    """
    Decorator that marks that an attribute should not be computed until
    needed, and that the value should be memoized.

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


def remember_last(f):
    """
    Decorator that remembers the last computed value of a function and doesn't
    recompute it when called with the same inputs multiple times.

    Parameters
    ----------
    f : The function to be memoized. All arguments to f should be hashable.

    Example
    -------
    >>> counter = 0
    >>> @remember_last
    ... def foo(x):
    ...     global counter
    ...     counter += 1
    ...     return x, counter
    >>> foo(1)
    (1, 1)
    >>> foo(1)
    (1, 1)
    >>> foo(0)
    (0, 2)
    >>> foo(1)
    (1, 3)

    Notes
    -----
    This decorator is equivalent to `lru_cache(1)` in Python 3, but with less
    bells and whistles for handling things like threadsafety.  If we ever
    decide we need such bells and whistles, we should just make functools32 a
    dependency.
    """
    # This needs to be a mutable data structure so we can change it from inside
    # the function.  In pure Python 3, we'd use the nonlocal keyword for this.
    _previous = [None, None]
    KEY, VALUE = 0, 1

    _kwd_mark = object()

    @wraps(f)
    def memoized_f(*args, **kwds):
        # Hashing logic taken from functools32.lru_cache.
        key = args
        if kwds:
            key += _kwd_mark + tuple(sorted(iteritems(kwds)))

        key_hash = hash(key)
        if key_hash != _previous[KEY]:
            _previous[VALUE] = f(*args, **kwds)
            _previous[KEY] = key_hash
        return _previous[VALUE]

    return memoized_f
