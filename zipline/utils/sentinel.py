"""
Construction of sentinel objects.

Sentinel objects are used when you only care to check for object identity.
"""
import sys


def sentinel(name, doc=None):
    @object.__new__   # bind a single instance to the name 'NotSpecified'
    class result(object):
        __doc__ = doc
        __slots__ = ('__weakref__',)

        def __new__(cls):
            raise TypeError("Can't construct new instances of %s" % name)

        def __repr__(self):
            return name

        def __reduce__(self):
            return name

        def __deepcopy__(self, _memo):
            return self

        def __copy__(self):
            return self

    cls = type(result)
    cls.__name__ = name
    try:
        # traverse up one frame to find the module where this is defined
        cls.__module__ = sys._getframe(1).f_globals.get(
            '__name__',
            '__main__',
        )
    except (AttributeError, ValueError):
        pass
    return result
