import functools
from operator import methodcaller
import sys

from six import PY2


if PY2:
    from abc import ABCMeta
    from types import DictProxyType
    from ctypes import py_object, pythonapi

    _new_mappingproxy = pythonapi.PyDictProxy_New
    _new_mappingproxy.argtypes = [py_object]
    _new_mappingproxy.restype = py_object

    # Make mappingproxy a "class" so that we can use multipledispatch
    # with it or do an ``isinstance(ob, mappingproxy)`` check in Python 2.
    # You will never actually get an instance of this object, you will just
    # get instances of ``types.DictProxyType``; however, ``mappingproxy`` is
    # registered as a virtual super class so ``isinstance`` and ``issubclass``
    # will work as expected. The only thing that will appear strange is that:
    # ``type(mappingproxy({})) is not mappingproxy``, but you shouldn't do
    # that.
    class mappingproxy(object):
        __metaclass__ = ABCMeta

        def __new__(cls, *args, **kwargs):
            return _new_mappingproxy(*args, **kwargs)

    mappingproxy.register(DictProxyType)

    # clear names not imported in the other branch
    del DictProxyType
    del ABCMeta
    del py_object
    del pythonapi

    def exc_clear():
        sys.exc_clear()

    def update_wrapper(wrapper,
                       wrapped,
                       assigned=functools.WRAPPER_ASSIGNMENTS,
                       updated=functools.WRAPPER_UPDATES):
        """Backport of Python 3's functools.update_wrapper for __wrapped__.
        """
        for attr in assigned:
            try:
                value = getattr(wrapped, attr)
            except AttributeError:
                pass
            else:
                setattr(wrapper, attr, value)
        for attr in updated:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
        # from the wrapped function when updating __dict__
        wrapper.__wrapped__ = wrapped
        # Return the wrapper so this can be used as a decorator via partial()
        return wrapper

    def wraps(wrapped,
              assigned=functools.WRAPPER_ASSIGNMENTS,
              updated=functools.WRAPPER_UPDATES):
        """Decorator factory to apply update_wrapper() to a wrapper function

           Returns a decorator that invokes update_wrapper() with the decorated
           function as the wrapper argument and the arguments to wraps() as the
           remaining arguments. Default arguments are as for update_wrapper().
           This is a convenience function to simplify applying partial() to
           update_wrapper().
        """
        return functools.partial(update_wrapper, wrapped=wrapped,
                                 assigned=assigned, updated=updated)

    values_as_list = methodcaller('values')

else:
    from types import MappingProxyType as mappingproxy

    def exc_clear():
        # exc_clear was removed in Python 3. The except statement automatically
        # clears the exception.
        pass

    update_wrapper = functools.update_wrapper
    wraps = functools.wraps

    def values_as_list(dictionary):
        """Return the dictionary values as a list without forcing a copy
        in Python 2.
        """
        return list(dictionary.values())


unicode = type(u'')

__all__ = [
    'PY2',
    'exc_clear',
    'mappingproxy',
    'unicode',
    'update_wrapper',
    'values_as_list',
    'wraps',
]
