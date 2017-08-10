from six import PY2
import functools
import sys


if PY2:
    from ctypes import py_object, pythonapi

    mappingproxy = pythonapi.PyDictProxy_New
    mappingproxy.argtypes = [py_object]
    mappingproxy.restype = py_object

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

else:
    from types import MappingProxyType as mappingproxy

    def exc_clear():
        # exc_clear was removed in Python 3. The except statement automatically
        # clears the exception.
        pass

    update_wrapper = functools.update_wrapper
    wraps = functools.wraps


unicode = type(u'')

__all__ = [
    'mappingproxy',
    'unicode',
]
