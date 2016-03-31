from six import PY2


if PY2:
    from functools32 import lru_cache
    from ctypes import py_object, pythonapi

    mappingproxy = pythonapi.PyDictProxy_New
    mappingproxy.argtypes = [py_object]
    mappingproxy.restype = py_object

else:
    from functools import lru_cache
    from types import MappingProxyType as mappingproxy

__all__ = [
    'lru_cache',
    'mappingproxy',
]
