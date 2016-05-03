from six import PY2


if PY2:
    from ctypes import py_object, pythonapi

    mappingproxy = pythonapi.PyDictProxy_New
    mappingproxy.argtypes = [py_object]
    mappingproxy.restype = py_object

else:
    from types import MappingProxyType as mappingproxy

__all__ = [
    'mappingproxy',
]
