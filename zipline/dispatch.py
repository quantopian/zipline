"""dispatcher object with a custom namespace.

Anything that has been dispatched will also be put into this module.
"""
from functools import partial
import sys

from multipledispatch import dispatch

try:
    from datashape.dispatch import namespace
except ImportError:
    pass
else:
    globals().update(namespace)

dispatch = partial(dispatch, namespace=globals())

del namespace
del partial
del sys
