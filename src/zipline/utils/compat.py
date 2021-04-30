import functools
import inspect

from contextlib import contextmanager, ExitStack
from html import escape as escape_html
from types import MappingProxyType as mappingproxy
from math import ceil


def consistent_round(val):
    if (val % 1) >= 0.5:
        return ceil(val)
    else:
        return round(val)


update_wrapper = functools.update_wrapper
wraps = functools.wraps


def getargspec(f):
    full_argspec = inspect.getfullargspec(f)
    return inspect.ArgSpec(
        args=full_argspec.args,
        varargs=full_argspec.varargs,
        keywords=full_argspec.varkw,
        defaults=full_argspec.defaults,
    )


unicode = type("")

__all__ = [
    "ExitStack",
    "consistent_round",
    "contextmanager",
    "escape_html",
    "mappingproxy",
    "unicode",
    "update_wrapper",
    "wraps",
]
