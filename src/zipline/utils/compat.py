import functools
import inspect
from collections import namedtuple  # noqa: compatibility with python 3.11
from contextlib import contextmanager, ExitStack
from html import escape as escape_html
from math import ceil
from types import MappingProxyType as mappingproxy


def consistent_round(val):
    if (val % 1) >= 0.5:
        return ceil(val)
    else:
        return round(val)


update_wrapper = functools.update_wrapper
wraps = functools.wraps


def getargspec(f):
    ArgSpec = namedtuple(
        "ArgSpec", "args varargs keywords defaults"
    )  # noqa: compatibility with python 3.11
    full_argspec = inspect.getfullargspec(f)
    return ArgSpec(
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
