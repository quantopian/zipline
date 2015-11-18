"""
Control flow utilities.
"""
from six import iteritems
from warnings import (
    catch_warnings,
    filterwarnings,
)


class nullctx(object):
    """
    Null context manager.  Useful for conditionally adding a contextmanager in
    a single line, e.g.:

    with SomeContextManager() if some_expr else nullctx():
        do_stuff()
    """
    def __enter__(self):
        return self

    def __exit__(*args):
        return False


class WarningContext(object):
    """
    Re-entrant contextmanager for contextually managing warnings.
    """
    def __init__(self, *warning_specs):
        self._warning_specs = warning_specs
        self._catchers = []

    def __enter__(self):
        catcher = catch_warnings()
        catcher.__enter__()
        self._catchers.append(catcher)
        for args, kwargs in self._warning_specs:
            filterwarnings(*args, **kwargs)
        return catcher

    def __exit__(self, *exc_info):
        catcher = self._catchers.pop()
        return catcher.__exit__(*exc_info)


def ignore_nanwarnings():
    """
    Helper for building a WarningContext that ignores warnings from numpy's
    nanfunctions.
    """
    return WarningContext(
        (
            ('ignore',),
            {'category': RuntimeWarning, 'module': 'numpy.lib.nanfunctions'},
        )
    )


def invert(d):
    """
    Invert a dictionary into a dictionary of sets.

    >>> invert({'a': 1, 'b': 2, 'c': 1})
    {1: {'a', 'c'}, 2: {'b'}}
    """
    out = {}
    for k, v in iteritems(d):
        try:
            out[v].add(k)
        except KeyError:
            out[v] = {k}
    return out
