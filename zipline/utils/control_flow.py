"""
Control flow utilities.
"""
from six import iteritems


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
