"""
Control flow utilities.
"""
from contextlib import contextmanager


@contextmanager
def nullctx():
    """
    Null context manager.  Useful for conditionally adding a contextmanager in
    a single line, e.g.:

    with SomeContextManager() if some_expr else nullctx():
        do_stuff()
    """
    yield
