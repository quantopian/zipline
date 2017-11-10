"""
Control flow utilities.
"""
# re-export for backwards compat
from .functional import invert  # noqa


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
