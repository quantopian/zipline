from contextlib import contextmanager


def _nop(*args, **kwargs):
    pass


class CallbackManager(object):
    """Create a context manager from a pre-execution callback and a
    post-execution callback.

    Parameters
    ----------
    pre : (...) -> any, optional
        A pre-execution callback. This will be passed ``*args`` and
        ``**kwargs``.
    post : (...) -> any, optional
        A post-execution callback. This will be passed ``*args`` and
        ``**kwargs``.

    Notes
    -----
    The enter value of this context manager will be the result of calling
    ``pre(*args, **kwargs)``

    Examples
    --------
    >>> def pre(where):
    ...     print('entering %s block' % where)
    >>> def post(where):
    ...     print('exiting %s block' % where)
    >>> manager = CallbackManager(pre, post)
    >>> with manager('example'):
    ...    print('inside example block')
    entering example block
    inside example
    exiting example block

    These are reusable with different args:
    >>> with manager('another'):
    ...     print('inside another block')
    entering another block
    inside another block
    exiting another block
    """
    def __init__(self, pre=None, post=None):
        pre = pre if pre is not None else _nop
        post = post if post is not None else _nop

        @contextmanager
        def _callback_manager_context(*args, **kwargs):
            try:
                yield pre(*args, **kwargs)
            finally:
                post(*args, **kwargs)

        self._callback_manager_context = _callback_manager_context

    def __call__(self, *args, **kwargs):
        return self._callback_manager_context(*args, **kwargs)
