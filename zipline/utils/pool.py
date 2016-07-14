from six.moves import map as imap
from toolz import compose, identity


class ApplyAsyncResult(object):
    """An object that boxes results for calls to
    :meth:`~zipline.utils.pool.SequentialPool.apply_async`.

    Parameters
    ----------
    value : any
        The result of calling the function, or any exception that was raised.
    successful : bool
        If ``True``, ``value`` is the return value of the function.
        If ``False``, ``value`` is the exception that was raised when calling
        the functions.
    """
    def __init__(self, value, successful):
        self._value = value
        self._successful = successful

    def successful(self):
        """Did the function execute without raising an exception?
        """
        return self._successful

    def get(self):
        """Return the result of calling the function or reraise any exceptions
        that were raised.
        """
        if not self._successful:
            raise self._value
        return self._value

    def ready(self):
        """Has the function finished executing.

        Notes
        -----
        In the :class:`~zipline.utils.pool.SequentialPool` case, this is always
        ``True``.
        """
        return True

    def wait(self):
        """Wait until the function is finished executing.

        Notes
        -----
        In the :class:`~zipline.utils.pool.SequentialPool` case, this is a nop
        because the function is computed eagerly in the same thread as the
        call to :meth:`~zipline.utils.pool.SequentialPool.apply_async`.
        """
        pass


class SequentialPool(object):
    """A dummy pool object that iterates sequentially in a single thread.

    Methods
    -------
    map(f: callable[A, B], iterable: iterable[A]) -> list[B]
        Apply a function to each of the elements of ``iterable``.
    imap(f: callable[A, B], iterable: iterable[A]) -> iterable[B]
        Lazily apply a function to each of the elements of ``iterable``.
    imap_unordered(f: callable[A, B], iterable: iterable[A]) -> iterable[B]
        Lazily apply a function to each of the elements of ``iterable`` but
        yield values as they become available. The resulting iterable is
        unordered.

    Notes
    -----
    This object is useful for testing to mock out the ``Pool`` interface
    provided by gevent or multiprocessing.

    See Also
    --------
    :class:`multiprocessing.Pool`
    """
    map = staticmethod(compose(list, imap))
    imap = imap_unordered = staticmethod(imap)

    @staticmethod
    def apply_async(f, args=(), kwargs=None, callback=None):
        """Apply a function but emulate the API of an asynchronous call.

        Parameters
        ----------
        f : callable
            The function to call.
        args : tuple, optional
            The positional arguments.
        kwargs : dict, optional
            The keyword arguments.

        Returns
        -------
        future : ApplyAsyncResult
            The result of calling the function boxed in a future-like api.

        Notes
        -----
        This calls the function eagerly but wraps it so that ``SequentialPool``
        can be used where a :class:`multiprocessing.Pool` or
        :class:`gevent.pool.Pool` would be used.
        """
        try:
            value = (identity if callback is None else callback)(
                f(*args, **kwargs or {}),
            )
            successful = True
        except Exception as e:
            value = e
            successful = False

        return ApplyAsyncResult(value, successful)

    @staticmethod
    def apply(f, args=(), kwargs=None):
        """Apply a function.

        Parameters
        ----------
        f : callable
            The function to call.
        args : tuple, optional
            The positional arguments.
        kwargs : dict, optional
            The keyword arguments.

        Returns
        -------
        result : any
            f(*args, **kwargs)
        """
        return f(*args, **kwargs or {})
