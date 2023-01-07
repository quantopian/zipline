"""Caching utilities for zipline"""
from collections.abc import MutableMapping
import errno
from functools import partial
import os
import pickle
from distutils import dir_util
from shutil import rmtree, move
from tempfile import mkdtemp, NamedTemporaryFile

import pandas as pd

from .context_tricks import nop_context
from .paths import ensure_directory
from .sentinel import sentinel


class Expired(Exception):
    """Marks that a :class:`CachedObject` has expired."""


ExpiredCachedObject = sentinel("ExpiredCachedObject")
AlwaysExpired = sentinel("AlwaysExpired")


class CachedObject:
    """A simple struct for maintaining a cached object with an expiration date.

    Parameters
    ----------
    value : object
        The object to cache.
    expires : datetime-like
        Expiration date of `value`. The cache is considered invalid for dates
        **strictly greater** than `expires`.

    Examples
    --------
    >>> from pandas import Timestamp, Timedelta
    >>> expires = Timestamp('2014', tz='UTC')
    >>> obj = CachedObject(1, expires)
    >>> obj.unwrap(expires - Timedelta('1 minute'))
    1
    >>> obj.unwrap(expires)
    1
    >>> obj.unwrap(expires + Timedelta('1 minute'))
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    Expired: 2014-01-01 00:00:00+00:00
    """

    def __init__(self, value, expires):
        self._value = value
        self._expires = expires

    @classmethod
    def expired(cls):
        """Construct a CachedObject that's expired at any time."""
        return cls(ExpiredCachedObject, expires=AlwaysExpired)

    def unwrap(self, dt):
        """
        Get the cached value.

        Returns
        -------
        value : object
            The cached value.

        Raises
        ------
        Expired
            Raised when `dt` is greater than self.expires.
        """
        expires = self._expires
        if expires is AlwaysExpired or expires < dt:
            raise Expired(self._expires)
        return self._value

    def _unsafe_get_value(self):
        """You almost certainly shouldn't use this."""
        return self._value


class ExpiringCache:
    """A cache of multiple CachedObjects, which returns the wrapped the value
    or raises and deletes the CachedObject if the value has expired.

    Parameters
    ----------
    cache : dict-like, optional
        An instance of a dict-like object which needs to support at least:
        `__del__`, `__getitem__`, `__setitem__`
        If `None`, than a dict is used as a default.

    cleanup : callable, optional
        A method that takes a single argument, a cached object, and is called
        upon expiry of the cached object, prior to deleting the object. If not
        provided, defaults to a no-op.

    Examples
    --------
    >>> from pandas import Timestamp, Timedelta
    >>> expires = Timestamp('2014', tz='UTC')
    >>> value = 1
    >>> cache = ExpiringCache()
    >>> cache.set('foo', value, expires)
    >>> cache.get('foo', expires - Timedelta('1 minute'))
    1
    >>> cache.get('foo', expires + Timedelta('1 minute'))
    Traceback (most recent call last):
        ...
    KeyError: 'foo'
    """

    def __init__(self, cache=None, cleanup=lambda value_to_clean: None):
        if cache is not None:
            self._cache = cache
        else:
            self._cache = {}

        self.cleanup = cleanup

    def get(self, key, dt):
        """Get the value of a cached object.

        Parameters
        ----------
        key : any
            The key to lookup.
        dt : datetime
            The time of the lookup.

        Returns
        -------
        result : any
            The value for ``key``.

        Raises
        ------
        KeyError
            Raised if the key is not in the cache or the value for the key
            has expired.
        """
        try:
            return self._cache[key].unwrap(dt)
        except Expired as exc:
            self.cleanup(self._cache[key]._unsafe_get_value())
            del self._cache[key]
            raise KeyError(key) from exc

    def set(self, key, value, expiration_dt):
        """Adds a new key value pair to the cache.

        Parameters
        ----------
        key : any
            The key to use for the pair.
        value : any
            The value to store under the name ``key``.
        expiration_dt : datetime
            When should this mapping expire? The cache is considered invalid
            for dates **strictly greater** than ``expiration_dt``.
        """
        self._cache[key] = CachedObject(value, expiration_dt)


class dataframe_cache(MutableMapping):
    """A disk-backed cache for dataframes.

    ``dataframe_cache`` is a mutable mapping from string names to pandas
    DataFrame objects.
    This object may be used as a context manager to delete the cache directory
    on exit.

    Parameters
    ----------
    path : str, optional
        The directory path to the cache. Files will be written as
        ``path/<keyname>``.
    lock : Lock, optional
        Thread lock for multithreaded/multiprocessed access to the cache.
        If not provided no locking will be used.
    clean_on_failure : bool, optional
        Should the directory be cleaned up if an exception is raised in the
        context manager.
    serialize : {'msgpack', 'pickle:<n>'}, optional
        How should the data be serialized. If ``'pickle'`` is passed, an
        optional pickle protocol can be passed like: ``'pickle:3'`` which says
        to use pickle protocol 3.

    Notes
    -----
    The syntax ``cache[:]`` will load all key:value pairs into memory as a
    dictionary.
    The cache uses a temporary file format that is subject to change between
    versions of zipline.
    """

    def __init__(
        self, path=None, lock=None, clean_on_failure=True, serialization="pickle"
    ):
        self.path = path if path is not None else mkdtemp()
        self.lock = lock if lock is not None else nop_context
        self.clean_on_failure = clean_on_failure

        if serialization == "msgpack":
            self.serialize = pd.DataFrame.to_msgpack
            self.deserialize = pd.read_msgpack
            self._protocol = None
        else:
            s = serialization.split(":", 1)
            if s[0] != "pickle":
                raise ValueError(
                    "'serialization' must be either 'msgpack' or 'pickle[:n]'",
                )
            self._protocol = int(s[1]) if len(s) == 2 else None

            self.serialize = self._serialize_pickle
            self.deserialize = partial(pickle.load, encoding="latin-1")

        ensure_directory(self.path)

    def _serialize_pickle(self, df, path):
        with open(path, "wb") as f:
            pickle.dump(df, f, protocol=self._protocol)

    def _keypath(self, key):
        return os.path.join(self.path, key)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, tb):
        if not (self.clean_on_failure or value is None):
            # we are not cleaning up after a failure and there was an exception
            return

        with self.lock:
            rmtree(self.path)

    def __getitem__(self, key):
        if key == slice(None):
            return dict(self.items())

        with self.lock:
            try:
                with open(self._keypath(key), "rb") as f:
                    return self.deserialize(f)
            except IOError as exc:
                if exc.errno != errno.ENOENT:
                    raise
                raise KeyError(key) from exc

    def __setitem__(self, key, value):
        with self.lock:
            self.serialize(value, self._keypath(key))

    def __delitem__(self, key):
        with self.lock:
            try:
                os.remove(self._keypath(key))
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    # raise a keyerror if this directory did not exist
                    raise KeyError(key) from exc
                # reraise the actual oserror otherwise
                raise

    def __iter__(self):
        return iter(os.listdir(self.path))

    def __len__(self):
        return len(os.listdir(self.path))

    def __repr__(self):
        return "<%s: keys={%s}>" % (
            type(self).__name__,
            ", ".join(map(repr, sorted(self))),
        )


class working_file:
    """A context manager for managing a temporary file that will be moved
    to a non-temporary location if no exceptions are raised in the context.

    Parameters
    ----------
    final_path : str
        The location to move the file when committing.
    *args, **kwargs
        Forwarded to NamedTemporaryFile.

    Notes
    -----
    The file is moved on __exit__ if there are no exceptions.
    ``working_file`` uses :func:`shutil.move` to move the actual files,
    meaning it has as strong of guarantees as :func:`shutil.move`.
    """

    def __init__(self, final_path, *args, **kwargs):
        self._tmpfile = NamedTemporaryFile(delete=False, *args, **kwargs)
        self._final_path = final_path

    @property
    def path(self):
        """Alias for ``name`` to be consistent with
        :class:`~zipline.utils.cache.working_dir`.
        """
        return self._tmpfile.name

    def _commit(self):
        """Sync the temporary file to the final path."""
        move(self.path, self._final_path)

    def __enter__(self):
        self._tmpfile.__enter__()
        return self

    def __exit__(self, *exc_info):
        self._tmpfile.__exit__(*exc_info)
        if exc_info[0] is None:
            self._commit()


class working_dir:
    """A context manager for managing a temporary directory that will be moved
    to a non-temporary location if no exceptions are raised in the context.

    Parameters
    ----------
    final_path : str
        The location to move the file when committing.
    *args, **kwargs
        Forwarded to tmp_dir.

    Notes
    -----
    The file is moved on __exit__ if there are no exceptions.
    ``working_dir`` uses :func:`dir_util.copy_tree` to move the actual files,
    meaning it has as strong of guarantees as :func:`dir_util.copy_tree`.
    """

    def __init__(self, final_path, *args, **kwargs):
        self.path = mkdtemp()
        self._final_path = final_path

    def ensure_dir(self, *path_parts):
        """Ensures a subdirectory of the working directory.

        Parameters
        ----------
        path_parts : iterable[str]
            The parts of the path after the working directory.
        """
        path = self.getpath(*path_parts)
        ensure_directory(path)
        return path

    def getpath(self, *path_parts):
        """Get a path relative to the working directory.

        Parameters
        ----------
        path_parts : iterable[str]
            The parts of the path after the working directory.
        """
        return os.path.join(self.path, *path_parts)

    def _commit(self):
        """Sync the temporary directory to the final path."""
        dir_util.copy_tree(self.path, self._final_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if exc_info[0] is None:
            self._commit()
        rmtree(self.path)
