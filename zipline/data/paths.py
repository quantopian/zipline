"""
Canonical path locations for zipline data.

Paths are rooted at $ZIPLINE_ROOT if that environment variable is set.
Otherwise default to expanduser(~/.zipline)
"""
import os
from os.path import (
    expanduser,
    join,
)


def zipline_root(environ=None):
    """
    Get the root directory for all zipline-managed files.

    For testing purposes, this accepts a dictionary to interpret as the os
    environment.

    Parameters
    ----------
    environ : dict, optional
        A dict to interpret as the os environment.

    Returns
    -------
    root : string
        Path to the zipline root dir.
    """
    if environ is None:
        environ = os.environ.copy()

    root = environ.get('ZIPLINE_ROOT', None)
    if root is None:
        root = expanduser('~/.zipline')

    return root


def zipline_root_path(path, environ=None):
    """
    Get a path relative to the zipline root.

    Parameters
    ----------
    path : str
        The requested path.
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    newpath : str
        The requested path joined with the zipline root.
    """
    return join(zipline_root(environ=environ), path)


def data_root(environ=None):
    """
    The root directory for zipline data files.

    Parameters
    ----------
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    data_root : str
       The zipline data root.
    """
    return zipline_root_path('data', environ=environ)


def cache_root(environ=None):
    """
    The root directory for zipline cache files.

    Parameters
    ----------
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    cache_root : str
       The zipline cache root.
    """
    return zipline_root_path('cache', environ=environ)
