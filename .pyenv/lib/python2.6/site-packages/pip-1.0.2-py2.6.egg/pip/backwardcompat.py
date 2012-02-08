"""Stuff that differs in different Python versions"""

import sys
import os
import shutil

__all__ = ['any', 'WindowsError', 'md5', 'copytree']

try:
    WindowsError = WindowsError
except NameError:
    class NeverUsedException(Exception):
        """this exception should never be raised"""
    WindowsError = NeverUsedException
try:
    from hashlib import md5
except ImportError:
    import md5 as md5_module
    md5 = md5_module.new

try:
    from pkgutil import walk_packages
except ImportError:
    # let's fall back as long as we can
    from pip._pkgutil import walk_packages

try:
    any = any
except NameError:

    def any(seq):
        for item in seq:
            if item:
                return True
        return False

console_encoding = sys.__stdout__.encoding

if sys.version_info >= (3,):
    from io import StringIO
    from functools import reduce
    from urllib.error import URLError, HTTPError
    from queue import Queue, Empty
    from urllib.request import url2pathname
    from urllib.request import urlretrieve
    from email import message as emailmessage
    import urllib.parse as urllib
    import urllib.request as urllib2
    import configparser as ConfigParser
    import xmlrpc.client as xmlrpclib
    import urllib.parse as urlparse
    import http.client as httplib
    def cmp(a, b):
        return (a > b) - (a < b)
    def b(s):
        return s.encode('utf-8')
    def u(s):
        return s.decode('utf-8')
    def console_to_str(s):
        return s.decode(console_encoding)
    bytes = bytes
    string_types = (str,)
    raw_input = input
else:
    from cStringIO import StringIO
    from urllib2 import URLError, HTTPError
    from Queue import Queue, Empty
    from urllib import url2pathname, urlretrieve
    from email import Message as emailmessage
    import urllib
    import urllib2
    import urlparse
    import ConfigParser
    import xmlrpclib
    import httplib
    def b(s):
        return s
    def u(s):
        return s
    def console_to_str(s):
        return s
    bytes = str
    string_types = (basestring,)
    reduce = reduce
    cmp = cmp
    raw_input = raw_input

try:
    from email.parser import FeedParser
except ImportError:
    # python lesser than 2.5
    from email.FeedParser import FeedParser

from distutils.sysconfig import get_python_lib, get_python_version

def copytree(src, dst):
    if sys.version_info < (2, 5):
        before_last_dir = os.path.dirname(dst)
        if not os.path.exists(before_last_dir):
            os.makedirs(before_last_dir)
        shutil.copytree(src, dst)
        shutil.copymode(src, dst)
    else:
        shutil.copytree(src, dst)


def product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
