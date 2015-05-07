"""
filter.py
"""
from abc import (
    ABCMeta,
    abstractmethod,
)

from six import with_metaclass


class Filter(with_metaclass(ABCMeta)):
    """
    A boolean predicate on a universe of Assets.
    """
    pass
