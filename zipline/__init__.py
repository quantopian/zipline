"""
Zipline
"""

# This is *not* a place to dump arbitrary classes/modules for convenience,
# it is a place to expose the public interfaces.

__version__ = "0.5.11.dev"

from . import data
from . import finance
from . import gens
from . import utils

from . algorithm import TradingAlgorithm

__all__ = [
    'data',
    'finance',
    'gens',
    'utils',
    'TradingAlgorithm'
]
