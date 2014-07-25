#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Zipline
"""

# This is *not* a place to dump arbitrary classes/modules for convenience,
# it is a place to expose the public interfaces.

__version__ = "0.7.0"

from . import data
from . import finance
from . import gens
from . import utils
from . import transforms
from . algorithm import TradingAlgorithm

from . import api

try:
    ip = get_ipython()  # flake8: noqa
    ip.register_magic_function(utils.parse_cell_magic, "line_cell", "zipline")
except:
    pass

__all__ = [
    'data',
    'finance',
    'gens',
    'utils',
    'transforms',
    'api',
    'TradingAlgorithm',
]
