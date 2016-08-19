#
# Copyright 2015 Quantopian, Inc.
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


# This is *not* a place to dump arbitrary classes/modules for convenience,
# it is a place to expose the public interfaces.
from . import data
from . import finance
from . import gens
from . import utils
from .utils.run_algo import run_algorithm
from ._version import get_versions
# These need to happen after the other imports.
from . algorithm import TradingAlgorithm
from . import api

__version__ = get_versions()['version']
del get_versions


def load_ipython_extension(ipython):
    from .__main__ import zipline_magic
    ipython.register_magic_function(zipline_magic, 'line_cell', 'zipline')


__all__ = [
    'TradingAlgorithm',
    'api',
    'data',
    'finance',
    'gens',
    'run_algorithm',
    'utils',
]
