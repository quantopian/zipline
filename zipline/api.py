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

# Note that part of the API is implemented in TradingAlgorithm as
# methods (e.g. order). These are added to this namespace via the
# decorator `api_methods` inside of algorithm.py.

import zipline
from .finance import (commission, slippage)
from .utils import math_utils

from zipline.finance.slippage import (
    FixedSlippage,
    VolumeShareSlippage,
)


batch_transform = zipline.transforms.BatchTransform


def symbol(symbol_str, as_of_date=None):
    """Default symbol lookup for any source that directly maps the
    symbol to the identifier (e.g. yahoo finance).

    Keyword argument as_of_date is ignored.
    """
    return symbol_str

__all__ = [
    'slippage',
    'commission',
    'math_utils',
    'batch_transform',
    'FixedSlippage',
    'VolumeShareSlippage'
]
