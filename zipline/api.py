#
# Copyright 2014 Quantopian, Inc.
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
from .utils import math_utils, events

from zipline.finance.slippage import (
    FixedSlippage,
    VolumeShareSlippage,
)

from zipline.utils.events import (
    date_rules,
    time_rules
)

batch_transform = zipline.transforms.BatchTransform


__all__ = [
    'slippage',
    'commission',
    'events',
    'math_utils',
    'batch_transform',
    'FixedSlippage',
    'VolumeShareSlippage',
    'date_rules',
    'time_rules'
]
