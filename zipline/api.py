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

from .finance import (commission, slippage, cancel_policy)
from .utils import math_utils, events

from zipline.finance.slippage import (
    FixedSlippage,
    VolumeShareSlippage,
)

from zipline.finance.cancel_policy import (
    NeverCancel,
    EODCancel
)

from zipline.utils.events import (
    date_rules,
    time_rules
)

__all__ = [
    'slippage',
    'commission',
    'cancel_policy',
    'NeverCancel',
    'EODCancel',
    'events',
    'math_utils',
    'FixedSlippage',
    'VolumeShareSlippage',
    'date_rules',
    'time_rules'
]
