#!/usr/bin/env python
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

"""Dual Moving Average Crossover algorithm.

This algorithm buys apple once its short moving average crosses
its long moving average (indicating upwards momentum) and sells
its shares once the averages cross again (indicating downwards
momentum).
"""

from zipline.api import order_target, record, symbol
from collections import deque as moving_window
import numpy as np

def initialize(context):
    # Add 2 windows, one with a long window, one
    # with a short window.
    # Note that this is bound to change soon and will be easier.
    context.short_window = moving_window(maxlen=100)
    context.long_window = moving_window(maxlen=300)

def handle_data(context, data):
    # Save price to window
    context.short_window.append(data[symbol('AAPL')].price)
    context.long_window.append(data[symbol('AAPL')].price)

    # Compute averages
    short_mavg = np.mean(context.short_window)
    long_mavg = np.mean(context.long_window)

    # Trading logic
    if short_mavg > long_mavg:
        order_target(symbol('AAPL'), 100)
    elif short_mavg < long_mavg:
        order_target(symbol('AAPL'), 0)

    # Save values for later inspection
    record(AAPL=data[symbol('AAPL')].price,
           short_mavg=short_mavg,
           long_mavg=long_mavg)
