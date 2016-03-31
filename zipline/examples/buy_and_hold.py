#!/usr/bin/env python
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
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, symbol
from zipline.data.loader import load_bars_from_yahoo

stocks = ['AAPL', 'MSFT']


def initialize(context):
    context.has_ordered = False
    context.stocks = stocks


def handle_data(context, data):
    if not context.has_ordered:
        for stock in context.stocks:
            order(symbol(stock), 100)
        context.has_ordered = True


if __name__ == '__main__':

    # creating time interval
    start = pd.Timestamp('2008-01-01', tz='UTC')
    end = pd.Timestamp('2013-01-01', tz='UTC')

    # loading the data
    input_data = load_bars_from_yahoo(
        stocks=stocks,
        start=start,
        end=end,
    )

    algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
    results = algo.run(input_data)
