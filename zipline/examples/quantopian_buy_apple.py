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

from datetime import datetime
import pytz

from zipline import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo

from zipline.api import order, symbol


def initialize(context):
    context.test = 10
    context.aapl = symbol('AAPL')


def handle_date(context, data):
    order(context.aapl, 10)
    print(context.test)


if __name__ == '__main__':
    import pylab as pl
    start = datetime(2008, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
                           end=end)
    data = data.dropna()
    algo = TradingAlgorithm(initialize=initialize,
                            handle_data=handle_date,
                            identifiers=['AAPL'])
    results = algo.run(data)
    results.portfolio_value.plot()
    pl.show()
