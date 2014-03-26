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

# Delete unused imports once algo is finished

from datetime import datetime
import pytz

from zipline import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo

from zipline.api import order
from zipline.api import order_percent
from zipline.api import order_target
from zipline.api import order_target_value
from zipline.api import order_target_percent
from zipline.api import order_value
from zipline.api import cancel_order
from zipline.api import get_open_orders
from zipline.api import get_order

from zipline.algorithm import set_slippage

from zipline.transforms import BatchTransform, batch_transform

# Loggin. Following imports are not approved in Quantopian
####################################################################################

import logbook
import sys

log_format = "{record.extra[algo_dt]}  {record.message}"

zipline_logging = logbook.NestedSetup([
    logbook.StreamHandler(sys.stdout, level=logbook.INFO, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.WARNING, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.NOTICE, format_string=log_format),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR, format_string=log_format),
])
zipline_logging.push_application()

log = logbook.Logger('Main Logger')

# Cut and past between the ### line below to Quantopian. Change symbols to sids. E.g. 'AAPL' to sid(24)
####################################################################################

# Place Quantopian approved imports here

def initialize(context):
    context.test = 10


def handle_date(context, data):
    order('AAPL', 10)
    print(context.test)


####################################################################################
# Cut and past between the ### line above to Quantopian. Change symbols to sids. E.g. 'AAPL' to sid(24)

if __name__ == '__main__':
    import pylab as pl
    
    # Any other non approved imports to make things work or to plot results    
    
    start = datetime(2008, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
    
    # Reference any stocks data you need below
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
                           end=end)
    data = data.dropna()
    algo = TradingAlgorithm(initialize=initialize,
                            handle_data=handle_date)
    results = algo.run(data)
    results.portfolio_value.plot()
    pl.show()
