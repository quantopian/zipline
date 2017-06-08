

# Delete unused imports once algo is finished

from datetime import datetime

from zipline import *
from zipline.algorithm import *
from zipline.api import *
from zipline.api import *
from zipline.data import *
from zipline.errors import *
from zipline.finance import *
from zipline.gens import *
from zipline.protocol import *
from zipline.sources import *
from zipline.transforms import *
from zipline.utils import *
from zipline.version import *


# Logging. Following imports are not approved in Quantopian
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
    pass


def handle_date(context, data):
    order(symbol('AAPL'), 10)


####################################################################################
# Cut and past between the ### line above to Quantopian. Change symbols to sids. E.g. 'AAPL' to sid(24)

if __name__ == '__main__':
    import pylab as pl
    
    # Any other non approved imports to make things work or to plot results    
    
    start = datetime(2008, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    
    # Reference any stocks data you need below
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,
                           end=end)
    data = data.dropna()
    
    
    algo = TradingAlgorithm(initialize=initialize,
                            handle_data=handle_date)
                
                
    results = algo.run(data)
    results.portfolio_value.plot()
    
    
    pl.show()
