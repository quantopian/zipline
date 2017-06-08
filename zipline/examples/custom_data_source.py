#!/usr/bin/env python
#
# Copyright 2013 Paul Cao
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

from zipline.sources.data_source import DataSource
from zipline.algorithm import TradingAlgorithm
from zipline.finance import trading

import matplotlib.pyplot as plt
import zipline.utils.factory as factory
import pandas as pd

import pytz
import random

class BuyApple(TradingAlgorithm):  # inherit from TradingAlgorithm
    """This is the simplest possible algorithm that does nothing but
    buy 1 share of AAPL on each event if we haven't ordered already.
    """
    def handle_data(self, data):  # overload handle_data() method
        if (hasattr(self, "ordered") == False):
            self.order("AAPL", 1)  # order AAPL and amount (=1 shares)
            self.ordered = True
    
class CustomDataSource(DataSource):
    """This is a custom data source that generates random quotes with random
    quote as an demonstration to how to create a custom data source
    """
    def __init__(self, symbols, bars, start, end):
        """ Constructor for the data source
        
        Parameters
        ----------
        symbols : array
            Symbols to simulate the custom data source on
        bars : string
            'minute' or 'daily'
        start : pd.Timestamp
            start date of data source
        end: pd.Timestamp
            end date of data sourceS
        """
        self._raw_data = None
        self.symbols = symbols
        self.start = start
        self.end = end
        self.bars = bars
        
    @property
    def mapping(self):
        return {
            'dt': (lambda x: x, 'dt'),
            'sid': (lambda x: x, 'sid'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
        }
    
    @property
    def instance_hash(self):
        return "CustomDataSource"
    
    def raw_data_gen(self):
        """ The generator function that is used by zipline to iterate through the custom
        data source, modify code here to connect to a database or parse through a file
        """
        
        # figure out the frequency of the data source
        if self.bars == 'daily':
            freq = pd.datetools.BDay()
        elif self.bars == 'minute':
            freq = pd.datetools.Minute()
        else:
            raise ValueError('%s bars not understood.' % freq)
        # figure out trading days in the given date range
        days = trading.environment.days_in_range(start, end)

        if self.bars == 'daily':
            index = days
        if self.bars == 'minute':
            index = pd.DatetimeIndex([], freq=freq)
            
            for day in days:
                day_index = trading.environment.market_minutes_for_day(day) #generate the trading minutes in the given day
                index = index.append(day_index)
        
        # iterate through the available trading interval in this data source's date range
        for i in range(len(index)):
            for symbol in self.symbols:
                
                # make up random price and random volume, change here to hook up to your data source
                randPrice = random.randrange(1,100)
                randVolume = random.randrange(1,100)
                
                # yield the data event to zipline backtester thread
                yield {'dt': index[i], # timestamp (e.g., 2013-12-10 00:00:00+00:00)
                       'sid': symbol, # symbol (e.g., AAPL)
                       'price': randPrice, 
                       'volume': randVolume
                }
    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
    
        return self._raw_data
    
if __name__ == "__main__":
    # set up the the simulation parameters for the backtesting
    start = pd.datetime(2013, 12, 10, 0, 0, 0, 0, pytz.utc)
    end = pd.datetime(2013, 12, 18, 0, 0, 0, 0, pytz.utc)
    sim_params = factory.create_simulation_parameters(
                start=start,
                end=end)
    #set this backtest to have minute bars
    sim_params.emission_rate = 'minute' 
    sim_params.data_frequency = 'minute'
    sim_params.first_open = sim_params.period_start
    sim_params.last_close = sim_params.period_end 
    
    # set up the custom data source and the trading algorithm
    source = CustomDataSource(['MSFT','AAPL'], 'minute', start, end)
    algo = BuyApple(sim_params=sim_params)
    
    # run the algo
    results = algo.run(source)
    
    # plot the results
    results.portfolio_value.plot()
    plt.show()
