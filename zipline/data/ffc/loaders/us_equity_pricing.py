import shelve

import bcolz

from zipline.data.baseloader import DataLoader
from zipline.data.ffc.loaders._us_equity_pricing import \
    _load_adjusted_array_from_bcolz


class USEquityPricingLoader(DataLoader):
    """
    RFC: Should we abstract out bcolz versus non-bcolz loaders at this time?
    """

    def __init__(self, daily_bar_path, daily_index_path, trading_days):
        self.daily_bar_table = bcolz.open(daily_bar_path)
        self.daily_bar_index = shelve.open(daily_index_path)
        self.trading_days = trading_days

    def load_adjusted_array(self, columns, assets, dates):
        return _load_adjusted_array_from_bcolz(self.daily_bar_table,
                                               self.daily_bar_index,
                                               self.trading_days,
                                               columns,
                                               dates,
                                               assets)
