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
from zipline.data.data_portal import DataPortal

from logbook import Logger

log = Logger('DataPortalLive')


class DataPortalLive(DataPortal):
    def __init__(self, broker, *args, **kwargs):
        self.broker = broker
        super(DataPortalLive, self).__init__(*args, **kwargs)

    def get_last_traded_dt(self, asset, dt, data_frequency):
        return self.broker.get_last_traded_dt(asset)

    def get_spot_value(self, assets, field, dt, data_frequency):
        return self.broker.get_spot_value(assets, field, dt, data_frequency)

    def get_history_window(self,
                           assets,
                           end_dt,
                           bar_count,
                           frequency,
                           field,
                           data_frequency,
                           ffill=True):
        # This method is responsible for merging the ingested historical data
        # with the real-time collected data through the Broker.
        # DataPortal.get_history_window() is called with ffill=False to mark
        # the missing fields with NaNs. After merge on the historical and
        # real-time data the missing values (NaNs) are filled based on their
        # next available values in the requested time window.
        #
        # Warning: setting ffill=True in DataPortal.get_history_window() call
        # results a wrong behavior: The last available value reported by
        # get_spot_value() will be used to fill the missing data - which is
        # always representing the current spot price presented by Broker.

        if frequency == '1d':
            historical_bars = super(DataPortalLive,
                                    self).get_history_window(
                assets,
                end_dt,
                bar_count,
                frequency,
                field,
                data_frequency,
                ffill=True)
            return historical_bars
        realtime_bars = self.broker.get_realtime_bars(assets, frequency)

        # Broker.get_realtime_history() returns the asset as level 0 column,
        # open, high, low, close, volume returned as level 1 columns.
        # To filter for field the levels needs to be swapped
        realtime_bars = realtime_bars.swaplevel(0, 1, axis=1)

        ohlcv_field = 'close' if field == 'price' else field
        realtime_bars = realtime_bars[ohlcv_field]
        if ffill and field == 'price':
            # Simple forward fill is not enough here as the last ingested
            # value might be outside of the requested time window. That case
            # the time series starts with NaN and forward filling won't help.
            # To provide values for such cases we backward fill.
            # Backward fill as a second operation will have no effect if the
            # forward-fill was successful.
            realtime_bars.fillna(method='ffill', inplace=True)
            realtime_bars.fillna(method='bfill', inplace=True)

        realtime_bars.columns = assets
        return realtime_bars[-bar_count:]

    def get_scalar_asset_spot_value(self, asset, field, dt, data_frequency):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ----------
        assets : Asset
            The asset or assets whose data is desired. This cannot be
            an arbitrary AssetConvertible.
        field : {'open', 'high', 'low', 'close', 'volume',
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        value : float, int, or pd.Timestamp
            The spot value of ``field`` for ``asset`` The return type is based
            on the ``field`` requested. If the field is one of 'open', 'high',
            'low', 'close', or 'price', the value will be a float. If the
            ``field`` is 'volume' the value will be a int. If the ``field`` is
            'last_traded' the value will be a Timestamp.
        """
        if data_frequency == 'minute':
            data_frequency = '1m'
        elif data_frequency == 'daily':
            data_frequency = '1d'
        prices = self.broker.get_realtime_bars([asset], data_frequency)
        if field == 'last_traded':
            return pd.Timestamp(prices[asset][-1:].index.get_values()[0])
        elif field == 'volume':
            return prices[asset][field][-1] * 100
        elif field == 'price':
            return prices[asset]['close'][-1]
        else:
            return prices[asset][field][-1]
