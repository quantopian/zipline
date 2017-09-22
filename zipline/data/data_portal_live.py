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

        historical_bars = super(DataPortalLive, self).get_history_window(
            assets, end_dt, bar_count, frequency, field, data_frequency,
            ffill=False)

        realtime_bars = self.broker.get_realtime_bars(
            assets, frequency)

        # Broker.get_realtime_history() returns the asset as level 0 column,
        # open, high, low, close, volume returned as level 1 columns.
        # To filter for field the levels needs to be swapped
        realtime_bars = realtime_bars.swaplevel(0, 1, axis=1)

        ohlcv_field = 'close' if field == 'price' else field

        # TODO: end_dt is ignored when historical & realtime bars are merged.
        # Should not cause issues as end_dt is set to current time in live
        # trading, but would be more proper if merge would make use of it.
        combined_bars = historical_bars.combine_first(
            realtime_bars[ohlcv_field])

        if ffill and field == 'price':
            # Simple forward fill is not enough here as the last ingested
            # value might be outside of the requested time window. That case
            # the time series starts with NaN and forward filling won't help.
            # To provide values for such cases we backward fill.
            # Backward fill as a second operation will have no effect if the
            # forward-fill was successful.
            combined_bars.fillna(method='ffill', inplace=True)
            combined_bars.fillna(method='bfill', inplace=True)

        return combined_bars[-bar_count:]
