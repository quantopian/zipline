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

from logbook import Logger

log = Logger('DataPortal')

BASE_FIELDS = {
    'open': 'open',
    'open_price': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'close_price': 'close',
    'volume': 'volume',
    'price': 'close'
}


class DataPortal(object):
    def __init__(self,
                 env,
                 equity_daily_reader=None,
                 equity_minute_reader=None,
                 future_daily_reader=None,
                 future_minute_reader=None,
                 adjustment_reader=None):

        self._adjustment_reader = adjustment_reader

        self._equity_daily_reader = equity_daily_reader
        self._equity_minute_reader = equity_minute_reader
        self._future_daily_reader = future_daily_reader
        self._future_minute_reader = future_minute_reader

    def get_previous_value(self, asset, field, dt, data_frequency):
        """
        Given an asset and a column and a dt, returns the previous value for
        the same asset/column pair.  If this data portal is in minute mode,
        it's the previous minute value, otherwise it's the previous day's
        value.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.

        field: string
            The desired field of the asset.  Valid values are "open",
            "open_price", "high", "low", "close", "close_price", "volume", and
            "price".

        dt: pd.Timestamp
            The timestamp from which to go back in time one slot.

        data_frequency: string
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        The value of the desired field at the desired time.
        """
        raise NotImplementedError

    def get_spot_value(self, asset, field, dt, data_frequency):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.gith

        field: string
            The desired field of the asset.  Valid values are "open",
            "open_price", "high", "low", "close", "close_price", "volume", and
            "price".

        dt: pd.Timestamp
            The timestamp for the desired value.

        data_frequency: string
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        The value of the desired field at the desired time.
        """
        raise NotImplementedError

    def get_history_window(self, assets, end_dt, bar_count, frequency, field,
                           ffill=True):
        """
        Public API method that returns a dataframe containing the requested
        history window.  Data is fully adjusted.

        Parameters
        ---------
        assets : list of zipline.data.Asset objects
            The assets whose data is desired.

        bar_count: int
            The number of bars desired.

        frequency: string
            "1d" or "1m"

        field: string
            The desired field of the asset.

        ffill: boolean
            Forward-fill missing values. Only has effect if field
            is 'price'.

        Returns
        -------
        A dataframe containing the requested data.
        """
        raise NotImplementedError

    def get_splits(self, sids, dt):
        """
        Returns any splits for the given sids and the given dt.

        Parameters
        ----------
        sids : list
            Sids for which we want splits.

        dt: pd.Timestamp
            The date for which we are checking for splits.  Note: this is
            expected to be midnight UTC.

        Returns
        -------
        list: List of splits, where each split is a (sid, ratio) tuple.
        """
        raise NotImplementedError

    def get_stock_dividends(self, sid, trading_days):
        """
        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_days: pd.DatetimeIndex
            The trading range.

        Returns
        -------
        list: A list of objects with all relevant attributes populated.
        All timestamp fields are converted to pd.Timestamps.
        """
        raise NotImplementedError

    def get_fetcher_assets(self, day):
        """
        Returns a list of assets for the current date, as defined by the
        fetcher data.

        Notes
        -----
        Data is forward-filled.  If there is no fetcher data defined for day
        N, we use day N-1's data (if available, otherwise we keep going back).

        Returns
        -------
        list: a list of Asset objects.
        """
        raise NotImplementedError
