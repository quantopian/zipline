#
# Copyright 2016 Quantopian, Inc.
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

from pandas.tslib import normalize_date
import pandas as pd
import numpy as np

from cpython cimport bool

from zipline.assets import Asset

cdef class BarData:
    """
    Provides methods to access spot value or history windows of price data.
    Also provides some utility methods to determine if an asset is alive,
    has recent trade data, etc.

    This is what is passed as `data` to the `handle_data` function.
    """
    def __init__(self, data_portal, simulation_dt_func, data_frequency):
        """
        Parameters
        ---------
        data_portal : DataPortal
            Provider for bar pricing data.

        simulation_dt_func: function
            Function which returns the current simulation time.
            This is usually bound to a method of TradingSimulation.

        data_frequency: string
            The frequency of the bar data; i.e. whether the data is
            'daily' or 'minute' bars
        """
        self.data_portal = data_portal
        self.simulation_dt_func = simulation_dt_func
        self.data_frequency = data_frequency
        self._views = {}

    cdef _get_equity_price_view(self, asset):
        """
        Returns a DataPortalSidView for the given asset.  Used to support the
        data[sid(N)] public API.  Not needed if DataPortal is used standalone.

        Parameters
        ----------
        asset : Asset
            Asset that is being queried.

        Returns
        -------
        SidView: Accessor into the given asset's data.
        """
        try:
            view = self._views[asset]
        except KeyError:
            try:
                asset = self.data_portal.env.asset_finder.retrieve_asset(asset)
            except ValueError:
                # assume fetcher
                pass
            view = self._views[asset] = self._create_sid_view(asset)

        return view

    cdef _create_sid_view(self, asset):
        return SidView(
            asset,
            self.data_portal,
            self.simulation_dt_func,
            self.data_frequency
        )

    def spot_value(self, assets, fields):
        """
        Returns the spot value of the given assets for the given fields
        at the current simulation time.  Spot values are the as-traded price
        and are usually not adjusted for events like splits or dividends (see
        notes for more information).

        Parameters
        ----------
        assets : Asset or iterable of Assets

        fields : string or iterable of strings.  Valid values are: "price",
            "last_traded", "open", "high", "low", "close", "volume"

        Returns
        -------
        Scalar, pandas Series, or pandas DataFrame.  See notes
        below.

        Notes
        -----
        If a single asset and a single field are passed in, a scalar float
        value is returned.

        If a single asset and a list of fields are passed in, a pandas Series
        is returned whose indices are the fields, and whose values are scalar
        values for this asset for each field.

        If a list of assets and a single field are passed in, a pandas Series
        is returned whose indices are the assets, and whose values are scalar
        values for each asset for the given field.

        If a list of assets and a list of fields are passed in, a pandas
        DataFrame is returned, indexed by asset.  The columns are the requested
        fields, filled with the scalar values for each asset for each field.

        "price" returns the last known close price of the asset.  If there is
        no last known value (either because the asset has never traded, or
        because it has delisted) NaN is returned.  If a value is found, and we
        had to cross an adjustment boundary (split, dividend, etc) to get it,
        the value is adjusted before being returned.

        "last_traded" returns the date of the last trade event of the asset,
        even if the asset has stopped trading. If there is no last known value,
        pd.NaT is returned.

        "volume" returns the trade volume for the current simulation time.  If
        there is no trade this minute, 0 is returned.

        "open", "high", "low", and "close" return the relevant information for
        the current trade bar.  If there is no current trade bar, NaN is
        returned.
        """
        if isinstance(assets, Asset):
            asset = assets

            if isinstance(fields, str):
                field = fields

                # return scalar value
                return self.data_portal.get_spot_value(
                    asset,
                    field,
                    self.simulation_dt_func(),
                    self.data_frequency
                )
            else:
                # assume fields is iterable
                # return a Series indexed by field
                return pd.Series(data={
                    field: self.data_portal.get_spot_value(
                        asset,
                        field,
                        self.simulation_dt_func(),
                        self.data_frequency)
                    for field in fields
                }, index=fields, name=assets.symbol)
        else:
            if isinstance(fields, str):
                field = fields

                # assume assets is iterable
                # return a Series indexed by asset
                return pd.Series(data={
                    asset: self.data_portal.get_spot_value(
                        asset,
                        field,
                        self.simulation_dt_func(),
                        self.data_frequency)
                    for asset in assets
                    }, index=assets, name=fields)
            else:
                # both assets and fields are iterable
                data = {}

                for field in fields:
                    series = pd.Series(data={
                        asset: self.data_portal.get_spot_value(
                            asset,
                            field,
                            self.simulation_dt_func(),
                            self.data_frequency)
                        for asset in assets
                        }, index=assets, name=field)

                    data[field] = series

                return pd.DataFrame(data)

    def can_trade(self, assets):
        """
        For the given asset or iterable of assets, returns true if the asset
        is alive at the current simulation time and there is a known last
        price.

        Parameters
        ----------
        assets: Asset or iterable of assets

        Returns
        -------
        boolean or Series of booleans, indexed by asset.
        """
        dt = self.simulation_dt_func()
        data_portal = self.data_portal

        if isinstance(assets, Asset):
            return self._can_trade_for_asset(assets, dt, data_portal)
        else:
            return pd.Series(data={
                asset: self._can_trade_for_asset(asset, dt, data_portal)
                for asset in assets
            })

    cdef bool _can_trade_for_asset(self, asset, dt, data_portal):
        if asset.start_date <= dt <= asset.end_date:
            # is there a last price?
            return not np.isnan(
                data_portal.get_spot_value(
                    asset, "price", dt, self.data_frequency
                )
            )

        return False

    def is_stale(self, assets):
        """
        For the given asset or iterable of assets, returns true if the asset
        is alive and there is no trade data for the current simulation time.

        If the asset has never traded, returns False.

        Parameters
        ----------
        assets: Asset or iterable of assets

        Returns
        -------
        boolean or Series of booleans, indexed by asset.
        """
        dt = self.simulation_dt_func()
        data_portal = self.data_portal

        if isinstance(assets, Asset):
            return self._is_stale_for_asset(assets, dt, data_portal)
        else:
            return pd.Series(data={
                asset: self._is_stale_for_asset(asset, dt, data_portal)
                for asset in assets
            })

    cdef bool _is_stale_for_asset(self, asset, dt, data_portal):
        if asset.start_date > dt:
            return False

        if asset.end_date <= dt:
            return False

        current_volume = data_portal.get_spot_value(asset, "volume", dt,
                                                     self.data_frequency)

        if current_volume > 0:
            # found a current value, so we know this asset is not stale.
            return False
        else:
            # we need to distinguish between if this asset has ever traded
            # (stale = True) or has never traded (stale = False)
            last_traded_dt = \
                data_portal.get_spot_value(asset, "last_traded", dt,
                                           self.data_frequency)

            return not (last_traded_dt is pd.NaT)

    def history(self, assets, fields, bar_count, frequency):
        """
        Returns a window of data for the given assets and fields.

        This data is adjusted for splits, dividends, and mergers as of the
        current algorithm time.

        The semantics of missing data are identical to the ones described in
        the notes for `get_spot_value`.

        Parameters
        ----------
        assets: Asset or iterable of Asset

        fields: string or iterable of string.  Valid values are "open", "high",
            "low", "close", "volume", "price", and "last_traded".

        bar_count: integer number of bars of trade data

        frequency: string. "1m" for minutely data or "1d" for daily date

        Returns
        -------
        Series or DataFrame or Panel, depending on the dimensionality of
            the 'assets' and 'fields' parameters.

            If single asset and field are passed in, the returned Series is
            indexed by dt.

            If multiple assets and single field are passed in, the returned
            DataFrame is indexed by dt, and has assets as columns.

            If a single asset and multiple fields are passed in, the returned
            DataFrame is indexed by dt, and has fields as columns.

            If multiple assets and multiple fields are passed in, the returned
            Panel is indexed by field, has dt as the major axis, and assets
            as the minor axis.
        """
        if isinstance(fields, str):
            single_asset = isinstance(assets, Asset)

            if single_asset:
                asset_list = [assets]
            else:
                asset_list = assets

            df = self.data_portal.get_history_window(
                asset_list,
                self.simulation_dt_func(),
                bar_count,
                frequency,
                fields
            )

            if single_asset:
                # single asset, single field, return a series.
                return df[assets]
            else:
                # multiple assets, single field, return a dataframe whose
                # columns are the assets, indexed by dt.
                return df
        else:
            if isinstance(assets, Asset):
                # one asset, multiple fields. for now, just make multiple
                # history calls, one per field, then stitch together the
                # results. this can definitely be optimized!

                # returned dataframe whose columns are the fields, indexed by
                # dt.
                return pd.DataFrame({
                    field: self.data_portal.get_history_window(
                        [assets],
                        self.simulation_dt_func(),
                        bar_count,
                        frequency,
                        field
                    )[assets] for field in fields
                })
            else:
                df_dict = {
                    field: self.data_portal.get_history_window(
                        assets,
                        self.simulation_dt_func(),
                        bar_count,
                        frequency,
                        field
                    ) for field in fields
                }

                # returned panel has:
                # items: fields
                # major axis: dt
                # minor axis: assets
                return pd.Panel(df_dict)

    def __iter__(self):
        raise ValueError("'BarData' object is not iterable")

    def __contains__(self, name):
        raise ValueError("'BarData' object is not iterable")

    def __getitem__(self, name):
        return self._get_equity_price_view(name)

    property current_dt:
        def __get__(self):
            return self.simulation_dt_func()

    @property
    def fetcher_assets(self):
        return self.data_portal.get_fetcher_assets(
            normalize_date(self.simulation_dt_func())
        )


cdef class SidView:
    def __init__(self, asset, data_portal, simulation_dt_func, data_frequency):
        """
        Parameters
        ---------
        asset : Asset
            The asset for which the instance retrieves data.

        data_portal : DataPortal
            Provider for bar pricing data.

        simulation_dt_func: function
            Function which returns the current simulation time.
            This is usually bound to a method of TradingSimulation.

        data_frequency: string
            The frequency of the bar data; i.e. whether the data is
            'daily' or 'minute' bars
        """
        self.asset = asset
        self.data_portal = data_portal
        self.simulation_dt_func = simulation_dt_func
        self.data_frequency = data_frequency

    def __getattr__(self, column):
        # backwards compatibility code for Q1 API
        if column == "close_price":
            column = "close"
        elif column == "open_price":
            column = "open"

        return self.data_portal.get_spot_value(
            self.asset,
            column,
            self.simulation_dt_func(),
            self.data_frequency)

    def __contains__(self, column):
        return self.data_portal.contains(self.asset, column)

    def __getitem__(self, column):
        return self.__getattr__(column)

    property sid:
        def __get__(self):
            return self.asset

    property dt:
        def __get__(self):
            return self.datetime

    property datetime:
        def __get__(self):
            return self.data_portal.get_last_traded_dt(
                self.asset,
                self.simulation_dt_func(),
                self.data_frequency)

    property current_dt:
        def __get__(self):
            return self.simulation_dt_func()