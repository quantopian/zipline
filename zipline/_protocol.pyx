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

from pandas.tslib import normalize_date


cdef class BarData:
    """
    Holds the event data for all sids for a given dt.

    This is what is passed as `data` to the `handle_data` function.
    """
    def __init__(self, data_portal, simulator):
        self.data_portal = data_portal
        self.simulator = simulator
        self.data_frequency = simulator.sim_params.data_frequency
        self._views = {}

    def _get_equity_price_view(self, asset):
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
            view = self._views[asset] = self._create_sid_view(asset)

        return view

    def _create_sid_view(self, asset):
        return SidView(
            asset,
            self.data_portal,
            self.simulator,
            self.data_frequency
        )

    def __iter__(self):
        raise ValueError("'BarData' object is not iterable")

    def __contains__(self, name):
        raise ValueError("'BarData' object is not iterable")

    def __getitem__(self, name):
        return self._get_equity_price_view(name)

    @property
    def fetcher_assets(self):
        return self.data_portal.get_fetcher_assets(
            normalize_date(self.simulator.simulation_dt)
        )

cdef class SidView:
    def __init__(self, asset, data_portal, simulator, data_frequency):
        self.asset = asset
        self.data_portal = data_portal
        self.simulator = simulator
        self.data_frequency = data_frequency

    def __getattr__(self, column):
        return self.data_portal.get_spot_value(
            self.asset, column,
            self.simulator.simulation_dt,
            self.data_frequency)

    def __contains__(self, column):
        return self.data_portal.contains(self.asset, column)

    def __getitem__(self, column):
        return self.__getattr__(column)
