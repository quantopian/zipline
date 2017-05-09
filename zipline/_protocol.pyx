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
import warnings
from contextlib import contextmanager
from functools import wraps

from pandas.tslib import normalize_date
import pandas as pd
import numpy as np

from six import iteritems, PY2, string_types
from cpython cimport bool
from collections import Iterable

from zipline.assets import (Asset,
                            AssetConvertible,
                            PricingDataAssociable,
                            Future)
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.zipline_warnings import ZiplineDeprecationWarning


cdef bool _is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, string_types)


# Wraps doesn't work for method objects in python2. Docs should be generated
# with python3 so it is not a big deal.
if PY2:
    def no_wraps_py2(f):
        def dec(g):
            return g
        return dec
else:
    no_wraps_py2 = wraps


cdef class check_parameters(object):
    """
    Asserts that the keywords passed into the wrapped function are included
    in those passed into this decorator. If not, raise a TypeError with a
    meaningful message, unlike the one Cython returns by default.

    Also asserts that the arguments passed into the wrapped function are
    consistent with the types passed into this decorator. If not, raise a
    TypeError with a meaningful message.
    """
    cdef tuple keyword_names
    cdef tuple types
    cdef dict keys_to_types

    def __init__(self, keyword_names, types):
        self.keyword_names = keyword_names
        self.types = types

        self.keys_to_types = dict(zip(keyword_names, types))

    def __call__(self, func):
        @no_wraps_py2(func)
        def assert_keywords_and_call(*args, **kwargs):
            cdef short i

            # verify all the keyword arguments
            for field in kwargs:
                if field not in self.keyword_names:
                    raise TypeError("%s() got an unexpected keyword argument"
                                    " '%s'" % (func.__name__, field))

            # verify type of each argument
            for i, arg in enumerate(args[1:]):
                expected_type = self.types[i]

                if (i == 0 or i == 1) and _is_iterable(arg):
                    if len(arg) == 0:
                        continue
                    arg = arg[0]

                if not isinstance(arg, expected_type):
                    expected_type_name = expected_type.__name__ \
                        if not _is_iterable(expected_type) \
                        else ', '.join([type_.__name__ for type_ in expected_type])

                    raise TypeError("Expected %s argument to be of type %s%s" %
                        (self.keyword_names[i],
                         'or iterable of type ' if i in (0, 1) else '',
                         expected_type_name)
                    )

            # verify type of each kwarg
            for keyword, arg in iteritems(kwargs):
                if keyword in ('assets', 'fields') and _is_iterable(arg):
                    if len(arg) == 0:
                        continue
                    arg = arg[0]
                if not isinstance(arg, self.keys_to_types[keyword]):
                    expected_type = self.keys_to_types[keyword].__name__ \
                        if not _is_iterable(self.keys_to_types[keyword]) \
                        else ', '.join([type_.__name__ for type_ in
                            self.keys_to_types[keyword]])

                    raise TypeError("Expected %s argument to be of type %s%s" %
                                    (keyword,
                                     'or iterable of type ' if keyword in
                                     ('assets', 'fields') else '',
                                     expected_type)
                    )

            return func(*args, **kwargs)

        return assert_keywords_and_call


@contextmanager
def handle_non_market_minutes(bar_data):
    try:
        bar_data._handle_non_market_minutes = True
        yield
    finally:
        bar_data._handle_non_market_minutes = False


cdef class BarData:
    """
    Provides methods to access spot value or history windows of price data.
    Also provides some utility methods to determine if an asset is alive,
    has recent trade data, etc.

    This is what is passed as ``data`` to the ``handle_data`` function.

    Parameters
    ----------
    data_portal : DataPortal
        Provider for bar pricing data.
    simulation_dt_func : callable
        Function which returns the current simulation time.
        This is usually bound to a method of TradingSimulation.
    data_frequency : {'minute', 'daily'}
        The frequency of the bar data; i.e. whether the data is
        daily or minute bars
    restrictions : zipline.finance.asset_restrictions.Restrictions
        Object that combines and returns restricted list information from
        multiple sources
    universe_func : callable, optional
        Function which returns the current 'universe'.  This is for
        backwards compatibility with older API concepts.
    """
    cdef object data_portal
    cdef object simulation_dt_func
    cdef object data_frequency
    cdef object restrictions
    cdef dict _views
    cdef object _universe_func
    cdef object _last_calculated_universe
    cdef object _universe_last_updated_at
    cdef bool _daily_mode
    cdef object _trading_calendar
    cdef object _is_restricted

    cdef bool _adjust_minutes

    def __init__(self, data_portal, simulation_dt_func, data_frequency,
                 trading_calendar, restrictions, universe_func=None):
        self.data_portal = data_portal
        self.simulation_dt_func = simulation_dt_func
        self.data_frequency = data_frequency
        self._views = {}

        self._daily_mode = (self.data_frequency == "daily")

        self._universe_func = universe_func
        self._last_calculated_universe = None
        self._universe_last_updated_at = None

        self._adjust_minutes = False

        self._trading_calendar = trading_calendar
        self._is_restricted = restrictions.is_restricted

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
        SidView : Accessor into the given asset's data.
        """
        try:
            self._warn_deprecated("`data[sid(N)]` is deprecated. Use "
                            "`data.current`.")
            view = self._views[asset]
        except KeyError:
            try:
                asset = self.data_portal.asset_finder.retrieve_asset(asset)
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

    cdef _get_current_minute(self):
        """
        Internal utility method to get the current simulation time.

        Possible answers are:
        - whatever the algorithm's get_datetime() method returns (this is what
            `self.simulation_dt_func()` points to)
        - sometimes we're knowingly not in a market minute, like if we're in
            before_trading_start.  In that case, `self._adjust_minutes` is
            True, and we get the previous market minute.
        - if we're in daily mode, get the session label for this minute.
        """
        dt = self.simulation_dt_func()

        if self._adjust_minutes:
            dt = \
                self.data_portal.trading_calendar.previous_minute(dt)

        if self._daily_mode:
            # if we're in daily mode, take the given dt (which is the last
            # minute of the session) and get the session label for it.
            dt = self.data_portal.trading_calendar.minute_to_session_label(dt)

        return dt

    @check_parameters(('assets', 'fields'),
                      ((Asset, ContinuousFuture) + string_types, string_types))
    def current(self, assets, fields):
        """
        Returns the current value of the given assets for the given fields
        at the current simulation time.  Current values are the as-traded price
        and are usually not adjusted for events like splits or dividends (see
        notes for more information).

        Parameters
        ----------
        assets : Asset or iterable of Assets
        fields : str or iterable[str].
            Valid values are: "price",
            "last_traded", "open", "high", "low", "close", "volume", or column
            names in files read by ``fetch_csv``.

        Returns
        -------
        current_value : Scalar, pandas Series, or pandas DataFrame.
            See notes below.

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

        If the current simulation time is not a valid market time, we use the
        last market close instead.

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
        multiple_assets = _is_iterable(assets)
        multiple_fields = _is_iterable(fields)

        # There's some overly verbose code in here, particularly around
        # 'do something if self._adjust_minutes is False, otherwise do
        # something else'. This could be less verbose, but the 99% case is that
        # `self._adjust_minutes` is False, so it's important to keep that code
        # path as fast as possible.

        # There's probably a way to make this method (and `history`) less
        # verbose, but this is OK for now.

        if not multiple_assets:
            asset = assets

            if not multiple_fields:
                field = fields

                # return scalar value
                if not self._adjust_minutes:
                    return self.data_portal.get_spot_value(
                        asset,
                        field,
                        self._get_current_minute(),
                        self.data_frequency
                    )
                else:
                    return self.data_portal.get_adjusted_value(
                        asset,
                        field,
                        self._get_current_minute(),
                        self.simulation_dt_func(),
                        self.data_frequency
                    )
            else:
                # assume fields is iterable
                # return a Series indexed by field
                if not self._adjust_minutes:
                    return pd.Series(data={
                        field: self.data_portal.get_spot_value(
                                    asset,
                                    field,
                                    self._get_current_minute(),
                                    self.data_frequency
                               )
                        for field in fields
                    }, index=fields, name=assets.symbol)
                else:
                    return pd.Series(data={
                        field: self.data_portal.get_adjusted_value(
                                    asset,
                                    field,
                                    self._get_current_minute(),
                                    self.simulation_dt_func(),
                                    self.data_frequency
                               )
                        for field in fields
                    }, index=fields, name=assets.symbol)
        else:
            if not multiple_fields:
                field = fields

                # assume assets is iterable
                # return a Series indexed by asset
                if not self._adjust_minutes:
                    return pd.Series(data={
                        asset: self.data_portal.get_spot_value(
                                    asset,
                                    field,
                                    self._get_current_minute(),
                                    self.data_frequency
                               )
                        for asset in assets
                        }, index=assets, name=fields)
                else:
                    return pd.Series(data={
                        asset: self.data_portal.get_adjusted_value(
                                    asset,
                                    field,
                                    self._get_current_minute(),
                                    self.simulation_dt_func(),
                                    self.data_frequency
                               )
                        for asset in assets
                        }, index=assets, name=fields)

            else:
                # both assets and fields are iterable
                data = {}

                if not self._adjust_minutes:
                    for field in fields:
                        series = pd.Series(data={
                            asset: self.data_portal.get_spot_value(
                                        asset,
                                        field,
                                        self._get_current_minute(),
                                        self.data_frequency
                                   )
                            for asset in assets
                            }, index=assets, name=field)
                        data[field] = series
                else:
                    for field in fields:
                        series = pd.Series(data={
                            asset: self.data_portal.get_adjusted_value(
                                        asset,
                                        field,
                                        self._get_current_minute(),
                                        self.simulation_dt_func(),
                                        self.data_frequency
                                   )
                            for asset in assets
                            }, index=assets, name=field)
                        data[field] = series

                return pd.DataFrame(data)

    @check_parameters(('continuous_future',),
                      (ContinuousFuture,))
    def current_chain(self, continuous_future):
        return self.data_portal.get_current_future_chain(
            continuous_future,
            self.simulation_dt_func())

    @check_parameters(('assets',), (Asset,))
    def can_trade(self, assets):
        """
        For the given asset or iterable of assets, returns true if all of the
        following are true:
        1) the asset is alive for the session of the current simulation time
          (if current simulation time is not a market minute, we use the next
          session)
        2) (if we are in minute mode) the asset's exchange is open at the
          current simulation time or at the simulation calendar's next market
          minute
        3) there is a known last price for the asset.

        Notes
        -----
        The second condition above warrants some further explanation.
        - If the asset's exchange calendar is identical to the simulation
        calendar, then this condition always returns True.
        - If there are market minutes in the simulation calendar outside of
        this asset's exchange's trading hours (for example, if the simulation
        is running on the CME calendar but the asset is MSFT, which trades on
        the NYSE), during those minutes, this condition will return false
        (for example, 3:15 am Eastern on a weekday, during which the CME is
        open but the NYSE is closed).

        Parameters
        ----------
        assets: Asset or iterable of assets

        Returns
        -------
        can_trade : bool or pd.Series[bool] indexed by asset.
        """
        dt = self.simulation_dt_func()

        if self._adjust_minutes:
            adjusted_dt = self._get_current_minute()
        else:
            adjusted_dt = dt

        data_portal = self.data_portal

        if isinstance(assets, Asset):
            return self._can_trade_for_asset(
                assets, dt, adjusted_dt, data_portal
            )
        else:
            tradeable = [
                self._can_trade_for_asset(
                    asset, dt, adjusted_dt, data_portal
                )
                for asset in assets
            ]
            return pd.Series(data=tradeable, index=assets, dtype=bool)

    cdef bool _can_trade_for_asset(self, asset, dt, adjusted_dt, data_portal):
        cdef object session_label
        cdef object dt_to_use_for_exchange_check,

        if self._is_restricted(asset, adjusted_dt):
            return False

        session_label = self._trading_calendar.minute_to_session_label(dt)

        if not asset.is_alive_for_session(session_label):
            # asset isn't alive
            return False

        if asset.auto_close_date and session_label >= asset.auto_close_date:
            return False

        if not self._daily_mode:
            # Find the next market minute for this calendar, and check if this
            # asset's exchange is open at that minute.
            if self._trading_calendar.is_open_on_minute(dt):
                dt_to_use_for_exchange_check = dt
            else:
                dt_to_use_for_exchange_check = \
                    self._trading_calendar.next_open(dt)

            if not asset.is_exchange_open(dt_to_use_for_exchange_check):
                return False

        # is there a last price?
        return not np.isnan(
            data_portal.get_spot_value(
                asset, "price", adjusted_dt, self.data_frequency
            )
        )

    @check_parameters(('assets',), (Asset,))
    def is_stale(self, assets):
        """
        For the given asset or iterable of assets, returns true if the asset
        is alive and there is no trade data for the current simulation time.

        If the asset has never traded, returns False.

        If the current simulation time is not a valid market time, we use the
        current time to check if the asset is alive, but we use the last
        market minute/day for the trade data check.

        Parameters
        ----------
        assets: Asset or iterable of assets

        Returns
        -------
        boolean or Series of booleans, indexed by asset.
        """
        dt = self.simulation_dt_func()
        if self._adjust_minutes:
            adjusted_dt = self._get_current_minute()
        else:
            adjusted_dt = dt

        data_portal = self.data_portal

        if isinstance(assets, Asset):
            return self._is_stale_for_asset(
                assets, dt, adjusted_dt, data_portal
            )
        else:
            return pd.Series(data={
                asset: self._is_stale_for_asset(
                    asset, dt, adjusted_dt, data_portal
                )
                for asset in assets
            })

    cdef bool _is_stale_for_asset(self, asset, dt, adjusted_dt, data_portal):
        session_label = normalize_date(dt) # FIXME

        if not asset.is_alive_for_session(session_label):
            return False

        current_volume = data_portal.get_spot_value(
            asset, "volume",  adjusted_dt, self.data_frequency
        )

        if current_volume > 0:
            # found a current value, so we know this asset is not stale.
            return False
        else:
            # we need to distinguish between if this asset has ever traded
            # (stale = True) or has never traded (stale = False)
            last_traded_dt = \
                data_portal.get_spot_value(asset, "last_traded", adjusted_dt,
                                           self.data_frequency)

            return not (last_traded_dt is pd.NaT)

    @check_parameters(('assets', 'fields', 'bar_count',
                       'frequency'),
                      ((Asset, ContinuousFuture) + string_types, string_types,
                       int,
                       string_types))
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
        history : Series or DataFrame or Panel
            Return type depends on the dimensionality of the 'assets' and
            'fields' parameters.

            If single asset and field are passed in, the returned Series is
            indexed by dt.

            If multiple assets and single field are passed in, the returned
            DataFrame is indexed by dt, and has assets as columns.

            If a single asset and multiple fields are passed in, the returned
            DataFrame is indexed by dt, and has fields as columns.

            If multiple assets and multiple fields are passed in, the returned
            Panel is indexed by field, has dt as the major axis, and assets
            as the minor axis.

        Notes
        -----
        If the current simulation time is not a valid market time, we use the
        last market close instead.
        """
        if isinstance(fields, string_types):
            single_asset = isinstance(assets, PricingDataAssociable)

            if single_asset:
                asset_list = [assets]
            else:
                asset_list = assets

            df = self.data_portal.get_history_window(
                asset_list,
                self._get_current_minute(),
                bar_count,
                frequency,
                fields,
                self.data_frequency,
            )

            if self._adjust_minutes:
                adjs = self.data_portal.get_adjustments(
                    assets,
                    fields,
                    self._get_current_minute(),
                    self.simulation_dt_func()
                )

                df = df * adjs

            if single_asset:
                # single asset, single field, return a series.
                return df[assets]
            else:
                # multiple assets, single field, return a dataframe whose
                # columns are the assets, indexed by dt.
                return df
        else:
            if isinstance(assets, PricingDataAssociable):
                # one asset, multiple fields. for now, just make multiple
                # history calls, one per field, then stitch together the
                # results. this can definitely be optimized!

                df_dict = {
                    field: self.data_portal.get_history_window(
                        [assets],
                        self._get_current_minute(),
                        bar_count,
                        frequency,
                        field,
                        self.data_frequency,
                    )[assets] for field in fields
                }

                if self._adjust_minutes:
                    adjs = {
                        field: self.data_portal.get_adjustments(
                            assets,
                            field,
                            self._get_current_minute(),
                            self.simulation_dt_func()
                        )[0] for field in fields
                    }

                    df_dict = {field: df * adjs[field]
                               for field, df in iteritems(df_dict)}

                # returned dataframe whose columns are the fields, indexed by
                # dt.
                return pd.DataFrame(df_dict)

            else:
                df_dict = {
                    field: self.data_portal.get_history_window(
                        assets,
                        self._get_current_minute(),
                        bar_count,
                        frequency,
                        field,
                        self.data_frequency,
                    ) for field in fields
                }

                if self._adjust_minutes:
                    adjs = {
                        field: self.data_portal.get_adjustments(
                            assets,
                            field,
                            self._get_current_minute(),
                            self.simulation_dt_func()
                        ) for field in fields
                    }

                    df_dict = {field: df * adjs[field]
                               for field, df in iteritems(df_dict)}

                # returned panel has:
                # items: fields
                # major axis: dt
                # minor axis: assets
                return pd.Panel(df_dict)

    property current_dt:
        def __get__(self):
            return self.simulation_dt_func()

    @property
    def fetcher_assets(self):
        return self.data_portal.get_fetcher_assets(self.simulation_dt_func())

    property _handle_non_market_minutes:
        def __set__(self, val):
            self._adjust_minutes = val

    property current_session:
        def __get__(self):
            return self._trading_calendar.minute_to_session_label(
                self.simulation_dt_func(),
                direction="next"
            )

    property current_session_minutes:
        def __get__(self):
            return self._trading_calendar.minutes_for_session(
                self.current_session
            )

    #################
    # OLD API SUPPORT
    #################
    cdef _calculate_universe(self):
        if self._universe_func is None:
            return []

        simulation_dt = self.simulation_dt_func()
        if self._last_calculated_universe is None or \
                self._universe_last_updated_at != simulation_dt:

            self._last_calculated_universe = self._universe_func()
            self._universe_last_updated_at = simulation_dt

        return self._last_calculated_universe

    def __iter__(self):
        self._warn_deprecated("Iterating over the assets in `data` is "
                        "deprecated.")
        for asset in self._calculate_universe():
            yield asset

    def __contains__(self, asset):
        self._warn_deprecated("Checking whether an asset is in data is "
                        "deprecated.")
        universe = self._calculate_universe()
        return asset in universe

    def items(self):
        self._warn_deprecated("Iterating over the assets in `data` is "
                        "deprecated.")
        return [(asset, self[asset]) for asset in self._calculate_universe()]

    def iteritems(self):
        self._warn_deprecated("Iterating over the assets in `data` is "
                        "deprecated.")
        for asset in self._calculate_universe():
            yield asset, self[asset]

    def __len__(self):
        self._warn_deprecated("Iterating over the assets in `data` is "
                        "deprecated.")

        return len(self._calculate_universe())

    def keys(self):
        self._warn_deprecated("Iterating over the assets in `data` is "
                        "deprecated.")

        return list(self._calculate_universe())

    def iterkeys(self):
        return iter(self.keys())

    def __getitem__(self, name):
        return self._get_equity_price_view(name)

    cdef _warn_deprecated(self, msg):
        warnings.warn(
            msg,
            category=ZiplineDeprecationWarning,
            stacklevel=1
        )

cdef class SidView:
    cdef object asset
    cdef object data_portal
    cdef object simulation_dt_func
    cdef object data_frequency

    """
    This class exists to temporarily support the deprecated data[sid(N)] API.
    """
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
        elif column == "dt":
            return self.dt
        elif column == "datetime":
            return self.datetime
        elif column == "sid":
            return self.sid

        return self.data_portal.get_spot_value(
            self.asset,
            column,
            self.simulation_dt_func(),
            self.data_frequency
        )

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

    def mavg(self, num_minutes):
        self._warn_deprecated("The `mavg` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "mavg", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def stddev(self, num_minutes):
        self._warn_deprecated("The `stddev` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "stddev", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def vwap(self, num_minutes):
        self._warn_deprecated("The `vwap` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "vwap", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def returns(self):
        self._warn_deprecated("The `returns` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "returns", self.simulation_dt_func(),
            self.data_frequency
        )

    cdef _warn_deprecated(self, msg):
        warnings.warn(
            msg,
            category=ZiplineDeprecationWarning,
            stacklevel=1
        )
