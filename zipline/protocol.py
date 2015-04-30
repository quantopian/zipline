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

from copy import copy

from six import iteritems, iterkeys
import pandas as pd
import numpy as np

from . utils.protocol_utils import Enum
from . utils.math_utils import nanstd, nanmean, nansum

from zipline.finance.trading import with_environment
from zipline.utils.algo_instance import get_algo_instance
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = Enum(
    'AS_TRADED_EQUITY',
    'MERGER',
    'SPLIT',
    'DIVIDEND',
    'TRADE',
    'TRANSACTION',
    'ORDER',
    'EMPTY',
    'DONE',
    'CUSTOM',
    'BENCHMARK',
    'COMMISSION'
)

# Expected fields/index values for a dividend Series.
DIVIDEND_FIELDS = [
    'declared_date',
    'ex_date',
    'gross_amount',
    'net_amount',
    'pay_date',
    'payment_sid',
    'ratio',
    'sid',
]
# Expected fields/index values for a dividend payment Series.
DIVIDEND_PAYMENT_FIELDS = [
    'id',
    'payment_sid',
    'cash_amount',
    'share_count'
]


def dividend_payment(data=None):
    """
    Take a dictionary whose values are in DIVIDEND_PAYMENT_FIELDS and return a
    series representing the payment of a dividend.

    Ids are assigned to each historical dividend in
    PerformanceTracker.update_dividends. They are guaranteed to be unique
    integers with the context of a single simulation. If @data is non-empty, a
    id is required to identify the historical dividend associated with this
    payment.

    Additionally, if @data is non-empty, either data['cash_amount'] should be
    nonzero or data['payment_sid'] should be an asset identifier and
    data['share_count'] should be nonzero.

    The returned Series is given its id value as a name so that concatenating
    payments results in a DataFrame indexed by id.  (Note, however, that the
    name value is not used to construct an index when this series is returned
    by function passed to `DataFrame.apply`.  In such a case, pandas preserves
    the index of the DataFrame on which `apply` is being called.)
    """
    return pd.Series(
        data=data,
        name=data['id'] if data is not None else None,
        index=DIVIDEND_PAYMENT_FIELDS,
        dtype=object,
    )


class Event(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__ = initial_values

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        delattr(self, name)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        return hasattr(other, '__dict__') and self.__dict__ == other.__dict__

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_series(self, index=None):
        return pd.Series(self.__dict__, index=index)


class Order(Event):
    pass


class Portfolio(object):

    def __init__(self):
        self.capital_used = 0.0
        self.starting_cash = 0.0
        self.portfolio_value = 0.0
        self.pnl = 0.0
        self.returns = 0.0
        self.cash = 0.0
        self.positions = Positions()
        self.start_date = None
        self.positions_value = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        # Have to convert to primitive dict
        state_dict['positions'] = dict(self.positions)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Portfolio saved state is too old.")

        self.positions = Positions()
        self.positions.update(state.pop('positions'))

        self.__dict__.update(state)


class Account(object):
    '''
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    '''

    def __init__(self):
        self.settled_cash = 0.0
        self.accrued_interest = 0.0
        self.buying_power = float('inf')
        self.equity_with_loan = 0.0
        self.total_positions_value = 0.0
        self.regt_equity = 0.0
        self.regt_margin = float('inf')
        self.initial_margin_requirement = 0.0
        self.maintenance_margin_requirement = 0.0
        self.available_funds = 0.0
        self.excess_liquidity = 0.0
        self.cushion = 0.0
        self.day_trades_remaining = float('inf')
        self.leverage = 0.0
        self.net_leverage = 0.0
        self.net_liquidation = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Account({0})".format(self.__dict__)

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Account saved state is too old.")

        self.__dict__.update(state)


class Position(object):

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Position({0})".format(self.__dict__)

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Protocol Position saved state is too old.")

        self.__dict__.update(state)


class Positions(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos


class SIDData(object):
    # Cache some data on the class so that this is shared for all instances of
    # siddata.

    # The dt where we cached the history.
    _history_cache_dt = None
    # _history_cache is a a dict mapping fields to pd.DataFrames. This is the
    # most data we have for a given field for the _history_cache_dt.
    _history_cache = {}

    # This is the cache that is used for returns. This will have a different
    # structure than the other history cache as this is always daily.
    _returns_cache_dt = None
    _returns_cache = None

    # The last dt that we needed to cache the number of minutes.
    _minute_bar_cache_dt = None
    # If we are in minute mode, there is some cost associated with computing
    # the number of minutes that we need to pass to the bar count of history.
    # This will remain constant for a given bar and day count.
    # This maps days to number of minutes.
    _minute_bar_cache = {}

    def __init__(self, sid, initial_values=None):
        self._sid = sid
        self._freqstr = None

        # To check if we have data, we use the __len__ which depends on the
        # __dict__. Because we are foward defining the attributes needed, we
        # need to account for their entrys in the __dict__.
        # We will add 1 because we need to account for the _initial_len entry
        # itself.
        self._initial_len = len(self.__dict__) + 1

        if initial_values:
            self.__dict__.update(initial_values)

    @property
    def datetime(self):
        """
        Provides an alias from data['foo'].datetime -> data['foo'].dt

        `datetime` was previously provided by adding a seperate `datetime`
        member of the SIDData object via a generator that wrapped the incoming
        data feed and added the field to each equity event.

        This alias is intended to be temporary, to provide backwards
        compatibility with existing algorithms, but should be considered
        deprecated, and may be removed in the future.
        """
        return self.dt

    def get(self, name, default=None):
        return self.__dict__.get(name, default)

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def __len__(self):
        return len(self.__dict__) - self._initial_len

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "SIDData({0})".format(self.__dict__)

    def _get_buffer(self, bars, field='price', raw=False):
        """
        Gets the result of history for the given number of bars and field.

        This will cache the results internally.
        """
        cls = self.__class__
        algo = get_algo_instance()

        now = algo.datetime
        if now != cls._history_cache_dt:
            # For a given dt, the history call for this field will not change.
            # We have a new dt, so we should reset the cache.
            cls._history_cache_dt = now
            cls._history_cache = {}

        if field not in self._history_cache \
           or bars > len(cls._history_cache[field][0].index):
            # If we have never cached this field OR the amount of bars that we
            # need for this field is greater than the amount we have cached,
            # then we need to get more history.
            hst = algo.history(
                bars, self._freqstr, field, ffill=True,
            )
            # Assert that the column holds ints, not security objects.
            if not isinstance(self._sid, str):
                hst.columns = hst.columns.astype(int)
            self._history_cache[field] = (hst, hst.values, hst.columns)

        # Slice of only the bars needed. This is because we strore the LARGEST
        # amount of history for the field, and we might request less than the
        # largest from the cache.
        buffer_, values, columns = cls._history_cache[field]
        if raw:
            sid_index = columns.get_loc(self._sid)
            return values[-bars:, sid_index]
        else:
            return buffer_[self._sid][-bars:]

    def _get_bars(self, days):
        """
        Gets the number of bars needed for the current number of days.

        Figures this out based on the algo datafrequency and caches the result.
        This caches the result by replacing this function on the object.
        This means that after the first call to _get_bars, this method will
        point to a new function object.

        """
        def daily_get_max_bars(days):
            return days

        def minute_get_max_bars(days):
            # max number of minute. regardless of current days or short
            # sessions
            return days * 390

        def daily_get_bars(days):
            return days

        @with_environment()
        def minute_get_bars(days, env=None):
            cls = self.__class__

            now = get_algo_instance().datetime
            if now != cls._minute_bar_cache_dt:
                cls._minute_bar_cache_dt = now
                cls._minute_bar_cache = {}

            if days not in cls._minute_bar_cache:
                # Cache this calculation to happen once per bar, even if we
                # use another transform with the same number of days.
                prev = env.previous_trading_day(now)
                ds = env.days_in_range(
                    env.add_trading_days(-days + 2, prev),
                    prev,
                )
                # compute the number of minutes in the (days - 1) days before
                # today.
                # 210 minutes in a an early close and 390 in a full day.
                ms = sum(210 if d in env.early_closes else 390 for d in ds)
                # Add the number of minutes for today.
                ms += int(
                    (now - env.get_open_and_close(now)[0]).total_seconds() / 60
                )

                cls._minute_bar_cache[days] = ms + 1  # Account for this minute

            return cls._minute_bar_cache[days]

        if get_algo_instance().sim_params.data_frequency == 'daily':
            self._freqstr = '1d'
            # update this method to point to the daily variant.
            self._get_bars = daily_get_bars
            self._get_max_bars = daily_get_max_bars
        else:
            self._freqstr = '1m'
            # update this method to point to the minute variant.
            self._get_bars = minute_get_bars
            self._get_max_bars = minute_get_max_bars

        # Not actually recursive because we have already cached the new method.
        return self._get_bars(days)

    def mavg(self, days):
        bars = self._get_bars(days)
        max_bars = self._get_max_bars(days)
        prices = self._get_buffer(max_bars, raw=True)[-bars:]
        return nanmean(prices)

    def stddev(self, days):
        bars = self._get_bars(days)
        max_bars = self._get_max_bars(days)
        prices = self._get_buffer(max_bars, raw=True)[-bars:]
        return nanstd(prices, ddof=1)

    def vwap(self, days):
        bars = self._get_bars(days)
        max_bars = self._get_max_bars(days)
        prices = self._get_buffer(max_bars, raw=True)[-bars:]
        vols = self._get_buffer(max_bars, field='volume', raw=True)[-bars:]

        vol_sum = nansum(vols)
        try:
            ret = nansum(prices * vols) / vol_sum
        except ZeroDivisionError:
            ret = np.nan

        return ret

    def returns(self):
        algo = get_algo_instance()

        now = algo.datetime
        if now != self._returns_cache_dt:
            self._returns_cache_dt = now
            self._returns_cache = algo.history(2, '1d', 'price', ffill=True)

        hst = self._returns_cache[self._sid]
        return (hst.iloc[-1] - hst.iloc[0]) / hst.iloc[0]


class BarData(object):
    """
    Holds the event data for all sids for a given dt.

    This is what is passed as `data` to the `handle_data` function.

    Note: Many methods are analogues of dictionary because of historical
    usage of what this replaced as a dictionary subclass.
    """

    def __init__(self, data=None):
        self._data = data or {}
        self._contains_override = None

    def __contains__(self, name):
        if self._contains_override:
            if self._contains_override(name):
                return name in self._data
            else:
                return False
        else:
            return name in self._data

    def has_key(self, name):
        """
        DEPRECATED: __contains__ is preferred, but this method is for
        compatibility with existing algorithms.
        """
        return name in self

    def __setitem__(self, name, value):
        self._data[name] = value

    def __getitem__(self, name):
        return self._data[name]

    def __delitem__(self, name):
        del self._data[name]

    def __iter__(self):
        for sid, data in iteritems(self._data):
            # Allow contains override to filter out sids.
            if sid in self:
                if len(data):
                    yield sid

    def iterkeys(self):
        # Allow contains override to filter out sids.
        return (sid for sid in iterkeys(self._data) if sid in self)

    def keys(self):
        # Allow contains override to filter out sids.
        return list(self.iterkeys())

    def itervalues(self):
        return (value for _sid, value in self.iteritems())

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        return ((sid, value) for sid, value
                in iteritems(self._data)
                if sid in self)

    def items(self):
        return list(self.iteritems())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self._data)
