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

from abc import (
    ABCMeta,
    abstractmethod,
    abstractproperty,
)

from cachetools import LRUCache
from numpy import around, hstack
from pandas.tslib import normalize_date

from six import with_metaclass

from zipline.lib._float64window import AdjustedArrayWindow as Float64Window
from zipline.lib.adjustment import Float64Multiply
from zipline.utils.cache import ExpiringCache
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import float64_dtype


class SlidingWindow(object):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """

    def __init__(self, window, size, cal_start, offset):
        self.window = window
        self.cal_start = cal_start
        self.current = around(next(window), 3)
        self.offset = offset
        self.most_recent_ix = self.cal_start + size

    def get(self, end_ix):
        """
        Returns
        -------
        out : A np.ndarray of the equity pricing up to end_ix after adjustments
              and rounding have been applied.
        """
        if self.most_recent_ix == end_ix:
            return self.current

        target = end_ix - self.cal_start - self.offset + 1
        self.current = around(self.window.seek(target), 3)

        self.most_recent_ix = end_ix
        return self.current


class USEquityHistoryLoader(with_metaclass(ABCMeta)):
    """
    Loader for sliding history windows of adjusted US Equity Pricing data.

    Parameters
    ----------
    reader : DailyBarReader, MinuteBarReader
        Reader for pricing bars.
    adjustment_reader : SQLiteAdjustmentReader
        Reader for adjustment data.
    """
    FIELDS = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self, env, reader, adjustment_reader, sid_cache_size=1000):
        self.env = env
        self._reader = reader
        self._adjustments_reader = adjustment_reader
        self._window_blocks = {
            field: ExpiringCache(LRUCache(maxsize=sid_cache_size))
            for field in self.FIELDS
        }

    @abstractproperty
    def _prefetch_length(self):
        pass

    @abstractproperty
    def _calendar(self):
        pass

    @abstractmethod
    def _array(self, start, end, assets, field):
        pass

    def _get_adjustments_in_range(self, asset, dts, field):
        """
        Get the Float64Multiply objects to pass to an AdjustedArrayWindow.

        For the use of AdjustedArrayWindow in the loader, which looks back
        from current simulation time back to a window of data the dictionary is
        structured with:
        - the key into the dictionary for adjustments is the location of the
        day from which the window is being viewed.
        - the start of all multiply objects is always 0 (in each window all
          adjustments are overlapping)
        - the end of the multiply object is the location before the calendar
          location of the adjustment action, making all days before the event
          adjusted.

        Parameters
        ----------
        asset : Asset
            The assets for which to get adjustments.
        days : iterable of datetime64-like
            The days for which adjustment data is needed.
        field : str
            OHLCV field for which to get the adjustments.

        Returns
        -------
        out : The adjustments as a dict of loc -> Float64Multiply
        """
        sid = int(asset)
        start = normalize_date(dts[0])
        end = normalize_date(dts[-1])
        adjs = {}
        if field != 'volume':
            mergers = self._adjustments_reader.get_adjustments_for_sid(
                'mergers', sid)
            for m in mergers:
                dt = m[0]
                if start < dt <= end:
                    end_loc = dts.searchsorted(dt)
                    mult = Float64Multiply(0,
                                           end_loc - 1,
                                           0,
                                           0,
                                           m[1])
                    try:
                        adjs[end_loc].append(mult)
                    except KeyError:
                        adjs[end_loc] = [mult]
            divs = self._adjustments_reader.get_adjustments_for_sid(
                'dividends', sid)
            for d in divs:
                dt = d[0]
                if start < dt <= end:
                    end_loc = dts.searchsorted(dt)
                    mult = Float64Multiply(0,
                                           end_loc - 1,
                                           0,
                                           0,
                                           d[1])
                    try:
                        adjs[end_loc].append(mult)
                    except KeyError:
                        adjs[end_loc] = [mult]
        splits = self._adjustments_reader.get_adjustments_for_sid(
            'splits', sid)
        for s in splits:
            dt = s[0]
            if field == 'volume':
                ratio = 1.0 / s[1]
            else:
                ratio = s[1]
            if start < dt <= end:
                end_loc = dts.searchsorted(dt)
                mult = Float64Multiply(0,
                                       end_loc - 1,
                                       0,
                                       0,
                                       ratio)
                try:
                    adjs[end_loc].append(mult)
                except KeyError:
                    adjs[end_loc] = [mult]
        return adjs

    def _ensure_sliding_windows(self, assets, dts, field):
        """
        Ensure that there is a Float64Multiply window for each asset that can
        provide data for the given parameters.
        If the corresponding window for the (assets, len(dts), field) does not
        exist, then create a new one.
        If a corresponding window does exist for (assets, len(dts), field), but
        can not provide data for the current dts range, then create a new
        one and replace the expired window.

        Parameters
        ----------
        assets : iterable of Assets
            The assets in the window
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str
            The OHLCV field for which to retrieve data.

        Returns
        -------
        out : list of Float64Window with sufficient data so that each asset's
        window can provide `get` for the index corresponding with the last
        value in `dts`
        """
        end = dts[-1]
        size = len(dts)
        asset_windows = {}
        needed_assets = []
        for asset in assets:
            try:
                asset_windows[asset] = self._window_blocks[field].get(
                    (asset, size), end)
            except KeyError:
                needed_assets.append(asset)

        if needed_assets:
            start = dts[0]

            offset = 0
            start_ix = self._calendar.get_loc(start)
            end_ix = self._calendar.get_loc(end)

            cal = self._calendar
            prefetch_end_ix = min(end_ix + self._prefetch_length, len(cal) - 1)
            prefetch_end = cal[prefetch_end_ix]
            prefetch_dts = cal[start_ix:prefetch_end_ix + 1]
            prefetch_len = len(prefetch_dts)
            array = self._array(prefetch_dts, needed_assets, field)
            view_kwargs = {}
            if field == 'volume':
                array = array.astype(float64_dtype)

            for i, asset in enumerate(needed_assets):
                if self._adjustments_reader:
                    adjs = self._get_adjustments_in_range(
                        asset, prefetch_dts, field)
                else:
                    adjs = {}
                window = Float64Window(
                    array[:, i].reshape(prefetch_len, 1),
                    view_kwargs,
                    adjs,
                    offset,
                    size
                )
                sliding_window = SlidingWindow(window, size, start_ix, offset)
                asset_windows[asset] = sliding_window
                self._window_blocks[field].set((asset, size),
                                               sliding_window,
                                               prefetch_end)

        return [asset_windows[asset] for asset in assets]

    def history(self, assets, dts, field):
        """
        A window of pricing data with adjustments applied assuming that the
        end of the window is the day before the current simulation time.

        Parameters
        ----------
        assets : iterable of Assets
            The assets in the window.
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str
            The OHLCV field for which to retrieve data.


        Returns
        -------
        out : np.ndarray with shape(len(days between start, end), len(assets))
        """
        block = self._ensure_sliding_windows(assets, dts, field)
        end_ix = self._calendar.get_loc(dts[-1])
        return hstack([window.get(end_ix) for window in block])


class USEquityDailyHistoryLoader(USEquityHistoryLoader):

    @property
    def _prefetch_length(self):
        return 40

    @property
    def _calendar(self):
        return self._reader._calendar

    def _array(self, dts, assets, field):
        return self._reader.load_raw_arrays(
            [field],
            dts[0],
            dts[-1],
            assets,
        )[0]


class USEquityMinuteHistoryLoader(USEquityHistoryLoader):

    @property
    def _prefetch_length(self):
        return 1560

    @lazyval
    def _calendar(self):
        mm = self.env.market_minutes
        return mm[mm.slice_indexer(start=self._reader.first_trading_day,
                                   end=self._reader.last_available_dt)]

    def _array(self, dts, assets, field):
        return self._reader.load_raw_arrays(
            [field],
            dts[0],
            dts[-1],
            assets,
        )[0]
