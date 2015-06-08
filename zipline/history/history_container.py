#
# Copyright 2014 Quantopian, Inc.
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
from bisect import insort_left
from collections import namedtuple
from itertools import groupby, product

import logbook
import numpy as np
import pandas as pd
from six import itervalues, iteritems, iterkeys

from . history import HistorySpec

from zipline.finance.trading import with_environment
from zipline.utils.data import RollingPanel, _ensure_index
from zipline.utils.munge import ffill, bfill

logger = logbook.Logger('History Container')


# The closing price is referred to by multiple names,
# allow both for price rollover logic etc.
CLOSING_PRICE_FIELDS = frozenset({'price', 'close_price'})


def ffill_buffer_from_prior_values(freq,
                                   field,
                                   buffer_frame,
                                   digest_frame,
                                   pv_frame,
                                   raw=False):
    """
    Forward-fill a buffer frame, falling back to the end-of-period values of a
    digest frame if the buffer frame has leading NaNs.
    """
    # convert to ndarray if necessary
    digest_values = digest_frame
    if raw and isinstance(digest_frame, pd.DataFrame):
        digest_values = digest_frame.values

    buffer_values = buffer_frame
    if raw and isinstance(buffer_frame, pd.DataFrame):
        buffer_values = buffer_frame.values

    nan_sids = pd.isnull(buffer_values[0])
    if np.any(nan_sids) and len(digest_values):
        # If we have any leading nans in the buffer and we have a non-empty
        # digest frame, use the oldest digest values as the initial buffer
        # values.
        buffer_values[0, nan_sids] = digest_values[-1, nan_sids]

    nan_sids = pd.isnull(buffer_values[0])
    if np.any(nan_sids):
        # If we still have leading nans, fall back to the last known values
        # from before the digest.
        key_loc = pv_frame.index.get_loc((freq.freq_str, field))
        filler = pv_frame.values[key_loc, nan_sids]
        buffer_values[0, nan_sids] = filler

    if raw:
        filled = ffill(buffer_values)
        return filled

    return buffer_frame.ffill()


def ffill_digest_frame_from_prior_values(freq,
                                         field,
                                         digest_frame,
                                         pv_frame,
                                         raw=False):
    """
    Forward-fill a digest frame, falling back to the last known prior values if
    necessary.
    """
    # convert to ndarray if necessary
    values = digest_frame
    if raw and isinstance(digest_frame, pd.DataFrame):
        values = digest_frame.values

    nan_sids = pd.isnull(values[0])
    if np.any(nan_sids):
        # If we have any leading nans in the frame, use values from pv_frame to
        # seed values for those sids.
        key_loc = pv_frame.index.get_loc((freq.freq_str, field))
        filler = pv_frame.values[key_loc, nan_sids]
        values[0, nan_sids] = filler

    if raw:
        filled = ffill(values)
        return filled

    return digest_frame.ffill()


def freq_str_and_bar_count(history_spec):
    """
    Helper for getting the frequency string and bar count from a history spec.
    """
    return (history_spec.frequency.freq_str, history_spec.bar_count)


@with_environment()
def next_bar(spec, env):
    """
    Returns a function that will return the next bar for a given datetime.
    """
    if spec.frequency.unit_str == 'd':
        if spec.frequency.data_frequency == 'minute':
            return lambda dt: env.get_open_and_close(
                env.next_trading_day(dt),
            )[1]
        else:
            return env.next_trading_day
    else:
        return env.next_market_minute


def compute_largest_specs(history_specs):
    """
    Maps a Frequency to the largest HistorySpec at that frequency from an
    iterable of HistorySpecs.
    """
    return {key: max(group, key=lambda f: f.bar_count)
            for key, group in groupby(
                sorted(history_specs, key=freq_str_and_bar_count),
                key=lambda spec: spec.frequency)}


# tuples to store a change to the shape of a HistoryContainer

FrequencyDelta = namedtuple(
    'FrequencyDelta',
    ['freq', 'buffer_delta'],
)


LengthDelta = namedtuple(
    'LengthDelta',
    ['freq', 'delta'],
)


HistoryContainerDeltaSuper = namedtuple(
    'HistoryContainerDelta',
    ['field', 'frequency_delta', 'length_delta'],
)


class HistoryContainerDelta(HistoryContainerDeltaSuper):
    """
    A class representing a resize of the history container.
    """
    def __new__(cls, field=None, frequency_delta=None, length_delta=None):
        """
        field is a new field that was added.
        frequency is a FrequencyDelta representing a new frequency was added.
        length is a bar LengthDelta which is a frequency and a bar_count.
        If any field is None, then no change occurred of that type.
        """
        return super(HistoryContainerDelta, cls).__new__(
            cls, field, frequency_delta, length_delta,
        )

    @property
    def empty(self):
        """
        Checks if the delta is empty.
        """
        return (self.field is None and
                self.frequency_delta is None and
                self.length_delta is None)


def normalize_to_data_freq(data_frequency, dt):
    if data_frequency == 'minute':
        return dt
    return pd.tslib.normalize_date(dt)


class HistoryContainer(object):
    """
    Container for all history panels and frames used by an algoscript.

    To be used internally by TradingAlgorithm, but *not* passed directly to the
    algorithm.

    Entry point for the algoscript is the result of `get_history`.
    """
    VALID_FIELDS = {
        'price', 'open_price', 'volume', 'high', 'low', 'close_price',
    }

    def __init__(self,
                 history_specs,
                 initial_sids,
                 initial_dt,
                 data_frequency,
                 bar_data=None):
        """
        A container to hold a rolling window of historical data within a user's
        algorithm.

        Args:
          history_specs (dict[Frequency:HistorySpec]): The starting history
            specs that this container should be able to service.

          initial_sids (set[Asset or Int]): The starting sids to watch.

          initial_dt (datetime): The datetime to start collecting history from.

          bar_data (BarData): If this container is being constructed during
            handle_data, this is the BarData for the current bar to fill the
            buffer with. If this is constructed elsewhere, it is None.

        Returns:
          An instance of a new HistoryContainer
        """

        # History specs to be served by this container.
        self.history_specs = history_specs
        self.largest_specs = compute_largest_specs(
            itervalues(self.history_specs)
        )

        # The set of fields specified by all history specs
        self.fields = pd.Index(
            sorted(set(spec.field for spec in itervalues(history_specs)))
        )
        self.sids = pd.Index(
            sorted(set(initial_sids or []))
        )

        self.data_frequency = data_frequency

        initial_dt = normalize_to_data_freq(self.data_frequency, initial_dt)

        # This panel contains raw minutes for periods that haven't been fully
        # completed.  When a frequency period rolls over, these minutes are
        # digested using some sort of aggregation call on the panel (e.g. `sum`
        # for volume, `max` for high, `min` for low, etc.).
        self.buffer_panel = self.create_buffer_panel(initial_dt, bar_data)

        # Dictionaries with Frequency objects as keys.
        self.digest_panels, self.cur_window_starts, self.cur_window_closes = \
            self.create_digest_panels(initial_sids, initial_dt)

        # Helps prop up the prior day panel against having a nan, when the data
        # has been seen.
        self.last_known_prior_values = pd.DataFrame(
            data=None,
            index=self.prior_values_index,
            columns=self.prior_values_columns,
            # Note: For bizarre "intricacies of the spaghetti that is pandas
            # indexing logic" reasons, setting this dtype prevents indexing
            # errors in update_last_known_values.  This is safe for the time
            # being because our only forward-fillable fields are floats.  If we
            # need to add a non-float-typed forward-fillable field, then we may
            # find ourselves having to track down and fix a pandas bug.
            dtype=np.float64,
        )

    _ffillable_fields = None

    @property
    def ffillable_fields(self):
        if self._ffillable_fields is None:
            fillables = self.fields.intersection(HistorySpec.FORWARD_FILLABLE)
            self._ffillable_fields = fillables
        return self._ffillable_fields

    @property
    def prior_values_index(self):
        index_values = list(
            product(
                (freq.freq_str for freq in self.unique_frequencies),
                # Only store prior values for forward-fillable fields.
                self.ffillable_fields,
            )
        )
        if index_values:
            return pd.MultiIndex.from_tuples(index_values)
        else:
            # MultiIndex doesn't gracefully support empty input, so we return
            # an empty regular Index if we have values.
            return pd.Index(index_values)

    @property
    def prior_values_columns(self):
        return self.sids

    @property
    def all_panels(self):
        yield self.buffer_panel
        for panel in self.digest_panels.values():
            yield panel

    @property
    def unique_frequencies(self):
        """
        Return an iterator over all the unique frequencies serviced by this
        container.
        """
        return iterkeys(self.largest_specs)

    @with_environment()
    def _add_frequency(self, spec, dt, data, env=None):
        """
        Adds a new frequency to the container. This reshapes the buffer_panel
        if needed.
        """
        freq = spec.frequency
        self.largest_specs[freq] = spec
        new_buffer_len = 0

        if freq.max_bars > self.buffer_panel.window_length:
            # More bars need to be held in the buffer_panel to support this
            # freq
            if freq.data_frequency \
               != self.buffer_spec.frequency.data_frequency:
                # If the data_frequencies are not the same, then we need to
                # create a fresh buffer.
                self.buffer_panel = self.create_buffer_panel(
                    dt, bar_data=data,
                )
                new_buffer_len = None
            else:
                # The frequencies are the same, we just need to add more bars.
                self._resize_panel(
                    self.buffer_panel,
                    freq.max_bars,
                    dt,
                    self.buffer_spec.frequency,
                )
                new_buffer_len = freq.max_minutes
                # update the current buffer_spec to reflect the new lenght.
                self.buffer_spec.bar_count = new_buffer_len + 1

        if spec.bar_count > 1:
            # This spec has more than one bar, construct a digest panel for it.
            self.digest_panels[freq] = self._create_digest_panel(
                dt, spec=spec, env=env,
            )
        else:
            self.cur_window_starts[freq] = dt
            self.cur_window_closes[freq] = freq.window_close(
                self.cur_window_starts[freq]
            )

        self.last_known_prior_values = self.last_known_prior_values.reindex(
            index=self.prior_values_index,
        )

        return FrequencyDelta(freq, new_buffer_len)

    def _add_field(self, field):
        """
        Adds a new field to the container.
        """
        # self.fields is already sorted, so we just need to insert the new
        # field in the correct index.
        ls = list(self.fields)
        insort_left(ls, field)
        self.fields = pd.Index(ls)
        # unset fillable fields cache
        self._ffillable_fields = None

        self._realign_fields()
        self.last_known_prior_values = self.last_known_prior_values.reindex(
            index=self.prior_values_index,
        )
        return field

    @with_environment()
    def _add_length(self, spec, dt, env=None):
        """
        Increases the length of the digest panel for spec.frequency. If this
        does not have a panel, and one is needed; a digest panel will be
        constructed.
        """
        old_count = self.largest_specs[spec.frequency].bar_count
        self.largest_specs[spec.frequency] = spec
        delta = spec.bar_count - old_count

        panel = self.digest_panels.get(spec.frequency)

        if panel is None:
            # The old length for this frequency was 1 bar, meaning no digest
            # panel was held. We must construct a new one here.
            panel = self._create_digest_panel(
                dt, spec=spec, env=env,
            )

        else:
            self._resize_panel(
                panel, spec.bar_count - 1, dt, freq=spec.frequency, env=env,
            )

        self.digest_panels[spec.frequency] = panel

        return LengthDelta(spec.frequency, delta)

    @with_environment()
    def _resize_panel(self, panel, size, dt, freq, env=None):
        """
        Resizes a panel, fills the date_buf with the correct values.
        """
        # This is the oldest datetime that will be shown in the current window
        # of the panel.
        oldest_dt = pd.Timestamp(panel.start_date, tz='utc',)
        delta = size - panel.window_length

        # Construct the missing dates.
        missing_dts = self._create_window_date_buf(
            delta, freq.unit_str, freq.data_frequency, oldest_dt,
        )

        panel.extend_back(missing_dts)

    @with_environment()
    def _create_window_date_buf(self,
                                window,
                                unit_str,
                                data_frequency,
                                dt,
                                env=None):
        """
        Creates a window length date_buf looking backwards from dt.
        """
        if unit_str == 'd':
            # Get the properly key'd datetime64 out of the pandas Timestamp
            if data_frequency != 'daily':
                arr = env.open_close_window(
                    dt,
                    window,
                    offset=-window,
                ).market_close.astype('datetime64[ns]').values
            else:
                arr = env.open_close_window(
                    dt,
                    window,
                    offset=-window,
                ).index.values

            return arr
        else:
            return env.market_minute_window(
                env.previous_market_minute(dt),
                window,
                step=-1,
            )[::-1].values

    @with_environment()
    def _create_panel(self, dt, spec, env=None):
        """
        Constructs a rolling panel with a properly aligned date_buf.
        """
        dt = normalize_to_data_freq(spec.frequency.data_frequency, dt)

        window = spec.bar_count - 1

        date_buf = self._create_window_date_buf(
            window,
            spec.frequency.unit_str,
            spec.frequency.data_frequency,
            dt,
            env=env,
        )

        panel = RollingPanel(
            window=window,
            items=self.fields,
            sids=self.sids,
            initial_dates=date_buf,
        )

        return panel

    @with_environment()
    def _create_digest_panel(self,
                             dt,
                             spec,
                             window_starts=None,
                             window_closes=None,
                             env=None):
        """
        Creates a digest panel, setting the window_starts and window_closes.
        If window_starts or window_closes are None, then self.cur_window_starts
        or self.cur_window_closes will be used.
        """
        freq = spec.frequency

        window_starts = window_starts if window_starts is not None \
            else self.cur_window_starts
        window_closes = window_closes if window_closes is not None \
            else self.cur_window_closes

        window_starts[freq] = freq.normalize(dt)
        window_closes[freq] = freq.window_close(window_starts[freq])

        return self._create_panel(dt, spec, env=env)

    def ensure_spec(self, spec, dt, bar_data):
        """
        Ensure that this container has enough space to hold the data for the
        given spec. This returns a HistoryContainerDelta to represent the
        changes in shape that the container made to support the new
        HistorySpec.
        """
        updated = {}
        if spec.field not in self.fields:
            updated['field'] = self._add_field(spec.field)
        if spec.frequency not in self.largest_specs:
            updated['frequency_delta'] = self._add_frequency(
                spec, dt, bar_data,
            )
        if spec.bar_count > self.largest_specs[spec.frequency].bar_count:
            updated['length_delta'] = self._add_length(spec, dt)
        return HistoryContainerDelta(**updated)

    def add_sids(self, to_add):
        """
        Add new sids to the container.
        """
        self.sids = pd.Index(
            sorted(self.sids.union(_ensure_index(to_add))),
        )
        self._realign_sids()

    def drop_sids(self, to_drop):
        """
        Remove sids from the container.
        """
        self.sids = pd.Index(
            sorted(self.sids.difference(_ensure_index(to_drop))),
        )
        self._realign_sids()

    def _realign_sids(self):
        """
        Realign our constituent panels after adding or removing sids.
        """
        self.last_known_prior_values = self.last_known_prior_values.reindex(
            columns=self.sids,
        )
        for panel in self.all_panels:
            panel.set_minor_axis(self.sids)

    def _realign_fields(self):
        self.last_known_prior_values = self.last_known_prior_values.reindex(
            index=self.prior_values_index,
        )
        for panel in self.all_panels:
            panel.set_items(self.fields)

    @with_environment()
    def create_digest_panels(self,
                             initial_sids,
                             initial_dt,
                             env=None):
        """
        Initialize a RollingPanel for each unique panel frequency being stored
        by this container.  Each RollingPanel pre-allocates enough storage
        space to service the highest bar-count of any history call that it
        serves.
        """
        # Map from frequency -> first/last minute of the next digest to be
        # rolled for that frequency.
        first_window_starts = {}
        first_window_closes = {}

        # Map from frequency -> digest_panels.
        panels = {}
        for freq, largest_spec in iteritems(self.largest_specs):
            if largest_spec.bar_count == 1:
                # No need to allocate a digest panel; this frequency will only
                # ever use data drawn from self.buffer_panel.
                first_window_starts[freq] = freq.normalize(initial_dt)
                first_window_closes[freq] = freq.window_close(
                    first_window_starts[freq]
                )

                continue

            dt = initial_dt

            rp = self._create_digest_panel(
                dt,
                spec=largest_spec,
                window_starts=first_window_starts,
                window_closes=first_window_closes,
                env=env,
            )

            panels[freq] = rp

        return panels, first_window_starts, first_window_closes

    def create_buffer_panel(self, initial_dt, bar_data):
        """
        Initialize a RollingPanel containing enough minutes to service all our
        frequencies.
        """
        max_bars_needed = max(
            freq.max_bars for freq in self.unique_frequencies
        )
        freq = '1m' if self.data_frequency == 'minute' else '1d'
        spec = HistorySpec(
            max_bars_needed + 1, freq, None, None, self.data_frequency,
        )

        rp = self._create_panel(
            initial_dt, spec,
        )
        self.buffer_spec = spec

        if bar_data is not None:
            frame = self.frame_from_bardata(bar_data, initial_dt)
            rp.add_frame(initial_dt, frame)

        return rp

    def convert_columns(self, values):
        """
        If columns have a specific type you want to enforce, overwrite this
        method and return the transformed values.
        """
        return values

    def digest_bars(self, history_spec, do_ffill):
        """
        Get the last (history_spec.bar_count - 1) bars from self.digest_panel
        for the requested HistorySpec.
        """
        bar_count = history_spec.bar_count
        if bar_count == 1:
            # slicing with [1 - bar_count:] doesn't work when bar_count == 1,
            # so special-casing this.
            res = pd.DataFrame(index=[], columns=self.sids, dtype=float)
            return res.values, res.index

        field = history_spec.field

        # Panel axes are (field, dates, sids).  We want just the entries for
        # the requested field, the last (bar_count - 1) data points, and all
        # sids.
        digest_panel = self.digest_panels[history_spec.frequency]
        frame = digest_panel.get_current(field, raw=True)
        if do_ffill:
            # Do forward-filling *before* truncating down to the requested
            # number of bars.  This protects us from losing data if an illiquid
            # stock has a gap in its price history.
            filled = ffill_digest_frame_from_prior_values(
                history_spec.frequency,
                history_spec.field,
                frame,
                self.last_known_prior_values,
                raw=True
                # Truncate only after we've forward-filled
            )
            indexer = slice(1 - bar_count, None)
            return filled[indexer], digest_panel.current_dates()[indexer]
        else:
            indexer = slice(1 - bar_count, None)
            return frame[indexer, :], digest_panel.current_dates()[indexer]

    def buffer_panel_minutes(self,
                             buffer_panel,
                             earliest_minute=None,
                             latest_minute=None,
                             raw=False):
        """
        Get the minutes in @buffer_panel between @earliest_minute and
        @latest_minute, inclusive.

        @buffer_panel can be a RollingPanel or a plain Panel.  If a
        RollingPanel is supplied, we call `get_current` to extract a Panel
        object.

        If no value is specified for @earliest_minute, use all the minutes we
        have up until @latest minute.

        If no value for @latest_minute is specified, use all values up until
        the latest minute.
        """
        if isinstance(buffer_panel, RollingPanel):
            buffer_panel = buffer_panel.get_current(start=earliest_minute,
                                                    end=latest_minute,
                                                    raw=raw)
            return buffer_panel
        # Using .ix here rather than .loc because loc requires that the keys
        # are actually in the index, whereas .ix returns all the values between
        # earliest_minute and latest_minute, which is what we want.
        return buffer_panel.ix[:, earliest_minute:latest_minute, :]

    def frame_from_bardata(self, data, algo_dt):
        """
        Create a DataFrame from the given BarData and algo dt.
        """
        data = data._data
        frame_data = np.empty((len(self.fields), len(self.sids))) * np.nan

        for j, sid in enumerate(self.sids):
            sid_data = data.get(sid)
            if not sid_data:
                continue
            if algo_dt != sid_data['dt']:
                continue
            for i, field in enumerate(self.fields):
                frame_data[i, j] = sid_data.get(field, np.nan)

        return pd.DataFrame(
            frame_data,
            index=self.fields.copy(),
            columns=self.sids.copy(),
        )

    def update(self, data, algo_dt):
        """
        Takes the bar at @algo_dt's @data, checks to see if we need to roll any
        new digests, then adds new data to the buffer panel.
        """
        frame = self.frame_from_bardata(data, algo_dt)

        self.update_last_known_values()
        self.update_digest_panels(algo_dt, self.buffer_panel)
        self.buffer_panel.add_frame(algo_dt, frame)

    def update_digest_panels(self, algo_dt, buffer_panel, freq_filter=None):
        """
        Check whether @algo_dt is greater than cur_window_close for any of our
        frequencies.  If so, roll a digest for that frequency using data drawn
        from @buffer panel and insert it into the appropriate digest panels.

        If @freq_filter is specified, only use the given data to update
        frequencies on which the filter returns True.

        This takes `buffer_panel` as an argument rather than using
        self.buffer_panel so that this method can be used to add supplemental
        data from an external source.
        """
        for frequency in filter(freq_filter, self.unique_frequencies):

            # We don't keep a digest panel if we only have a length-1 history
            # spec for a given frequency
            digest_panel = self.digest_panels.get(frequency, None)

            while algo_dt > self.cur_window_closes[frequency]:

                earliest_minute = self.cur_window_starts[frequency]
                latest_minute = self.cur_window_closes[frequency]
                minutes_to_process = self.buffer_panel_minutes(
                    buffer_panel,
                    earliest_minute=earliest_minute,
                    latest_minute=latest_minute,
                    raw=True
                )

                if digest_panel is not None:
                    # Create a digest from minutes_to_process and add it to
                    # digest_panel.
                    digest_frame = self.create_new_digest_frame(
                        minutes_to_process,
                        self.fields,
                        self.sids
                    )
                    digest_panel.add_frame(
                        latest_minute,
                        digest_frame,
                        self.fields,
                        self.sids
                    )

                # Update panel start/close for this frequency.
                self.cur_window_starts[frequency] = \
                    frequency.next_window_start(latest_minute)
                self.cur_window_closes[frequency] = \
                    frequency.window_close(self.cur_window_starts[frequency])

    def frame_to_series(self, field, frame, columns=None):
        """
        Convert a frame with a DatetimeIndex and sid columns into a series with
        a sid index, using the aggregator defined by the given field.
        """
        if isinstance(frame, pd.DataFrame):
            columns = frame.columns
            frame = frame.values

        if not len(frame):
            return pd.Series(
                data=(0 if field == 'volume' else np.nan),
                index=columns,
            ).values

        if field in ['price', 'close_price']:
            # shortcircuit for full last row
            vals = frame[-1]
            if np.all(~np.isnan(vals)):
                return vals
            return ffill(frame)[-1]
        elif field == 'open_price':
            return bfill(frame)[0]
        elif field == 'volume':
            return np.nansum(frame, axis=0)
        elif field == 'high':
            return np.nanmax(frame, axis=0)
        elif field == 'low':
            return np.nanmin(frame, axis=0)
        else:
            raise ValueError("Unknown field {}".format(field))

    def aggregate_ohlcv_panel(self,
                              fields,
                              ohlcv_panel,
                              items=None,
                              minor_axis=None):
        """
        Convert an OHLCV Panel into a DataFrame by aggregating each field's
        frame into a Series.
        """
        vals = ohlcv_panel
        if isinstance(ohlcv_panel, pd.Panel):
            vals = ohlcv_panel.values
            items = ohlcv_panel.items
            minor_axis = ohlcv_panel.minor_axis

        data = [
            self.frame_to_series(
                field,
                vals[items.get_loc(field)],
                minor_axis
            )
            for field in fields
        ]
        return np.array(data)

    def create_new_digest_frame(self, buffer_minutes, items=None,
                                minor_axis=None):
        """
        Package up minutes in @buffer_minutes into a single digest frame.
        """
        return self.aggregate_ohlcv_panel(
            self.fields,
            buffer_minutes,
            items=items,
            minor_axis=minor_axis
        )

    def update_last_known_values(self):
        """
        Store the non-NaN values from our oldest frame in each frequency.
        """
        ffillable = self.ffillable_fields
        if not len(ffillable):
            return

        for frequency in self.unique_frequencies:
            digest_panel = self.digest_panels.get(frequency, None)
            if digest_panel:
                oldest_known_values = digest_panel.oldest_frame(raw=True)
            else:
                oldest_known_values = self.buffer_panel.oldest_frame(raw=True)

            oldest_vals = oldest_known_values
            oldest_columns = self.fields
            for field in ffillable:
                f_idx = oldest_columns.get_loc(field)
                field_vals = oldest_vals[f_idx]
                # isnan would be fast, possible to use?
                non_nan_sids = np.where(pd.notnull(field_vals))
                key = (frequency.freq_str, field)
                key_loc = self.last_known_prior_values.index.get_loc(key)
                self.last_known_prior_values.values[
                    key_loc, non_nan_sids
                ] = field_vals[non_nan_sids]

    def get_history(self, history_spec, algo_dt):
        """
        Main API used by the algoscript is mapped to this function.

        Selects from the overarching history panel the values for the
        @history_spec at the given @algo_dt.
        """
        field = history_spec.field
        do_ffill = history_spec.ffill

        # Get our stored values from periods prior to the current period.
        digest_frame, index = self.digest_bars(history_spec, do_ffill)

        # Get minutes from our buffer panel to build the last row of the
        # returned frame.
        buffer_panel = self.buffer_panel_minutes(
            self.buffer_panel,
            earliest_minute=self.cur_window_starts[history_spec.frequency],
            raw=True
        )
        buffer_frame = buffer_panel[self.fields.get_loc(field)]

        if do_ffill:
            buffer_frame = ffill_buffer_from_prior_values(
                history_spec.frequency,
                field,
                buffer_frame,
                digest_frame,
                self.last_known_prior_values,
                raw=True
            )
        last_period = self.frame_to_series(field, buffer_frame, self.sids)
        return fast_build_history_output(digest_frame,
                                         last_period,
                                         algo_dt,
                                         index=index,
                                         columns=self.sids)


def fast_build_history_output(buffer_frame,
                              last_period,
                              algo_dt,
                              index=None,
                              columns=None):
    """
    Optimized concatenation of DataFrame and Series for use in
    HistoryContainer.get_history.

    Relies on the fact that the input arrays have compatible shapes.
    """
    buffer_values = buffer_frame
    if isinstance(buffer_frame, pd.DataFrame):
        buffer_values = buffer_frame.values
        index = buffer_frame.index
        columns = buffer_frame.columns

    return pd.DataFrame(
        data=np.vstack(
            [
                buffer_values,
                last_period,
            ]
        ),
        index=fast_append_date_to_index(
            index,
            pd.Timestamp(algo_dt)
        ),
        columns=columns,
    )


def fast_append_date_to_index(index, timestamp):
    """
    Append a timestamp to a DatetimeIndex.  DatetimeIndex.append does not
    appear to work.
    """
    return pd.DatetimeIndex(
        np.hstack(
            [
                index.values,
                [timestamp.asm8],
            ]
        ),
        tz='UTC',
    )
