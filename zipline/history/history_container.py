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
from itertools import groupby, product
import logbook
import numpy as np
import pandas as pd

from six import itervalues, iteritems, iterkeys

from . history import (
    index_at_dt,
    HistorySpec,
)

from zipline.utils.data import RollingPanel, _ensure_index

logger = logbook.Logger('History Container')


# The closing price is referred to by multiple names,
# allow both for price rollover logic etc.
CLOSING_PRICE_FIELDS = frozenset({'price', 'close_price'})


def ffill_buffer_from_prior_values(freq,
                                   field,
                                   buffer_frame,
                                   digest_frame,
                                   pv_frame):
    """
    Forward-fill a buffer frame, falling back to the end-of-period values of a
    digest frame if the buffer frame has leading NaNs.
    """

    nan_sids = buffer_frame.iloc[0].isnull()
    if any(nan_sids) and len(digest_frame):
        # If we have any leading nans in the buffer and we have a non-empty
        # digest frame, use the oldest digest values as the initial buffer
        # values.
        buffer_frame.ix[0, nan_sids] = digest_frame.ix[-1, nan_sids]

    nan_sids = buffer_frame.iloc[0].isnull()
    if any(nan_sids):
        # If we still have leading nans, fall back to the last known values
        # from before the digest.
        buffer_frame.ix[0, nan_sids] = pv_frame.loc[
            (freq.freq_str, field), nan_sids
        ]

    return buffer_frame.ffill()


def ffill_digest_frame_from_prior_values(freq,
                                         field,
                                         digest_frame,
                                         pv_frame):
    """
    Forward-fill a digest frame, falling back to the last known prior values if
    necessary.
    """
    nan_sids = digest_frame.iloc[0].isnull()
    if any(nan_sids):
        # If we have any leading nans in the frame, use values from pv_frame to
        # seed values for those sids.
        digest_frame.ix[0, nan_sids] = pv_frame.loc[
            (freq.freq_str, field), nan_sids
        ]

    return digest_frame.ffill()


def freq_str_and_bar_count(history_spec):
    """
    Helper for getting the frequency string and bar count from a history spec.
    """
    return (history_spec.frequency.freq_str, history_spec.bar_count)


def group_by_frequency(history_specs):
    """
    Takes an iterable of history specs and returns a dictionary mapping unique
    frequencies to a list of specs with that frequency.

    Within each list, the HistorySpecs are sorted by ascending bar count.

    Example:

    [HistorySpec(3, '1d', 'price', True),
     HistorySpec(2, '2d', 'open', True),
     HistorySpec(2, '1d', 'open', False),
     HistorySpec(5, '1m', 'open', True)]

    yields

    {Frequency('1d') : [HistorySpec(2, '1d', 'open', False)],
                        HistorySpec(3, '1d', 'price', True),
     Frequency('2d') : [HistorySpec(2, '2d', 'open', True)],
     Frequency('1m') : [HistorySpec(5, '1m', 'open', True)]}
    """
    return {key: list(group)
            for key, group in groupby(
                sorted(history_specs, key=freq_str_and_bar_count),
                key=lambda spec: spec.frequency)}


class HistoryContainer(object):
    """
    Container for all history panels and frames used by an algoscript.

    To be used internally by TradingAlgorithm, but *not* passed directly to the
    algorithm.

    Entry point for the algoscript is the result of `get_history`.
    """

    def __init__(self, history_specs, initial_sids, initial_dt):

        # History specs to be served by this container.
        self.history_specs = history_specs
        self.frequency_groups = \
            group_by_frequency(itervalues(self.history_specs))

        # The set of fields specified by all history specs
        self.fields = pd.Index(
            sorted(set(spec.field for spec in itervalues(history_specs)))
        )
        self.sids = pd.Index(
            sorted(set(initial_sids))
        )

        # This panel contains raw minutes for periods that haven't been fully
        # completed.  When a frequency period rolls over, these minutes are
        # digested using some sort of aggregation call on the panel (e.g. `sum`
        # for volume, `max` for high, `min` for low, etc.).
        self.buffer_panel = self.create_buffer_panel(
            initial_dt,
        )

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

    @property
    def ffillable_fields(self):
        return self.fields.intersection(HistorySpec.FORWARD_FILLABLE)

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
        return iterkeys(self.frequency_groups)

    def add_sids(self, to_add):
        """
        Add new sids to the container.
        """
        self.sids = pd.Index(
            sorted(self.sids + _ensure_index(to_add)),
        )
        self._realign()

    def drop_sids(self, to_drop):
        """
        Remove sids from the container.
        """
        self.sids = pd.Index(
            sorted(self.sids - _ensure_index(to_drop)),
        )
        self._realign()

    def _realign(self):
        """
        Realign our constituent panels after adding or removing sids.
        """
        self.last_known_prior_values = self.last_known_prior_values.reindex(
            columns=self.sids,
        )
        for panel in self.all_panels:
            panel.set_sids(self.sids)

    def create_digest_panels(self, initial_sids, initial_dt):
        """
        Initialize a RollingPanel for each unique panel frequency being stored
        by this container.  Each RollingPanel pre-allocates enough storage
        space to service the highest bar-count of any history call that it
        serves.

        Relies on the fact that group_by_frequency sorts the value lists by
        ascending bar count.
        """
        # Map from frequency -> first/last minute of the next digest to be
        # rolled for that frequency.
        first_window_starts = {}
        first_window_closes = {}

        # Map from frequency -> digest_panels.
        panels = {}
        for freq, specs in iteritems(self.frequency_groups):

            # Relying on the sorting of group_by_frequency to get the spec
            # requiring the largest number of bars.
            largest_spec = specs[-1]
            if largest_spec.bar_count == 1:

                # No need to allocate a digest panel; this frequency will only
                # ever use data drawn from self.buffer_panel.
                first_window_starts[freq] = freq.window_open(initial_dt)
                first_window_closes[freq] = freq.window_close(
                    first_window_starts[freq]
                )

                continue

            initial_dates = index_at_dt(largest_spec, initial_dt)

            # Set up dates for our first digest roll, which is keyed to the
            # close of the first entry in our initial index.
            first_window_closes[freq] = initial_dates[0]
            first_window_starts[freq] = freq.window_open(initial_dates[0])

            rp = RollingPanel(
                window=len(initial_dates) - 1,
                items=self.fields,
                sids=initial_sids,
            )

            panels[freq] = rp

        return panels, first_window_starts, first_window_closes

    def create_buffer_panel(self, initial_dt):
        """
        Initialize a RollingPanel containing enough minutes to service all our
        frequencies.
        """
        max_bars_needed = max(freq.max_minutes
                              for freq in self.unique_frequencies)
        rp = RollingPanel(
            window=max_bars_needed,
            items=self.fields,
            sids=self.sids,
        )
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
            return pd.DataFrame(index=[], columns=self.sids)

        field = history_spec.field

        # Panel axes are (field, dates, sids).  We want just the entries for
        # the requested field, the last (bar_count - 1) data points, and all
        # sids.
        panel = self.digest_panels[history_spec.frequency].get_current()
        if do_ffill:
            # Do forward-filling *before* truncating down to the requested
            # number of bars.  This protects us from losing data if an illiquid
            # stock has a gap in its price history.
            return ffill_digest_frame_from_prior_values(
                history_spec.frequency,
                history_spec.field,
                panel.loc[field],
                self.last_known_prior_values,
                # Truncate only after we've forward-filled
            ).iloc[1 - bar_count:]
        else:
            return panel.ix[field, 1 - bar_count:, :]

    def buffer_panel_minutes(self,
                             buffer_panel,
                             earliest_minute=None,
                             latest_minute=None):
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
            buffer_panel = buffer_panel.get_current()

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
                )

                if digest_panel is not None:
                    # Create a digest from minutes_to_process and add it to
                    # digest_panel.
                    digest_panel.add_frame(
                        latest_minute,
                        self.create_new_digest_frame(minutes_to_process)
                    )

                # Update panel start/close for this frequency.
                self.cur_window_starts[frequency] = \
                    frequency.next_window_start(latest_minute)
                self.cur_window_closes[frequency] = \
                    frequency.window_close(self.cur_window_starts[frequency])

    def frame_to_series(self, field, frame):
        """
        Convert a frame with a DatetimeIndex and sid columns into a series with
        a sid index, using the aggregator defined by the given field.
        """
        if not len(frame):
            return pd.Series(
                data=(0 if field == 'volume' else np.nan),
                index=frame.columns,
            )

        if field in ['price', 'close_price']:
            return frame.ffill().iloc[-1].values
        elif field == 'open_price':
            return frame.bfill().iloc[0].values
        elif field == 'volume':
            return frame.sum().values
        elif field == 'high':
            return frame.max().values
        elif field == 'low':
            return frame.min().values
        else:
            raise ValueError("Unknown field {}".format(field))

    def aggregate_ohlcv_panel(self, fields, ohlcv_panel):
        """
        Convert an OHLCV Panel into a DataFrame by aggregating each field's
        frame into a Series.
        """
        return pd.DataFrame(
            [
                self.frame_to_series(field, ohlcv_panel.loc[field])
                for field in fields
            ],
            index=fields,
            columns=ohlcv_panel.minor_axis,
        )

    def create_new_digest_frame(self, buffer_minutes):
        """
        Package up minutes in @buffer_minutes into a single digest frame.
        """
        return self.aggregate_ohlcv_panel(
            self.fields,
            buffer_minutes,
        )

    def update_last_known_values(self):
        """
        Store the non-NaN values from our oldest frame in each frequency.
        """
        ffillable = self.ffillable_fields
        if len(ffillable) == 0:
            return

        for frequency in self.unique_frequencies:
            digest_panel = self.digest_panels.get(frequency, None)
            if digest_panel:
                oldest_known_values = digest_panel.oldest_frame()
            else:
                oldest_known_values = self.buffer_panel.oldest_frame()

            for field in ffillable:
                non_nan_sids = oldest_known_values[field].notnull()
                self.last_known_prior_values.loc[
                    (frequency.freq_str, field), non_nan_sids
                ] = oldest_known_values[field].dropna()

    def get_history(self, history_spec, algo_dt):
        """
        Main API used by the algoscript is mapped to this function.

        Selects from the overarching history panel the values for the
        @history_spec at the given @algo_dt.
        """

        field = history_spec.field
        do_ffill = history_spec.ffill

        # Get our stored values from periods prior to the current period.
        digest_frame = self.digest_bars(history_spec, do_ffill)

        # Get minutes from our buffer panel to build the last row of the
        # returned frame.
        buffer_frame = self.buffer_panel_minutes(
            self.buffer_panel,
            earliest_minute=self.cur_window_starts[history_spec.frequency],
        )[field]

        if do_ffill:
            buffer_frame = ffill_buffer_from_prior_values(
                history_spec.frequency,
                field,
                buffer_frame,
                digest_frame,
                self.last_known_prior_values,
            )

        last_period = self.frame_to_series(field, buffer_frame)
        return fast_build_history_output(digest_frame, last_period, algo_dt)


def fast_build_history_output(buffer_frame, last_period, algo_dt):
    """
    Optimized concatenation of DataFrame and Series for use in
    HistoryContainer.get_history.

    Relies on the fact that the input arrays have compatible shapes.
    """
    return pd.DataFrame(
        data=np.vstack(
            [
                buffer_frame.values,
                last_period,
            ]
        ),
        index=fast_append_date_to_index(
            buffer_frame.index,
            pd.Timestamp(algo_dt)
        ),
        columns=buffer_frame.columns,
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
