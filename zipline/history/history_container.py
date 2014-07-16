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
from itertools import groupby

import numpy as np
import pandas as pd

from six import itervalues, iteritems, iterkeys

from . history import (
    index_at_dt,
)

from zipline.utils.data import RollingPanel


# The closing price is referred to by multiple names,
# allow both for price rollover logic etc.
CLOSING_PRICE_FIELDS = frozenset({'price', 'close_price'})


def ffill_buffer_from_prior_values(field,
                                   buffer_frame,
                                   digest_frame,
                                   pre_digest_values):
    """
    Forward-fill a buffer frame, falling back to the end-of-period values of a
    digest frame if the buffer frame has leading NaNs.
    """

    # Get values which are NaN at the beginning of the period.
    first_bar = buffer_frame.iloc[0]

    def iter_nan_sids():
        """
        Helper for iterating over the remaining nan sids in first_bar.
        """
        return (sid for sid in first_bar[first_bar.isnull()].index)

    # Try to fill with the last entry from the digest frame.
    if digest_frame is not None:
        # We don't store a digest frame for frequencies that only have a bar
        # count of 1.
        for sid in iter_nan_sids():
            buffer_frame[sid][0] = digest_frame.ix[-1, sid]

    # If we still have nan sids, try to fill with pre_digest_values.
    for sid in iter_nan_sids():
        prior_sid_value = pre_digest_values[field].get(sid)
        if prior_sid_value:
            # If the prior value is greater than the timestamp of our first
            # bar.
            if prior_sid_value.get('dt', first_bar.name) > first_bar.name:
                buffer_frame[sid][0] = prior_sid_value.get('value', np.nan)

    return buffer_frame.ffill()


def ffill_digest_frame_from_prior_values(field, digest_frame, prior_values):
    """
    Forward-fill a digest frame, falling back to the last known priof values if
    necessary.
    """
    if digest_frame is not None:
        # Digest frame is None in the case that we only have length 1 history
        # specs for a given frequency.

        # It's possible that the first bar in our digest frame is storing NaN
        # values. If so, check if we've tracked an older value and use that as
        # an ffill value for the first bar.
        first_bar = digest_frame.ix[0]
        nan_sids = first_bar[first_bar.isnull()].index
        for sid in nan_sids:
            try:
                # Only use prior value if it is before the index,
                # so that a backfill does not accidentally occur.
                if prior_values[field][sid]['dt'] <= digest_frame.index[0]:
                    digest_frame[sid][0] = prior_values[field][sid]['value']

            except KeyError:
                # Allow case where there is no previous value.
                # e.g. with leading nans.
                pass
        digest_frame = digest_frame.ffill()
    return digest_frame


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
        self.fields = set(spec.field for spec in itervalues(history_specs))

        # This panel contains raw minutes for periods that haven't been fully
        # completed.  When a frequency period rolls over, these minutes are
        # digested using some sort of aggregation call on the panel (e.g. `sum`
        # for volume, `max` for high, `min` for low, etc.).
        self.buffer_panel = self.create_buffer_panel(
            initial_sids,
            initial_dt,
        )

        # Dictionaries with Frequency objects as keys.
        self.digest_panels, self.cur_window_starts, self.cur_window_closes = \
            self.create_digest_panels(initial_sids, initial_dt)

        # Populating initial frames here, so that the cost of creating the
        # initial frames does not show up when profiling.  These frames are
        # cached since mid-stream creation of containing data frames on every
        # bar is expensive.
        self.create_return_frames(initial_dt)

        # Helps prop up the prior day panel against having a nan, when the data
        # has been seen.
        self.last_known_prior_values = {field: {} for field in self.fields}

    @property
    def unique_frequencies(self):
        """
        Return an iterator over all the unique frequencies serviced by this
        container.
        """
        return iterkeys(self.frequency_groups)

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

            rp = RollingPanel(len(initial_dates) - 1,
                              self.fields,
                              initial_sids)

            panels[freq] = rp

        return panels, first_window_starts, first_window_closes

    def create_buffer_panel(self, initial_sids, initial_dt):
        """
        Initialize a RollingPanel containing enough minutes to service all our
        frequencies.
        """
        max_bars_needed = max(freq.max_minutes
                              for freq in self.unique_frequencies)
        rp = RollingPanel(
            max_bars_needed,
            self.fields,
            initial_sids,
            # Restrict the initial data down to just the fields being used in
            # this container.
        )
        return rp

    def convert_columns(self, values):
        """
        If columns have a specific type you want to enforce, overwrite this
        method and return the transformed values.
        """
        return values

    def create_return_frames(self, algo_dt):
        """
        Populates the return frame cache.

        Called during init and at universe rollovers.
        """
        self.return_frames = {}
        for spec_key, history_spec in iteritems(self.history_specs):
            index = pd.to_datetime(index_at_dt(history_spec, algo_dt))
            frame = pd.DataFrame(
                index=index,
                columns=self.convert_columns(
                    self.buffer_panel.minor_axis.values),
                dtype=np.float64)
            self.return_frames[spec_key] = frame

    def buffer_panel_minutes(self,
                             buffer_panel=None,
                             earliest_minute=None,
                             latest_minute=None):
        """
        Get the minutes in @buffer_panel between @earliest_minute and
        @last_minute, inclusive.

        @buffer_panel can be a RollingPanel or a plain Panel.  If a
        RollingPanel is supplied, we call `get_current` to extract a Panel
        object.  If no panel is supplied, we use self.buffer_panel.

        If no value is specified for @earliest_minute, use all the minutes we
        have up until @latest minute.

        If no value for @latest_minute is specified, use all values up until
        the latest minute.
        """
        buffer_panel = buffer_panel or self.buffer_panel
        if isinstance(buffer_panel, RollingPanel):
            buffer_panel = buffer_panel.get_current()

        return buffer_panel.ix[:, earliest_minute:latest_minute, :]

    def update(self, data, algo_dt):
        """
        Takes the bar at @algo_dt's @data, checks to see if we need to roll any
        new digests, then adds new data to the buffer panel.
        """
        self.update_digest_panels(algo_dt, self.buffer_panel)

        fields = self.fields
        frame = pd.DataFrame(
            {sid: {field: bar[field] for field in fields}
             for sid, bar in data.iteritems()
             if (bar
                 and
                 bar['dt'] == algo_dt
                 and
                 # Only use data which is keyed in the data panel.
                 # Prevents crashes due to custom data.
                 sid in self.buffer_panel.minor_axis)})
        self.buffer_panel.add_frame(algo_dt, frame)

    def update_digest_panels(self, algo_dt, buffer_panel, freq_filter=None):
        """
        Check whether @algo_dt is greater than cur_window_close for any of our
        frequencies.  If so, roll a digest for that frequency using data drawn
        from @buffer panel and insert it into the appropriate digest panels.

        If @freq_filter is specified, only use the given data to update
        frequencies on which the filter returns True.
        """
        for frequency in self.unique_frequencies:

            if freq_filter is not None and not freq_filter(frequency):
                continue

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

                # Create a digest from minutes_to_process and add it to
                # digest_panel.
                self.roll(frequency,
                          digest_panel,
                          minutes_to_process,
                          latest_minute)

                # Update panel start/close for this frequency.
                self.cur_window_starts[frequency] = \
                    frequency.next_window_start(latest_minute)
                self.cur_window_closes[frequency] = \
                    frequency.window_close(self.cur_window_starts[frequency])

    def roll(self, frequency, digest_panel, buffer_minutes, digest_dt):
        """
        Package up minutes in @buffer_minutes insert that bar into
        @digest_panel at index @last_minute, and update
        self.cur_window_{starts|closes} for the given frequency.
        """
        if digest_panel is None:
            # This happens if the only spec we have at this frequency has a bar
            # count of 1.
            return

        rolled = pd.DataFrame(
            index=self.fields,
            columns=buffer_minutes.minor_axis)

        for field in self.fields:

            if field in CLOSING_PRICE_FIELDS:
                # Use the last close, or NaN if we have no minutes.
                try:
                    prices = buffer_minutes.loc[field].ffill().iloc[-1]
                except IndexError:
                    # Scalar assignment sets the value for all entries.
                    prices = np.nan
                rolled.ix[field] = prices

            elif field == 'open_price':
                # Use the first open, or NaN if we have no minutes.
                try:
                    opens = buffer_minutes.loc[field].bfill().iloc[0]
                except IndexError:
                    # Scalar assignment sets the value for all entries.
                    opens = np.nan
                rolled.ix['open_price'] = opens

            elif field == 'volume':
                # Volume is the sum of the volumes during the
                # course of the period.
                volumes = buffer_minutes.ix['volume'].sum().fillna(0)
                rolled.ix['volume'] = volumes

            elif field == 'high':
                # Use the highest high.
                highs = buffer_minutes.ix['high'].max()
                rolled.ix['high'] = highs

            elif field == 'low':
                # Use the lowest low.
                lows = buffer_minutes.ix['low'].min()
                rolled.ix['low'] = lows

            for sid, value in rolled.ix[field].iterkv():
                if not np.isnan(value):
                    try:
                        prior_values = \
                            self.last_known_prior_values[field][sid]
                    except KeyError:
                        prior_values = {}
                        self.last_known_prior_values[field][sid] = \
                            prior_values
                    prior_values['dt'] = digest_dt
                    prior_values['value'] = value

        digest_panel.add_frame(digest_dt, rolled)

    def get_history(self, history_spec, algo_dt):
        """
        Main API used by the algoscript is mapped to this function.

        Selects from the overarching history panel the values for the
        @history_spec at the given @algo_dt.
        """

        field = history_spec.field
        bar_count = history_spec.bar_count
        do_ffill = history_spec.ffill

        index = pd.to_datetime(index_at_dt(history_spec, algo_dt))
        return_frame = self.return_frames[history_spec.key_str]

        # Overwrite the index.
        # Not worrying about values here since the values are overwritten
        # in the next step.
        return_frame.index = index

        if bar_count > 1:
            # Get the last bar_count - 1 frames from our stored historical
            # frames.
            digest_panel = self.digest_panels[history_spec.frequency]\
                               .get_current()
            digest_frame = digest_panel[field].copy().ix[1 - bar_count:]
        else:
            digest_frame = None

        # Get minutes from our buffer panel to build the last row.
        buffer_frame = self.buffer_panel_minutes(
            earliest_minute=self.cur_window_starts[history_spec.frequency],
        )[field]

        if do_ffill:
            digest_frame = ffill_digest_frame_from_prior_values(
                field,
                digest_frame,
                self.last_known_prior_values,
            )
            buffer_frame = ffill_buffer_from_prior_values(
                field,
                buffer_frame,
                digest_frame,
                self.last_known_prior_values,
            )

        if digest_frame is not None:
            return_frame.ix[:-1] = digest_frame.ix[:]

        if field == 'volume':
            return_frame.ix[algo_dt] = buffer_frame.fillna(0).sum()
        elif field == 'high':
            return_frame.ix[algo_dt] = buffer_frame.max()
        elif field == 'low':
            return_frame.ix[algo_dt] = buffer_frame.min()
        elif field == 'open_price':
            return_frame.ix[algo_dt] = buffer_frame.iloc[0]
        else:
            return_frame.ix[algo_dt] = buffer_frame.loc[algo_dt]

        # Returning a copy of the DataFrame so that we don't crash if the user
        # adds columns to the frame.  Ideally we would just drop any added
        # columns, but pandas 0.12.0 doesn't support in-place dropping of
        # columns.  We should re-evaluate this implementation once we're on a
        # more up-to-date pandas.
        return return_frame.copy()
