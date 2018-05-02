import numpy as np
import pandas as pd

from six import viewvalues
from toolz import groupby, merge

from .base import PipelineLoader
from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.pipeline.loaders.utils import (
    next_event_indexer,
    previous_event_indexer,
)


def required_event_fields(next_value_columns, previous_value_columns):
    """
    Compute the set of resource columns required to serve
    ``next_value_columns`` and ``previous_value_columns``.
    """
    # These metadata columns are used to align event indexers.
    return {
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        EVENT_DATE_FIELD_NAME,
    }.union(
        # We also expect any of the field names that our loadable columns
        # are mapped to.
        viewvalues(next_value_columns),
        viewvalues(previous_value_columns),
    )


def validate_column_specs(events, next_value_columns, previous_value_columns):
    """
    Verify that the columns of ``events`` can be used by an EventsLoader to
    serve the BoundColumns described by ``next_value_columns`` and
    ``previous_value_columns``.
    """
    required = required_event_fields(next_value_columns,
                                     previous_value_columns)
    received = set(events.columns)
    missing = required - received
    if missing:
        raise ValueError(
            "EventsLoader missing required columns {missing}.\n"
            "Got Columns: {received}\n"
            "Expected Columns: {required}".format(
                missing=sorted(missing),
                received=sorted(received),
                required=sorted(required),
            )
        )


class EventsLoader(PipelineLoader):
    """
    Base class for PipelineLoaders that supports loading the next and previous
    value of an event field.

    Does not currently support adjustments.

    Parameters
    ----------
    events : pd.DataFrame
        A DataFrame representing events (e.g. share buybacks or
        earnings announcements) associated with particular companies.

        ``events`` must contain at least three columns::
            sid : int64
                The asset id associated with each event.

            event_date : datetime64[ns]
                The date on which the event occurred.

            timestamp : datetime64[ns]
                The date on which we learned about the event.

    next_value_columns : dict[BoundColumn -> str]
        Map from dataset columns to raw field names that should be used when
        searching for a next event value.

    previous_value_columns : dict[BoundColumn -> str]
        Map from dataset columns to raw field names that should be used when
        searching for a previous event value.
    """
    def __init__(self,
                 events,
                 next_value_columns,
                 previous_value_columns):
        validate_column_specs(
            events,
            next_value_columns,
            previous_value_columns,
        )

        events = events[events[EVENT_DATE_FIELD_NAME].notnull()]

        # We always work with entries from ``events`` directly as numpy arrays,
        # so we coerce from a frame to a dict of arrays here.
        self.events = {
            name: np.asarray(series)
            for name, series in (
                events.sort_values(EVENT_DATE_FIELD_NAME).iteritems()
            )
        }

        # Columns to load with self.load_next_events.
        self.next_value_columns = next_value_columns

        # Columns to load with self.load_previous_events.
        self.previous_value_columns = previous_value_columns

    def split_next_and_previous_event_columns(self, requested_columns):
        """
        Split requested columns into columns that should load the next known
        value and columns that should load the previous known value.

        Parameters
        ----------
        requested_columns : iterable[BoundColumn]

        Returns
        -------
        next_cols, previous_cols : iterable[BoundColumn], iterable[BoundColumn]
            ``requested_columns``, partitioned into sub-sequences based on
            whether the column should produce values from the next event or the
            previous event
        """
        def next_or_previous(c):
            if c in self.next_value_columns:
                return 'next'
            elif c in self.previous_value_columns:
                return 'previous'

            raise ValueError(
                "{c} not found in next_value_columns "
                "or previous_value_columns".format(c=c)
            )
        groups = groupby(next_or_previous, requested_columns)
        return groups.get('next', ()), groups.get('previous', ())

    def next_event_indexer(self, dates, sids):
        return next_event_indexer(
            dates,
            sids,
            self.events[EVENT_DATE_FIELD_NAME],
            self.events[TS_FIELD_NAME],
            self.events[SID_FIELD_NAME],
        )

    def previous_event_indexer(self, dates, sids):
        return previous_event_indexer(
            dates,
            sids,
            self.events[EVENT_DATE_FIELD_NAME],
            self.events[TS_FIELD_NAME],
            self.events[SID_FIELD_NAME],
        )

    def load_next_events(self, columns, dates, sids, mask):
        if not columns:
            return {}

        return self._load_events(
            name_map=self.next_value_columns,
            indexer=self.next_event_indexer(dates, sids),
            columns=columns,
            dates=dates,
            sids=sids,
            mask=mask,
        )

    def load_previous_events(self, columns, dates, sids, mask):
        if not columns:
            return {}

        return self._load_events(
            name_map=self.previous_value_columns,
            indexer=self.previous_event_indexer(dates, sids),
            columns=columns,
            dates=dates,
            sids=sids,
            mask=mask,
        )

    def _load_events(self, name_map, indexer, columns, dates, sids, mask):
        def to_frame(array):
            return pd.DataFrame(array, index=dates, columns=sids)

        assert indexer.shape == (len(dates), len(sids))

        out = {}
        for c in columns:
            # Array holding the value for column `c` for every event we have.
            col_array = self.events[name_map[c]]

            if not len(col_array):
                # We don't have **any** events, so return col.missing_value
                # every day for every sid. We have to special case empty events
                # because in normal branch we depend on being able to index
                # with -1 for missing values, which fails if there are no
                # events at all.
                raw = np.full(
                    (len(dates), len(sids)), c.missing_value, dtype=c.dtype
                )
            else:
                # Slot event values into sid/date locations using `indexer`.
                # This produces a 2D array of the same shape as `indexer`,
                # which must be (len(dates), len(sids))`.
                raw = col_array[indexer]

                # indexer will be -1 for locations where we don't have a known
                # value. Overwrite those locations with c.missing_value.
                raw[indexer < 0] = c.missing_value

            # Delegate the actual array formatting logic to a DataFrameLoader.
            loader = DataFrameLoader(c, to_frame(raw), adjustments=None)
            out[c] = loader.load_adjusted_array([c], dates, sids, mask)[c]
        return out

    def load_adjusted_array(self, columns, dates, sids, mask):
        n, p = self.split_next_and_previous_event_columns(columns)
        return merge(
            self.load_next_events(n, dates, sids, mask),
            self.load_previous_events(p, dates, sids, mask),
        )
