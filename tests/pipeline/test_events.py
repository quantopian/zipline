"""
Tests for setting up an EventsLoader.
"""
import re
from datetime import time
from itertools import product
from unittest import skipIf

import numpy as np
import pandas as pd
import pytest
import pytz

from zipline.pipeline import Pipeline, SimplePipelineEngine
from zipline.pipeline.common import EVENT_DATE_FIELD_NAME, SID_FIELD_NAME, TS_FIELD_NAME
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.domain import US_EQUITIES, EquitySessionDomain
from zipline.pipeline.loaders.events import EventsLoader
from zipline.pipeline.loaders.utils import next_event_indexer, previous_event_indexer
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithAssetFinder, WithTradingSessions
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import (
    categorical_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
)
from zipline.utils.pandas_utils import new_pandas, skip_pipeline_new_pandas


class EventDataSet(DataSet):
    previous_event_date = Column(dtype=datetime64ns_dtype)
    next_event_date = Column(dtype=datetime64ns_dtype)

    previous_float = Column(dtype=float64_dtype)
    next_float = Column(dtype=float64_dtype)

    previous_datetime = Column(dtype=datetime64ns_dtype)
    next_datetime = Column(dtype=datetime64ns_dtype)

    previous_int = Column(dtype=int64_dtype, missing_value=-1)
    next_int = Column(dtype=int64_dtype, missing_value=-1)

    previous_string = Column(dtype=categorical_dtype, missing_value=None)
    next_string = Column(dtype=categorical_dtype, missing_value=None)

    previous_string_custom_missing = Column(
        dtype=categorical_dtype,
        missing_value="<<NULL>>",
    )
    next_string_custom_missing = Column(
        dtype=categorical_dtype,
        missing_value="<<NULL>>",
    )


EventDataSet_US = EventDataSet.specialize(US_EQUITIES)

critical_dates = pd.to_datetime(
    [
        "2014-01-05",
        "2014-01-10",
        "2014-01-15",
        "2014-01-20",
    ]
)


def make_events_for_sid(sid, event_dates, event_timestamps):
    num_events = len(event_dates)
    return pd.DataFrame(
        {
            "sid": np.full(num_events, sid, dtype=np.int64),
            "timestamp": event_timestamps,
            "event_date": event_dates,
            "float": np.arange(num_events, dtype=np.float64) + sid,
            "int": np.arange(num_events) + sid,
            "datetime": pd.date_range("1990-01-01", periods=num_events).shift(sid),
            "string": ["-".join([str(sid), str(i)]) for i in range(num_events)],
        }
    )


def make_null_event_date_events(all_sids, timestamp):
    """
    Make an event with a null event_date for all sids.

    Used to test that EventsLoaders filter out null events.
    """
    return pd.DataFrame(
        {
            "sid": all_sids,
            "timestamp": timestamp,
            "event_date": pd.NaT,
            "float": -9999.0,
            "int": -9999,
            "datetime": pd.Timestamp("1980"),
            "string": "should be ignored",
        }
    )


def make_events(add_nulls):
    """
    Every event has at least three pieces of data associated with it:

    1. sid : The ID of the asset associated with the event.
    2. event_date : The date on which an event occurred.
    3. timestamp : The date on which we learned about the event.
                   This can be before the occurence_date in the case of an
                   announcement about an upcoming event.

    Events for two different sids shouldn't interact in any way, so the
    interesting cases are determined by the possible interleavings of
    event_date and timestamp for a single sid.

    Fix two events with dates e1, e2 and timestamps t1 and t2.

    Without loss of generality, assume that e1 < e2. (If two events have the
    same occurrence date, the behavior of next/previous event is undefined).

    The remaining possible sequences of events are given by taking all possible
    4-tuples of four ascending dates. For each possible interleaving, we
    generate a set of fake events with those dates and assign them to a new
    sid.
    """

    def gen_date_interleavings():
        for e1, e2, t1, t2 in product(*[critical_dates] * 4):
            if e1 < e2:
                yield (e1, e2, t1, t2)

    event_frames = []
    for sid, (e1, e2, t1, t2) in enumerate(gen_date_interleavings()):
        event_frames.append(make_events_for_sid(sid, [e1, e2], [t1, t2]))

    if add_nulls:
        for date in critical_dates:
            event_frames.append(
                make_null_event_date_events(
                    np.arange(sid + 1),
                    timestamp=date,
                )
            )

    return pd.concat(event_frames, ignore_index=True)


@pytest.fixture(scope="class")
def event(request):
    request.cls.events = make_events(add_nulls=False).sort_values("event_date")
    request.cls.events.reset_index(inplace=True)


@pytest.mark.usefixtures("event")
class TestEventIndexer:
    def test_previous_event_indexer(self):
        events = self.events
        event_sids = events["sid"].values
        event_dates = events["event_date"].values
        event_timestamps = events["timestamp"].values

        all_dates = pd.date_range("2014", "2014-01-31")
        all_sids = np.unique(event_sids)

        domain = EquitySessionDomain(
            all_dates,
            "US",
            time(8, 45, tzinfo=pytz.timezone("US/Eastern")),
        )

        indexer = previous_event_indexer(
            domain.data_query_cutoff_for_sessions(all_dates),
            all_sids,
            event_dates,
            event_timestamps,
            event_sids,
        )

        # Compute expected results without knowledge of null events.
        for i, sid in enumerate(all_sids):
            self.check_previous_event_indexer(
                events,
                all_dates,
                sid,
                indexer[:, i],
            )

    def check_previous_event_indexer(self, events, all_dates, sid, indexer):
        relevant_events = events[events.sid == sid]
        assert len(relevant_events) == 2

        ix1, ix2 = relevant_events.index

        # An event becomes a possible value once we're past both its event_date
        # and its timestamp.
        event1_first_eligible = max(
            relevant_events.loc[ix1, ["event_date", "timestamp"]],
        )
        event2_first_eligible = max(
            relevant_events.loc[ix2, ["event_date", "timestamp"]],
        )

        for date, computed_index in zip(all_dates, indexer):
            if date >= event2_first_eligible:
                # If we've seen event 2, it should win even if we've seen event
                # 1, because events are sorted by event_date.
                assert computed_index == ix2
            elif date >= event1_first_eligible:
                # If we've seen event 1 but not event 2, event 1 should win.
                assert computed_index == ix1
            else:
                # If we haven't seen either event, then we should have -1 as
                # sentinel.
                assert computed_index == -1

    def test_next_event_indexer(self):
        events = self.events
        event_sids = events["sid"].to_numpy()
        event_dates = events["event_date"].to_numpy()
        event_timestamps = events["timestamp"].to_numpy()

        all_dates = pd.date_range("2014", "2014-01-31")
        all_sids = np.unique(event_sids)

        domain = EquitySessionDomain(
            all_dates,
            "US",
            time(8, 45, tzinfo=pytz.timezone("US/Eastern")),
        )

        indexer = next_event_indexer(
            all_dates,
            domain.data_query_cutoff_for_sessions(all_dates),
            all_sids,
            event_dates,
            event_timestamps,
            event_sids,
        )

        # Compute expected results without knowledge of null events.
        for i, sid in enumerate(all_sids):
            self.check_next_event_indexer(
                events,
                all_dates,
                sid,
                indexer[:, i],
            )

    def check_next_event_indexer(self, events, all_dates, sid, indexer):
        relevant_events = events[events.sid == sid]
        assert len(relevant_events) == 2

        ix1, ix2 = relevant_events.index

        e1, e2 = relevant_events["event_date"]
        t1, t2 = relevant_events["timestamp"]

        for date, computed_index in zip(all_dates, indexer):
            # An event is eligible to be the next event if it's between the
            # timestamp and the event_date, inclusive.
            if t1 <= date <= e1:
                # If e1 is eligible, it should be chosen even if e2 is
                # eligible, since it's earlier.
                assert computed_index == ix1
            elif t2 <= date <= e2:
                # e2 is eligible and e1 is not, so e2 should be chosen.
                assert computed_index == ix2
            else:
                # Neither event is eligible.  Return -1 as a sentinel.
                assert computed_index == -1


class EventsLoaderEmptyTestCase(WithAssetFinder, WithTradingSessions, ZiplineTestCase):
    START_DATE = pd.Timestamp("2014-01-01")
    END_DATE = pd.Timestamp("2014-01-30")
    ASSET_FINDER_COUNTRY_CODE = "US"

    @classmethod
    def init_class_fixtures(cls):
        cls.ASSET_FINDER_EQUITY_SIDS = [0, 1]
        cls.ASSET_FINDER_EQUITY_SYMBOLS = ["A", "B"]
        super(EventsLoaderEmptyTestCase, cls).init_class_fixtures()

    def frame_containing_all_missing_values(self, index, columns):
        frame = pd.DataFrame(
            index=index,
            data={c.name: c.missing_value for c in EventDataSet.columns},
        )
        for c in columns:
            # The construction above produces columns of dtype `object` when
            # the missing value is string, but we expect categoricals in the
            # final result.
            if c.dtype == categorical_dtype:
                frame[c.name] = frame[c.name].astype("category")
        return frame

    @skipIf(new_pandas, skip_pipeline_new_pandas)
    def test_load_empty(self):
        """
        For the case where raw data is empty, make sure we have a result for
        all sids, that the dimensions are correct, and that we have the
        correct missing value.
        """
        raw_events = pd.DataFrame(
            columns=[
                "sid",
                "timestamp",
                "event_date",
                "float",
                "int",
                "datetime",
                "string",
            ]
        )
        next_value_columns = {
            EventDataSet_US.next_datetime: "datetime",
            EventDataSet_US.next_event_date: "event_date",
            EventDataSet_US.next_float: "float",
            EventDataSet_US.next_int: "int",
            EventDataSet_US.next_string: "string",
            EventDataSet_US.next_string_custom_missing: "string",
        }
        previous_value_columns = {
            EventDataSet_US.previous_datetime: "datetime",
            EventDataSet_US.previous_event_date: "event_date",
            EventDataSet_US.previous_float: "float",
            EventDataSet_US.previous_int: "int",
            EventDataSet_US.previous_string: "string",
            EventDataSet_US.previous_string_custom_missing: "string",
        }
        loader = EventsLoader(raw_events, next_value_columns, previous_value_columns)
        engine = SimplePipelineEngine(
            lambda x: loader,
            self.asset_finder,
        )

        results = engine.run_pipeline(
            Pipeline(
                {c.name: c.latest for c in EventDataSet_US.columns}, domain=US_EQUITIES
            ),
            start_date=self.trading_days[0],
            end_date=self.trading_days[-1],
        )

        assets = self.asset_finder.retrieve_all(self.ASSET_FINDER_EQUITY_SIDS)
        dates = self.trading_days

        expected = self.frame_containing_all_missing_values(
            index=pd.MultiIndex.from_product([dates, assets]),
            columns=EventDataSet_US.columns,
        )

        assert_equal(results, expected)


class EventsLoaderTestCase(WithAssetFinder, WithTradingSessions, ZiplineTestCase):
    START_DATE = pd.Timestamp("2014-01-01")
    END_DATE = pd.Timestamp("2014-01-30")
    ASSET_FINDER_COUNTRY_CODE = "US"

    @classmethod
    def init_class_fixtures(cls):
        # This is a rare case where we actually want to do work **before** we
        # call init_class_fixtures.  We choose our sids for WithAssetFinder
        # based on the events generated by make_event_data.
        cls.raw_events = make_events(add_nulls=True)
        cls.raw_events_no_nulls = cls.raw_events[cls.raw_events["event_date"].notnull()]
        cls.next_value_columns = {
            EventDataSet_US.next_datetime: "datetime",
            EventDataSet_US.next_event_date: "event_date",
            EventDataSet_US.next_float: "float",
            EventDataSet_US.next_int: "int",
            EventDataSet_US.next_string: "string",
            EventDataSet_US.next_string_custom_missing: "string",
        }
        cls.previous_value_columns = {
            EventDataSet_US.previous_datetime: "datetime",
            EventDataSet_US.previous_event_date: "event_date",
            EventDataSet_US.previous_float: "float",
            EventDataSet_US.previous_int: "int",
            EventDataSet_US.previous_string: "string",
            EventDataSet_US.previous_string_custom_missing: "string",
        }
        cls.loader = cls.make_loader(
            events=cls.raw_events,
            next_value_columns=cls.next_value_columns,
            previous_value_columns=cls.previous_value_columns,
        )
        cls.ASSET_FINDER_EQUITY_SIDS = list(cls.raw_events["sid"].unique())
        cls.ASSET_FINDER_EQUITY_SYMBOLS = [
            "s" + str(n) for n in cls.ASSET_FINDER_EQUITY_SIDS
        ]
        super(EventsLoaderTestCase, cls).init_class_fixtures()

        cls.engine = SimplePipelineEngine(
            lambda c: cls.loader,
            asset_finder=cls.asset_finder,
            default_domain=US_EQUITIES,
        )

    @classmethod
    def make_loader(cls, events, next_value_columns, previous_value_columns):
        # This method exists to be overridden by EventsLoaderTestCases using alternative loaders
        return EventsLoader(events, next_value_columns, previous_value_columns)

    @skipIf(new_pandas, skip_pipeline_new_pandas)
    def test_load_with_trading_calendar(self):
        results = self.engine.run_pipeline(
            Pipeline({c.name: c.latest for c in EventDataSet_US.columns}),
            start_date=self.trading_days[0],
            end_date=self.trading_days[-1],
        )

        for c in EventDataSet_US.columns:
            if c in self.next_value_columns:
                self.check_next_value_results(
                    c,
                    results[c.name].unstack(),
                    self.trading_days,
                )
            elif c in self.previous_value_columns:
                self.check_previous_value_results(
                    c,
                    results[c.name].unstack(),
                    self.trading_days,
                )
            else:
                raise AssertionError("Unexpected column %s." % c)

    @skipIf(new_pandas, skip_pipeline_new_pandas)
    def test_load_properly_forward_fills(self):

        # Cut the dates in half so we need to forward fill some data which
        # is not in our window. The results should be computed the same as if
        # we had computed across the entire window and then sliced after the
        # computation.
        dates = self.trading_days[len(self.trading_days) // 2 :]
        results = self.engine.run_pipeline(
            Pipeline({c.name: c.latest for c in EventDataSet_US.columns}),
            start_date=dates[0],
            end_date=dates[-1],
        )

        for c in EventDataSet_US.columns:
            if c in self.next_value_columns:
                self.check_next_value_results(
                    c,
                    results[c.name].unstack(),
                    dates,
                )
            elif c in self.previous_value_columns:
                self.check_previous_value_results(
                    c,
                    results[c.name].unstack(),
                    dates,
                )
            else:
                raise AssertionError("Unexpected column %s." % c)

    def assert_result_contains_all_sids(self, results):
        assert_equal(
            list(map(int, results.columns)),
            self.ASSET_FINDER_EQUITY_SIDS,
        )

    def check_previous_value_results(self, column, results, dates):
        """
        Check previous value results for a single column.
        """
        # Verify that we got a result for every sid.
        self.assert_result_contains_all_sids(results)

        events = self.raw_events_no_nulls
        # Remove timezone info from trading days, since the outputs
        # from pandas won't be tz_localized.
        dates = dates.tz_localize(None)

        for asset, asset_result in results.iteritems():
            relevant_events = events[events.sid == asset.sid]
            assert len(relevant_events) == 2

            v1, v2 = relevant_events[self.previous_value_columns[column]]
            event1_first_eligible = max(
                # .ix doesn't work here because the frame index contains
                # integers, so 0 is still interpreted as a key.
                relevant_events.iloc[0].loc[["event_date", "timestamp"]],
            )
            event2_first_eligible = max(
                relevant_events.iloc[1].loc[["event_date", "timestamp"]]
            )

            for date, computed_value in zip(dates, asset_result):
                if date >= event2_first_eligible:
                    # If we've seen event 2, it should win even if we've seen
                    # event 1, because events are sorted by event_date.
                    assert computed_value == v2
                elif date >= event1_first_eligible:
                    # If we've seen event 1 but not event 2, event 1 should
                    # win.
                    assert computed_value == v1
                else:
                    # If we haven't seen either event, then we should have
                    # column.missing_value.
                    assert_equal(
                        computed_value,
                        column.missing_value,
                        # Coerce from Timestamp to datetime64.
                        allow_datetime_coercions=True,
                    )

    def check_next_value_results(self, column, results, dates):
        """
        Check results for a single column.
        """
        self.assert_result_contains_all_sids(results)

        events = self.raw_events_no_nulls
        # Remove timezone info from trading days, since the outputs
        # from pandas won't be tz_localized.
        dates = dates.tz_localize(None)
        for asset, asset_result in results.iteritems():
            relevant_events = events[events.sid == asset.sid]
            assert len(relevant_events) == 2

            v1, v2 = relevant_events[self.next_value_columns[column]]
            e1, e2 = relevant_events["event_date"]
            t1, t2 = relevant_events["timestamp"]

            for date, computed_value in zip(dates, asset_result):
                if t1 <= date <= e1:
                    # If we've seen event 2, it should win even if we've seen
                    # event 1, because events are sorted by event_date.
                    assert computed_value == v1
                elif t2 <= date <= e2:
                    # If we've seen event 1 but not event 2, event 1 should
                    # win.
                    assert computed_value == v2
                else:
                    # If we haven't seen either event, then we should have
                    # column.missing_value.
                    assert_equal(
                        computed_value,
                        column.missing_value,
                        # Coerce from Timestamp to datetime64.
                        allow_datetime_coercions=True,
                    )

    def test_wrong_cols(self):
        # Test wrong cols (cols != expected)
        events = pd.DataFrame(
            {
                "c": [5],
                SID_FIELD_NAME: [1],
                TS_FIELD_NAME: [pd.Timestamp("2014")],
                EVENT_DATE_FIELD_NAME: [pd.Timestamp("2014")],
            }
        )

        EventsLoader(events, {EventDataSet_US.next_float: "c"}, {})
        EventsLoader(events, {}, {EventDataSet_US.previous_float: "c"})

        expected = (
            "EventsLoader missing required columns ['d'].\n"
            "Got Columns: ['c', 'event_date', 'sid', 'timestamp']\n"
            "Expected Columns: ['d', 'event_date', 'sid', 'timestamp']"
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            EventsLoader(events, {EventDataSet_US.next_float: "d"}, {})
