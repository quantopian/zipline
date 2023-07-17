from datetime import timedelta
from functools import partial

import itertools
from parameterized import parameterized
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import pandas as pd
from toolz import merge
from zipline.pipeline import SimplePipelineEngine, Pipeline, CustomFactor
from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.data import DataSet
from zipline.pipeline.data import Column
from zipline.pipeline.domain import EquitySessionDomain
from zipline.pipeline.loaders.earnings_estimates import (
    NextEarningsEstimatesLoader,
    NextSplitAdjustedEarningsEstimatesLoader,
    normalize_quarters,
    PreviousEarningsEstimatesLoader,
    PreviousSplitAdjustedEarningsEstimatesLoader,
    split_normalized_quarters,
)
from zipline.testing.fixtures import (
    WithAdjustmentReader,
    WithTradingSessions,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal
from zipline.testing.predicates import assert_frame_equal
from zipline.utils.numpy_utils import datetime64ns_dtype
from zipline.utils.numpy_utils import float64_dtype
import pytest


class Estimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate = Column(dtype=float64_dtype)


class MultipleColumnsEstimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate1 = Column(dtype=float64_dtype)
    estimate2 = Column(dtype=float64_dtype)


def QuartersEstimates(announcements_out):
    class QtrEstimates(Estimates):
        num_announcements = announcements_out
        name = Estimates

    return QtrEstimates


def MultipleColumnsQuartersEstimates(announcements_out):
    class QtrEstimates(MultipleColumnsEstimates):
        num_announcements = announcements_out
        name = Estimates

    return QtrEstimates


def QuartersEstimatesNoNumQuartersAttr(num_qtr):
    class QtrEstimates(Estimates):
        name = Estimates

    return QtrEstimates


def create_expected_df_for_factor_compute(start_date, sids, tuples, end_date):
    """Given a list of tuples of new data we get for each sid on each critical
    date (when information changes), create a DataFrame that fills that
    data through a date range ending at `end_date`.
    """
    df = pd.DataFrame(tuples, columns=[SID_FIELD_NAME, "estimate", "knowledge_date"])
    df = df.pivot_table(
        columns=SID_FIELD_NAME, values="estimate", index="knowledge_date", dropna=False
    )
    df = df.reindex(pd.date_range(start_date, end_date))
    # Index name is lost during reindex.
    df.index = df.index.rename("knowledge_date")
    df["at_date"] = end_date
    df = df.set_index(["at_date", df.index]).ffill()
    new_sids = set(sids) - set(df.columns)
    df = df.reindex(columns=df.columns.union(new_sids))
    return df


class WithEstimates(WithTradingSessions, WithAdjustmentReader):
    """ZiplineTestCase mixin providing cls.loader and cls.events as class
    level fixtures.


    Methods
    -------
    make_loader(events, columns) -> PipelineLoader
        Method which returns the loader to be used throughout tests.

        events : pd.DataFrame
            The raw events to be used as input to the pipeline loader.
        columns : dict[str -> str]
            The dictionary mapping the names of BoundColumns to the
            associated column name in the events DataFrame.
    make_columns() -> dict[BoundColumn -> str]
       Method which returns a dictionary of BoundColumns mapped to the
       associated column names in the raw data.
    """

    # Short window defined in order for test to run faster.
    START_DATE = pd.Timestamp("2014-12-28")
    END_DATE = pd.Timestamp("2015-02-04")

    @classmethod
    def make_loader(cls, events, columns):
        raise NotImplementedError("make_loader")

    @classmethod
    def make_events(cls):
        raise NotImplementedError("make_events")

    @classmethod
    def get_sids(cls):
        return cls.events[SID_FIELD_NAME].unique()

    @classmethod
    def make_columns(cls):
        return {
            Estimates.event_date: "event_date",
            Estimates.fiscal_quarter: "fiscal_quarter",
            Estimates.fiscal_year: "fiscal_year",
            Estimates.estimate: "estimate",
        }

    def make_engine(self, loader=None):
        if loader is None:
            loader = self.loader

        return SimplePipelineEngine(
            lambda x: loader,
            self.asset_finder,
            default_domain=EquitySessionDomain(
                self.trading_days,
                self.ASSET_FINDER_COUNTRY_CODE,
            ),
        )

    @classmethod
    def init_class_fixtures(cls):
        cls.events = cls.make_events()
        cls.ASSET_FINDER_EQUITY_SIDS = cls.get_sids()
        cls.ASSET_FINDER_EQUITY_SYMBOLS = [
            "s" + str(n) for n in cls.ASSET_FINDER_EQUITY_SIDS
        ]
        # We need to instantiate certain constants needed by supers of
        # `WithEstimates` before we call their `init_class_fixtures`.
        super(WithEstimates, cls).init_class_fixtures()
        cls.columns = cls.make_columns()
        # Some tests require `WithAdjustmentReader` to be set up by the time we
        # make the loader.
        cls.loader = cls.make_loader(
            cls.events, {column.name: val for column, val in cls.columns.items()}
        )


class WithOneDayPipeline(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events as a class level fixture and
    defining a test for all inheritors to use.

    Attributes
    ----------
    events : pd.DataFrame
        A simple DataFrame with columns needed for estimates and a single sid
        and no other data.

    Tests
    ------
    test_wrong_num_announcements_passed()
        Tests that loading with an incorrect quarter number raises an error.
    test_no_num_announcements_attr()
        Tests that the loader throws an AssertionError if the dataset being
        loaded has no `num_announcements` attribute.
    """

    @classmethod
    def make_columns(cls):
        return {
            MultipleColumnsEstimates.event_date: "event_date",
            MultipleColumnsEstimates.fiscal_quarter: "fiscal_quarter",
            MultipleColumnsEstimates.fiscal_year: "fiscal_year",
            MultipleColumnsEstimates.estimate1: "estimate1",
            MultipleColumnsEstimates.estimate2: "estimate2",
        }

    @classmethod
    def make_events(cls):
        return pd.DataFrame(
            {
                SID_FIELD_NAME: [0] * 2,
                TS_FIELD_NAME: [pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-06")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-10"),
                    pd.Timestamp("2015-01-20"),
                ],
                "estimate1": [1.0, 2.0],
                "estimate2": [3.0, 4.0],
                FISCAL_QUARTER_FIELD_NAME: [1, 2],
                FISCAL_YEAR_FIELD_NAME: [2015, 2015],
            }
        )

    @classmethod
    def make_expected_out(cls):
        raise NotImplementedError("make_expected_out")

    @classmethod
    def init_class_fixtures(cls):
        super(WithOneDayPipeline, cls).init_class_fixtures()
        cls.sid0 = cls.asset_finder.retrieve_asset(0)
        cls.expected_out = cls.make_expected_out()

    def test_load_one_day(self):
        # We want to test multiple columns
        dataset = MultipleColumnsQuartersEstimates(1)
        engine = self.make_engine()
        results = engine.run_pipeline(
            Pipeline({c.name: c.latest for c in dataset.columns}),
            start_date=pd.Timestamp("2015-01-15"),
            end_date=pd.Timestamp("2015-01-15"),
        )

        assert_frame_equal(
            results.sort_index(axis=1), self.expected_out.sort_index(axis=1)
        )


class PreviousWithOneDayPipeline(WithOneDayPipeline, ZiplineTestCase):
    """
    Tests that previous quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_out(cls):
        return pd.DataFrame(
            {
                EVENT_DATE_FIELD_NAME: pd.Timestamp("2015-01-10"),
                "estimate1": 1.0,
                "estimate2": 3.0,
                FISCAL_QUARTER_FIELD_NAME: 1.0,
                FISCAL_YEAR_FIELD_NAME: 2015.0,
            },
            index=pd.MultiIndex.from_tuples(((pd.Timestamp("2015-01-15"), cls.sid0),)),
        )


class NextWithOneDayPipeline(WithOneDayPipeline, ZiplineTestCase):
    """
    Tests that next quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_out(cls):
        return pd.DataFrame(
            {
                EVENT_DATE_FIELD_NAME: pd.Timestamp("2015-01-20"),
                "estimate1": 2.0,
                "estimate2": 4.0,
                FISCAL_QUARTER_FIELD_NAME: 2.0,
                FISCAL_YEAR_FIELD_NAME: 2015.0,
            },
            index=pd.MultiIndex.from_tuples(((pd.Timestamp("2015-01-15"), cls.sid0),)),
        )


dummy_df = pd.DataFrame(
    {SID_FIELD_NAME: 0},
    columns=[
        SID_FIELD_NAME,
        TS_FIELD_NAME,
        EVENT_DATE_FIELD_NAME,
        FISCAL_QUARTER_FIELD_NAME,
        FISCAL_YEAR_FIELD_NAME,
        "estimate",
    ],
    index=[0],
)


class WithWrongLoaderDefinition(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events as a class level fixture and
    defining a test for all inheritors to use.

    Attributes
    ----------
    events : pd.DataFrame
        A simple DataFrame with columns needed for estimates and a single sid
        and no other data.

    Tests
    ------
    test_wrong_num_announcements_passed()
        Tests that loading with an incorrect quarter number raises an error.
    test_no_num_announcements_attr()
        Tests that the loader throws an AssertionError if the dataset being
        loaded has no `num_announcements` attribute.
    """

    @classmethod
    def make_events(cls):
        return dummy_df

    def test_wrong_num_announcements_passed(self):
        bad_dataset1 = QuartersEstimates(-1)
        bad_dataset2 = QuartersEstimates(-2)
        good_dataset = QuartersEstimates(1)
        engine = self.make_engine()
        columns = {
            c.name + str(dataset.num_announcements): c.latest
            for dataset in (bad_dataset1, bad_dataset2, good_dataset)
            for c in dataset.columns
        }
        p = Pipeline(columns)

        err_msg = (
            r"Passed invalid number of quarters -[0-9],-[0-9]; "
            r"must pass a number of quarters >= 0"
        )
        with pytest.raises(ValueError, match=err_msg):
            engine.run_pipeline(
                p,
                start_date=self.trading_days[0],
                end_date=self.trading_days[-1],
            )

    def test_no_num_announcements_attr(self):
        dataset = QuartersEstimatesNoNumQuartersAttr(1)
        engine = self.make_engine()
        p = Pipeline({c.name: c.latest for c in dataset.columns})

        with pytest.raises(AttributeError):
            engine.run_pipeline(
                p,
                start_date=self.trading_days[0],
                end_date=self.trading_days[-1],
            )


class PreviousWithWrongNumQuarters(WithWrongLoaderDefinition, ZiplineTestCase):
    """
    Tests that previous quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)


class NextWithWrongNumQuarters(WithWrongLoaderDefinition, ZiplineTestCase):
    """
    Tests that next quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)


options = [
    "split_adjustments_loader",
    "split_adjusted_column_names",
    "split_adjusted_asof",
]


class WrongSplitsLoaderDefinition(WithEstimates, ZiplineTestCase):
    """
    Test class that tests that loaders break correctly when incorrectly
    instantiated.

    Tests
    -----
    test_extra_splits_columns_passed(SplitAdjustedEstimatesLoader)
        A test that checks that the loader correctly breaks when an
        unexpected column is passed in the list of split-adjusted columns.
    """

    @classmethod
    def init_class_fixtures(cls):
        super(WithEstimates, cls).init_class_fixtures()

    @parameterized.expand(
        itertools.product(
            (
                NextSplitAdjustedEarningsEstimatesLoader,
                PreviousSplitAdjustedEarningsEstimatesLoader,
            ),
        )
    )
    def test_extra_splits_columns_passed(self, loader):
        columns = {
            Estimates.event_date: "event_date",
            Estimates.fiscal_quarter: "fiscal_quarter",
            Estimates.fiscal_year: "fiscal_year",
            Estimates.estimate: "estimate",
        }

        with pytest.raises(ValueError):
            loader(
                dummy_df,
                {column.name: val for column, val in columns.items()},
                split_adjustments_loader=self.adjustment_reader,
                split_adjusted_column_names=["estimate", "extra_col"],
                split_adjusted_asof=pd.Timestamp("2015-01-01"),
            )


class WithEstimatesTimeZero(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events as a class level fixture and
    defining a test for all inheritors to use.

    Attributes
    ----------
    cls.events : pd.DataFrame
        Generated dynamically in order to test inter-leavings of estimates and
        event dates for multiple quarters to make sure that we select the
        right immediate 'next' or 'previous' quarter relative to each date -
        i.e., the right 'time zero' on the timeline. We care about selecting
        the right 'time zero' because we use that to calculate which quarter's
        data needs to be returned for each day.

    Methods
    -------
    get_expected_estimate(q1_knowledge,
                          q2_knowledge,
                          comparable_date) -> pd.DataFrame
        Retrieves the expected estimate given the latest knowledge about each
        quarter and the date on which the estimate is being requested. If
        there is no expected estimate, returns an empty DataFrame.

    Tests
    ------
    test_estimates()
        Tests that we get the right 'time zero' value on each day for each
        sid and for each column.
    """

    # Shorter date range for performance
    END_DATE = pd.Timestamp("2015-01-28")

    q1_knowledge_dates = [
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2015-01-04"),
        pd.Timestamp("2015-01-07"),
        pd.Timestamp("2015-01-11"),
    ]
    q2_knowledge_dates = [
        pd.Timestamp("2015-01-14"),
        pd.Timestamp("2015-01-17"),
        pd.Timestamp("2015-01-20"),
        pd.Timestamp("2015-01-23"),
    ]
    # We want to model the possibility of an estimate predicting a release date
    # that doesn't match the actual release. This could be done by dynamically
    # generating more combinations with different release dates, but that
    # significantly increases the amount of time it takes to run the tests.
    # These hard-coded cases are sufficient to know that we can update our
    # beliefs when we get new information.
    q1_release_dates = [
        pd.Timestamp("2015-01-13"),
        pd.Timestamp("2015-01-14"),
    ]  # One day late
    q2_release_dates = [
        pd.Timestamp("2015-01-25"),  # One day early
        pd.Timestamp("2015-01-26"),
    ]

    @classmethod
    def make_events(cls):
        """
        In order to determine which estimate we care about for a particular
        sid, we need to look at all estimates that we have for that sid and
        their associated event dates.

        We define q1 < q2, and thus event1 < event2 since event1 occurs
        during q1 and event2 occurs during q2 and we assume that there can
        only be 1 event per quarter. We assume that there can be multiple
        estimates per quarter leading up to the event. We assume that estimates
        will not surpass the relevant event date. We will look at 2 estimates
        for an event before the event occurs, since that is the simplest
        scenario that covers the interesting edge cases:
            - estimate values changing
            - a release date changing
            - estimates for different quarters interleaving

        Thus, we generate all possible inter-leavings of 2 estimates per
        quarter-event where estimate1 < estimate2 and all estimates are < the
        relevant event and assign each of these inter-leavings to a
        different sid.
        """

        sid_estimates = []
        sid_releases = []
        # We want all permutations of 2 knowledge dates per quarter.
        it = enumerate(
            itertools.permutations(cls.q1_knowledge_dates + cls.q2_knowledge_dates, 4)
        )
        for sid, (q1e1, q1e2, q2e1, q2e2) in it:
            # We're assuming that estimates must come before the relevant
            # release.
            if (
                q1e1 < q1e2
                and q2e1 < q2e2
                # All estimates are < Q2's event, so just constrain Q1
                # estimates.
                and q1e1 < cls.q1_release_dates[0]
                and q1e2 < cls.q1_release_dates[0]
            ):
                sid_estimates.append(
                    cls.create_estimates_df(q1e1, q1e2, q2e1, q2e2, sid)
                )
                sid_releases.append(cls.create_releases_df(sid))
        return pd.concat(sid_estimates + sid_releases).reset_index(drop=True)

    @classmethod
    def get_sids(cls):
        sids = cls.events[SID_FIELD_NAME].unique()
        # Tack on an extra sid to make sure that sids with no data are
        # included but have all-null columns.
        return list(sids) + [max(sids) + 1]

    @classmethod
    def create_releases_df(cls, sid):
        # Final release dates never change. The quarters have very tight date
        # ranges in order to reduce the number of dates we need to iterate
        # through when testing.
        return pd.DataFrame(
            {
                TS_FIELD_NAME: [pd.Timestamp("2015-01-13"), pd.Timestamp("2015-01-26")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-13"),
                    pd.Timestamp("2015-01-26"),
                ],
                "estimate": [0.5, 0.8],
                FISCAL_QUARTER_FIELD_NAME: [1.0, 2.0],
                FISCAL_YEAR_FIELD_NAME: [2015.0, 2015.0],
                SID_FIELD_NAME: sid,
            }
        )

    @classmethod
    def create_estimates_df(cls, q1e1, q1e2, q2e1, q2e2, sid):
        return pd.DataFrame(
            {
                EVENT_DATE_FIELD_NAME: cls.q1_release_dates + cls.q2_release_dates,
                "estimate": [0.1, 0.2, 0.3, 0.4],
                FISCAL_QUARTER_FIELD_NAME: [1.0, 1.0, 2.0, 2.0],
                FISCAL_YEAR_FIELD_NAME: [2015.0, 2015.0, 2015.0, 2015.0],
                TS_FIELD_NAME: [q1e1, q1e2, q2e1, q2e2],
                SID_FIELD_NAME: sid,
            }
        )

    def get_expected_estimate(self, q1_knowledge, q2_knowledge, comparable_date):
        return pd.DataFrame()

    def test_estimates(self):
        dataset = QuartersEstimates(1)
        engine = self.make_engine()
        results = engine.run_pipeline(
            Pipeline({c.name: c.latest for c in dataset.columns}),
            start_date=self.trading_days[1],
            end_date=self.trading_days[-2],
        )
        for sid in self.ASSET_FINDER_EQUITY_SIDS:
            sid_estimates = results.xs(sid, level=1)
            # Separate assertion for all-null DataFrame to avoid setting
            # column dtypes on `all_expected`.
            if sid == max(self.ASSET_FINDER_EQUITY_SIDS):
                assert sid_estimates.isnull().all().all()
            else:
                ts_sorted_estimates = self.events[
                    self.events[SID_FIELD_NAME] == sid
                ].sort_values(TS_FIELD_NAME)
                q1_knowledge = ts_sorted_estimates[
                    ts_sorted_estimates[FISCAL_QUARTER_FIELD_NAME] == 1
                ]
                q2_knowledge = ts_sorted_estimates[
                    ts_sorted_estimates[FISCAL_QUARTER_FIELD_NAME] == 2
                ]
                all_expected = pd.concat(
                    [
                        self.get_expected_estimate(
                            q1_knowledge[
                                q1_knowledge[TS_FIELD_NAME] <= date.tz_localize(None)
                            ],
                            q2_knowledge[
                                q2_knowledge[TS_FIELD_NAME] <= date.tz_localize(None)
                            ],
                            date.tz_localize(None),
                        ).set_index([[date]])
                        for date in sid_estimates.index
                    ],
                    axis=0,
                )
                sid_estimates.index = all_expected.index.copy()
                assert_equal(all_expected[sid_estimates.columns], sid_estimates)


class NextEstimate(WithEstimatesTimeZero, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    def get_expected_estimate(self, q1_knowledge, q2_knowledge, comparable_date):
        # If our latest knowledge of q1 is that the release is
        # happening on this simulation date or later, then that's
        # the estimate we want to use.
        if (
            not q1_knowledge.empty
            and q1_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] >= comparable_date
        ):
            return q1_knowledge.iloc[-1:]
        # If q1 has already happened or we don't know about it
        # yet and our latest knowledge indicates that q2 hasn't
        # happened yet, then that's the estimate we want to use.
        elif (
            not q2_knowledge.empty
            and q2_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] >= comparable_date
        ):
            return q2_knowledge.iloc[-1:]
        return pd.DataFrame(columns=q1_knowledge.columns, index=[comparable_date])


class PreviousEstimate(WithEstimatesTimeZero, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    def get_expected_estimate(self, q1_knowledge, q2_knowledge, comparable_date):

        # The expected estimate will be for q2 if the last thing
        # we've seen is that the release date already happened.
        # Otherwise, it'll be for q1, as long as the release date
        # for q1 has already happened.
        if (
            not q2_knowledge.empty
            and q2_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] <= comparable_date
        ):
            return q2_knowledge.iloc[-1:]
        elif (
            not q1_knowledge.empty
            and q1_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] <= comparable_date
        ):
            return q1_knowledge.iloc[-1:]
        return pd.DataFrame(columns=q1_knowledge.columns, index=[comparable_date])


class WithEstimateMultipleQuarters(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events, cls.make_expected_out as
    class-level fixtures and self.test_multiple_qtrs_requested as a test.

    Attributes
    ----------
    events : pd.DataFrame
        Simple DataFrame with estimates for 2 quarters for a single sid.

    Methods
    -------
    make_expected_out() --> pd.DataFrame
        Returns the DataFrame that is expected as a result of running a
        Pipeline where estimates are requested for multiple quarters out.
    fill_expected_out(expected)
        Fills the expected DataFrame with data.

    Tests
    ------
    test_multiple_qtrs_requested()
        Runs a Pipeline that calculate which estimates for multiple quarters
        out and checks that the returned columns contain data for the correct
        number of quarters out.
    """

    @classmethod
    def make_events(cls):
        return pd.DataFrame(
            {
                SID_FIELD_NAME: [0] * 2,
                TS_FIELD_NAME: [pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-06")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-10"),
                    pd.Timestamp("2015-01-20"),
                ],
                "estimate": [1.0, 2.0],
                FISCAL_QUARTER_FIELD_NAME: [1, 2],
                FISCAL_YEAR_FIELD_NAME: [2015, 2015],
            }
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithEstimateMultipleQuarters, cls).init_class_fixtures()
        cls.expected_out = cls.make_expected_out()

    @classmethod
    def make_expected_out(cls):
        expected = pd.DataFrame(
            columns=[cls.columns[col] + "1" for col in cls.columns]
            + [cls.columns[col] + "2" for col in cls.columns],
            index=cls.trading_days,
        )

        for (col, raw_name), suffix in itertools.product(
            cls.columns.items(), ("1", "2")
        ):
            expected_name = raw_name + suffix
            if col.dtype == datetime64ns_dtype:
                expected[expected_name] = pd.to_datetime(expected[expected_name])
            else:
                expected[expected_name] = expected[expected_name].astype(col.dtype)
        cls.fill_expected_out(expected)
        return expected.reindex(cls.trading_days)

    def test_multiple_qtrs_requested(self):
        dataset1 = QuartersEstimates(1)
        dataset2 = QuartersEstimates(2)
        engine = self.make_engine()

        results = engine.run_pipeline(
            Pipeline(
                merge(
                    [
                        {c.name + "1": c.latest for c in dataset1.columns},
                        {c.name + "2": c.latest for c in dataset2.columns},
                    ]
                )
            ),
            start_date=self.trading_days[0],
            end_date=self.trading_days[-1],
        )
        q1_columns = [col.name + "1" for col in self.columns]
        q2_columns = [col.name + "2" for col in self.columns]

        # We now expect a column for 1 quarter out and a column for 2
        # quarters out for each of the dataset columns.
        assert_equal(
            sorted(np.array(q1_columns + q2_columns)), sorted(results.columns.values)
        )
        assert_equal(
            self.expected_out.sort_index(axis=1),
            results.xs(0, level=1).sort_index(axis=1),
        )


class NextEstimateMultipleQuarters(WithEstimateMultipleQuarters, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def fill_expected_out(cls, expected):
        # Fill columns for 1 Q out
        for raw_name in cls.columns.values():
            expected.loc[
                pd.Timestamp("2015-01-01") : pd.Timestamp("2015-01-11"),
                raw_name + "1",
            ] = cls.events[raw_name].iloc[0]
            expected.loc[
                pd.Timestamp("2015-01-11") : pd.Timestamp("2015-01-20"),
                raw_name + "1",
            ] = cls.events[raw_name].iloc[1]

        # Fill columns for 2 Q out
        # We only have an estimate and event date for 2 quarters out before
        # Q1's event happens; after Q1's event, we know 1 Q out but not 2 Qs
        # out.
        for col_name in ["estimate", "event_date"]:
            expected.loc[
                pd.Timestamp("2015-01-06") : pd.Timestamp("2015-01-10"),
                col_name + "2",
            ] = cls.events[col_name].iloc[1]
        # But we know what FQ and FY we'd need in both Q1 and Q2
        # because we know which FQ is next and can calculate from there
        expected.loc[
            pd.Timestamp("2015-01-01") : pd.Timestamp("2015-01-09"),
            FISCAL_QUARTER_FIELD_NAME + "2",
        ] = 2
        expected.loc[
            pd.Timestamp("2015-01-12") : pd.Timestamp("2015-01-20"),
            FISCAL_QUARTER_FIELD_NAME + "2",
        ] = 3
        expected.loc[
            pd.Timestamp("2015-01-01") : pd.Timestamp("2015-01-20"),
            FISCAL_YEAR_FIELD_NAME + "2",
        ] = 2015

        return expected


class PreviousEstimateMultipleQuarters(WithEstimateMultipleQuarters, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def fill_expected_out(cls, expected):
        # Fill columns for 1 Q out
        for raw_name in cls.columns.values():
            expected[raw_name + "1"].loc[
                pd.Timestamp(
                    "2015-01-12",
                ) : pd.Timestamp("2015-01-19")
            ] = cls.events[raw_name].iloc[0]
            expected[raw_name + "1"].loc[pd.Timestamp("2015-01-20") :] = cls.events[
                raw_name
            ].iloc[1]

        # Fill columns for 2 Q out
        for col_name in ["estimate", "event_date"]:
            expected[col_name + "2"].loc[pd.Timestamp("2015-01-20") :] = cls.events[
                col_name
            ].iloc[0]
        expected[FISCAL_QUARTER_FIELD_NAME + "2"].loc[
            pd.Timestamp("2015-01-12") : pd.Timestamp("2015-01-20")
        ] = 4
        expected[FISCAL_YEAR_FIELD_NAME + "2"].loc[
            pd.Timestamp("2015-01-12") : pd.Timestamp("2015-01-20")
        ] = 2014
        expected[FISCAL_QUARTER_FIELD_NAME + "2"].loc[pd.Timestamp("2015-01-20") :] = 1
        expected[FISCAL_YEAR_FIELD_NAME + "2"].loc[pd.Timestamp("2015-01-20") :] = 2015
        return expected


class WithVaryingNumEstimates(WithEstimates):
    """ZiplineTestCase mixin providing fixtures and a test to ensure that we
    have the correct overwrites when the event date changes. We want to make
    sure that if we have a quarter with an event date that gets pushed back,
    we don't start overwriting for the next quarter early. Likewise,
    if we have a quarter with an event date that gets pushed forward, we want
    to make sure that we start applying adjustments at the appropriate, earlier
    date, rather than the later date.

    Methods
    -------
    assert_compute()
        Defines how to determine that results computed for the `SomeFactor`
        factor are correct.

    Tests
    -----
    test_windows_with_varying_num_estimates()
        Tests that we create the correct overwrites from 2015-01-13 to
        2015-01-14 regardless of how event dates were updated for each
        quarter for each sid.
    """

    @classmethod
    def make_events(cls):
        return pd.DataFrame(
            {
                SID_FIELD_NAME: [0] * 3 + [1] * 3,
                TS_FIELD_NAME: [
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-12"),
                    pd.Timestamp("2015-01-13"),
                ]
                * 2,
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-12"),
                    pd.Timestamp("2015-01-13"),
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-01-13"),
                    pd.Timestamp("2015-01-12"),
                    pd.Timestamp("2015-01-20"),
                ],
                "estimate": [11.0, 12.0, 21.0] * 2,
                FISCAL_QUARTER_FIELD_NAME: [1, 1, 2] * 2,
                FISCAL_YEAR_FIELD_NAME: [2015] * 6,
            }
        )

    @classmethod
    def assert_compute(cls, estimate, today):
        raise NotImplementedError("assert_compute")

    def test_windows_with_varying_num_estimates(self):
        dataset = QuartersEstimates(1)
        assert_compute = self.assert_compute

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate]
            window_length = 3

            def compute(self, today, assets, out, estimate):
                assert_compute(estimate, today)

        engine = self.make_engine()
        engine.run_pipeline(
            Pipeline({"est": SomeFactor()}),
            start_date=pd.Timestamp("2015-01-13"),
            # last event date we have
            end_date=pd.Timestamp("2015-01-14"),
        )


class PreviousVaryingNumEstimates(WithVaryingNumEstimates, ZiplineTestCase):
    def assert_compute(self, estimate, today):
        if today == pd.Timestamp("2015-01-13"):
            assert_array_equal(estimate[:, 0], np.array([np.NaN, np.NaN, 12]))
            assert_array_equal(estimate[:, 1], np.array([np.NaN, 12, 12]))
        else:
            assert_array_equal(estimate[:, 0], np.array([np.NaN, 12, 12]))
            assert_array_equal(estimate[:, 1], np.array([12, 12, 12]))

    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)


class NextVaryingNumEstimates(WithVaryingNumEstimates, ZiplineTestCase):
    def assert_compute(self, estimate, today):
        if today == pd.Timestamp("2015-01-13"):
            assert_array_equal(estimate[:, 0], np.array([11, 12, 12]))
            assert_array_equal(estimate[:, 1], np.array([np.NaN, np.NaN, 21]))
        else:
            assert_array_equal(estimate[:, 0], np.array([np.NaN, 21, 21]))
            assert_array_equal(estimate[:, 1], np.array([np.NaN, 21, 21]))

    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)


class WithEstimateWindows(WithEstimates):
    """ZiplineTestCase mixin providing fixures and a test to test running a
    Pipeline with an estimates loader over differently-sized windows.

    Attributes
    ----------
    events : pd.DataFrame
        DataFrame with estimates for 2 quarters for 2 sids.
    window_test_start_date : pd.Timestamp
        The date from which the window should start.
    timelines : dict[int -> pd.DataFrame]
        A dictionary mapping to the number of quarters out to
        snapshots of how the data should look on each date in the date range.

    Methods
    -------
    make_expected_timelines() -> dict[int -> pd.DataFrame]
        Creates a dictionary of expected data. See `timelines`, above.

    Tests
    -----
    test_estimate_windows_at_quarter_boundaries()
        Tests that we overwrite values with the correct quarter's estimate at
        the correct dates when we have a factor that asks for a window of data.
    """

    END_DATE = pd.Timestamp("2015-02-10")
    window_test_start_date = pd.Timestamp("2015-01-05")
    critical_dates = [
        pd.Timestamp("2015-01-09"),
        pd.Timestamp("2015-01-15"),
        pd.Timestamp("2015-01-20"),
        pd.Timestamp("2015-01-26"),
        pd.Timestamp("2015-02-05"),
        pd.Timestamp("2015-02-10"),
    ]
    # Starting date, number of announcements out.
    window_test_cases = list(itertools.product(critical_dates, (1, 2)))

    @classmethod
    def make_events(cls):
        # Typical case: 2 consecutive quarters.
        sid_0_timeline = pd.DataFrame(
            {
                TS_FIELD_NAME: [
                    cls.window_test_start_date,
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-01-12"),
                    pd.Timestamp("2015-02-10"),
                    # We want a case where we get info for a later
                    # quarter before the current quarter is over but
                    # after the split_asof_date to make sure that
                    # we choose the correct date to overwrite until.
                    pd.Timestamp("2015-01-18"),
                ],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-02-10"),
                    pd.Timestamp("2015-02-10"),
                    pd.Timestamp("2015-04-01"),
                ],
                "estimate": [100.0, 101.0] + [200.0, 201.0] + [400],
                FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2 + [4],
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 0,
            }
        )

        # We want a case where we skip a quarter. We never find out about Q2.
        sid_10_timeline = pd.DataFrame(
            {
                TS_FIELD_NAME: [
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-12"),
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-15"),
                ],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-22"),
                    pd.Timestamp("2015-01-22"),
                    pd.Timestamp("2015-02-05"),
                    pd.Timestamp("2015-02-05"),
                ],
                "estimate": [110.0, 111.0] + [310.0, 311.0],
                FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [3] * 2,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 10,
            }
        )

        # We want to make sure we have correct overwrites when sid quarter
        # boundaries collide. This sid's quarter boundaries collide with sid 0.
        sid_20_timeline = pd.DataFrame(
            {
                TS_FIELD_NAME: [
                    cls.window_test_start_date,
                    pd.Timestamp("2015-01-07"),
                    cls.window_test_start_date,
                    pd.Timestamp("2015-01-17"),
                ],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-02-10"),
                    pd.Timestamp("2015-02-10"),
                ],
                "estimate": [120.0, 121.0] + [220.0, 221.0],
                FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 20,
            }
        )
        concatted = pd.concat(
            [sid_0_timeline, sid_10_timeline, sid_20_timeline]
        ).reset_index()
        np.random.seed(0)
        return concatted.reindex(np.random.permutation(concatted.index))

    @classmethod
    def get_sids(cls):
        sids = sorted(cls.events[SID_FIELD_NAME].unique())
        # Add extra sids between sids in our data. We want to test that we
        # apply adjustments to the correct sids.
        return [
            sid for i in range(len(sids) - 1) for sid in range(sids[i], sids[i + 1])
        ] + [sids[-1]]

    @classmethod
    def make_expected_timelines(cls):
        return {}

    @classmethod
    def init_class_fixtures(cls):
        super(WithEstimateWindows, cls).init_class_fixtures()
        cls.create_expected_df_for_factor_compute = partial(
            create_expected_df_for_factor_compute,
            cls.window_test_start_date,
            cls.get_sids(),
        )
        cls.timelines = cls.make_expected_timelines()

    @parameterized.expand(window_test_cases)
    def test_estimate_windows_at_quarter_boundaries(
        self, start_date, num_announcements_out
    ):
        dataset = QuartersEstimates(num_announcements_out)
        trading_days = self.trading_days
        timelines = self.timelines
        # The window length should be from the starting index back to the first
        # date on which we got data. The goal is to ensure that as we
        # progress through the timeline, all data we got, starting from that
        # first date, is correctly overwritten.
        window_len = (
            self.trading_days.get_loc(start_date)
            - self.trading_days.get_loc(self.window_test_start_date)
            + 1
        )

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate]
            window_length = window_len

            def compute(self, today, assets, out, estimate):
                today_idx = trading_days.get_loc(today)
                today_timeline = (
                    timelines[num_announcements_out]
                    .loc[today]
                    .reindex(trading_days[: today_idx + 1])
                    .values
                )
                timeline_start_idx = len(today_timeline) - window_len
                assert_almost_equal(estimate, today_timeline[timeline_start_idx:])

        engine = self.make_engine()
        engine.run_pipeline(
            Pipeline({"est": SomeFactor()}),
            start_date=start_date,
            # last event date we have
            end_date=pd.Timestamp("2015-02-10"),
        )


class PreviousEstimateWindows(WithEstimateWindows, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_timelines(cls):
        oneq_previous = pd.concat(
            [
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, np.NaN, cls.window_test_start_date),
                                (10, np.NaN, cls.window_test_start_date),
                                (20, np.NaN, cls.window_test_start_date),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-09", "2015-01-19")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 101, pd.Timestamp("2015-01-20")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 121, pd.Timestamp("2015-01-20")),
                    ],
                    pd.Timestamp("2015-01-20"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 101, pd.Timestamp("2015-01-20")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 121, pd.Timestamp("2015-01-20")),
                    ],
                    pd.Timestamp("2015-01-21"),
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 101, pd.Timestamp("2015-01-20")),
                                (10, 111, pd.Timestamp("2015-01-22")),
                                (20, 121, pd.Timestamp("2015-01-20")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-22", "2015-02-04")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 101, pd.Timestamp("2015-01-20")),
                                (10, 311, pd.Timestamp("2015-02-05")),
                                (20, 121, pd.Timestamp("2015-01-20")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-02-05", "2015-02-09")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 201, pd.Timestamp("2015-02-10")),
                        (10, 311, pd.Timestamp("2015-02-05")),
                        (20, 221, pd.Timestamp("2015-02-10")),
                    ],
                    pd.Timestamp("2015-02-10"),
                ),
            ]
        )

        twoq_previous = pd.concat(
            [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-09", "2015-02-09")
            ]
            # We never get estimates for S1 for 2Q ago because once Q3
            # becomes our previous quarter, 2Q ago would be Q2, and we have
            # no data on it.
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 101, pd.Timestamp("2015-02-10")),
                        (10, np.NaN, pd.Timestamp("2015-02-05")),
                        (20, 121, pd.Timestamp("2015-02-10")),
                    ],
                    pd.Timestamp("2015-02-10"),
                )
            ]
        )
        return {1: oneq_previous, 2: twoq_previous}


class NextEstimateWindows(WithEstimateWindows, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_timelines(cls):
        oneq_next = pd.concat(
            [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100, cls.window_test_start_date),
                        (10, 110, pd.Timestamp("2015-01-09")),
                        (20, 120, cls.window_test_start_date),
                        (20, 121, pd.Timestamp("2015-01-07")),
                    ],
                    pd.Timestamp("2015-01-09"),
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 100, cls.window_test_start_date),
                                (10, 110, pd.Timestamp("2015-01-09")),
                                (10, 111, pd.Timestamp("2015-01-12")),
                                (20, 120, cls.window_test_start_date),
                                (20, 121, pd.Timestamp("2015-01-07")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-12", "2015-01-19")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100, cls.window_test_start_date),
                        (0, 101, pd.Timestamp("2015-01-20")),
                        (10, 110, pd.Timestamp("2015-01-09")),
                        (10, 111, pd.Timestamp("2015-01-12")),
                        (20, 120, cls.window_test_start_date),
                        (20, 121, pd.Timestamp("2015-01-07")),
                    ],
                    pd.Timestamp("2015-01-20"),
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 200, pd.Timestamp("2015-01-12")),
                                (10, 110, pd.Timestamp("2015-01-09")),
                                (10, 111, pd.Timestamp("2015-01-12")),
                                (20, 220, cls.window_test_start_date),
                                (20, 221, pd.Timestamp("2015-01-17")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-21", "2015-01-22")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 200, pd.Timestamp("2015-01-12")),
                                (10, 310, pd.Timestamp("2015-01-09")),
                                (10, 311, pd.Timestamp("2015-01-15")),
                                (20, 220, cls.window_test_start_date),
                                (20, 221, pd.Timestamp("2015-01-17")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-23", "2015-02-05")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 200, pd.Timestamp("2015-01-12")),
                                (10, np.NaN, cls.window_test_start_date),
                                (20, 220, cls.window_test_start_date),
                                (20, 221, pd.Timestamp("2015-01-17")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-02-06", "2015-02-09")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200, pd.Timestamp("2015-01-12")),
                        (0, 201, pd.Timestamp("2015-02-10")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220, cls.window_test_start_date),
                        (20, 221, pd.Timestamp("2015-01-17")),
                    ],
                    pd.Timestamp("2015-02-10"),
                ),
            ]
        )

        twoq_next = pd.concat(
            [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-09", "2015-01-11")
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200, pd.Timestamp("2015-01-12")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-12", "2015-01-16")
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200, pd.Timestamp("2015-01-12")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220, cls.window_test_start_date),
                        (20, 221, pd.Timestamp("2015-01-17")),
                    ],
                    pd.Timestamp("2015-01-20"),
                )
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-21", "2015-02-10")
            ]
        )

        return {1: oneq_next, 2: twoq_next}


class WithSplitAdjustedWindows(WithEstimateWindows):
    """
    ZiplineTestCase mixin providing fixures and a test to test running a
    Pipeline with an estimates loader over differently-sized windows and with
    split adjustments.
    """

    split_adjusted_asof_date = pd.Timestamp("2015-01-14")

    @classmethod
    def make_events(cls):
        # Add an extra sid that has a release before the split-asof-date in
        # order to test that we're reversing splits correctly in the previous
        # case (without an overwrite) and in the next case (with an overwrite).
        sid_30 = pd.DataFrame(
            {
                TS_FIELD_NAME: [
                    cls.window_test_start_date,
                    pd.Timestamp("2015-01-09"),
                    # For Q2, we want it to start early enough
                    # that we can have several adjustments before
                    # the end of the first quarter so that we
                    # can test un-adjusting & readjusting with an
                    # overwrite.
                    cls.window_test_start_date,
                    # We want the Q2 event date to be enough past
                    # the split-asof-date that we can have
                    # several splits and can make sure that they
                    # are applied correctly.
                    pd.Timestamp("2015-01-20"),
                ],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-01-20"),
                ],
                "estimate": [130.0, 131.0, 230.0, 231.0],
                FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 30,
            }
        )

        # An extra sid to test no splits before the split-adjusted-asof-date.
        # We want an event before and after the split-adjusted-asof-date &
        # timestamps for data points also before and after
        # split-adjsuted-asof-date (but also before the split dates, so that
        # we can test that splits actually get applied at the correct times).
        sid_40 = pd.DataFrame(
            {
                TS_FIELD_NAME: [pd.Timestamp("2015-01-09"), pd.Timestamp("2015-01-15")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-02-10"),
                ],
                "estimate": [140.0, 240.0],
                FISCAL_QUARTER_FIELD_NAME: [1, 2],
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 40,
            }
        )

        # An extra sid to test all splits before the
        # split-adjusted-asof-date. All timestamps should be before that date
        # so that we have cases where we un-apply and re-apply splits.
        sid_50 = pd.DataFrame(
            {
                TS_FIELD_NAME: [pd.Timestamp("2015-01-09"), pd.Timestamp("2015-01-12")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-02-10"),
                ],
                "estimate": [150.0, 250.0],
                FISCAL_QUARTER_FIELD_NAME: [1, 2],
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 50,
            }
        )

        return pd.concat(
            [
                # Slightly hacky, but want to make sure we're using the same
                # events as WithEstimateWindows.
                cls.__base__.make_events(),
                sid_30,
                sid_40,
                sid_50,
            ]
        )

    @classmethod
    def make_splits_data(cls):
        # For sid 0, we want to apply a series of splits before and after the
        #  split-adjusted-asof-date we well as between quarters (for the
        # previous case, where we won't see any values until after the event
        # happens).
        sid_0_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 0,
                "ratio": (-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100),
                "effective_date": (
                    pd.Timestamp("2014-01-01"),  # Filter out
                    # Split before Q1 event & after first estimate
                    pd.Timestamp("2015-01-07"),
                    # Split before Q1 event
                    pd.Timestamp("2015-01-09"),
                    # Split before Q1 event
                    pd.Timestamp("2015-01-13"),
                    # Split before Q1 event
                    pd.Timestamp("2015-01-15"),
                    # Split before Q1 event
                    pd.Timestamp("2015-01-18"),
                    # Split after Q1 event and before Q2 event
                    pd.Timestamp("2015-01-30"),
                    # Filter out - this is after our date index
                    pd.Timestamp("2016-01-01"),
                ),
            }
        )

        sid_10_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 10,
                "ratio": (0.2, 0.3),
                "effective_date": (
                    # We want a split before the first estimate and before the
                    # split-adjusted-asof-date but within our calendar index so
                    # that we can test that the split is NEVER applied.
                    pd.Timestamp("2015-01-07"),
                    # Apply a single split before Q1 event.
                    pd.Timestamp("2015-01-20"),
                ),
            }
        )

        # We want a sid with split dates that collide with another sid (0) to
        # make sure splits are correctly applied for both sids.
        sid_20_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 20,
                "ratio": (
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ),
                "effective_date": (
                    pd.Timestamp("2015-01-07"),
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-13"),
                    pd.Timestamp("2015-01-15"),
                    pd.Timestamp("2015-01-18"),
                    pd.Timestamp("2015-01-30"),
                ),
            }
        )

        # This sid has event dates that are shifted back so that we can test
        # cases where an event occurs before the split-asof-date.
        sid_30_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 30,
                "ratio": (8, 9, 10, 11, 12),
                "effective_date": (
                    # Split before the event and before the
                    # split-asof-date.
                    pd.Timestamp("2015-01-07"),
                    # Split on date of event but before the
                    # split-asof-date.
                    pd.Timestamp("2015-01-09"),
                    # Split after the event, but before the
                    # split-asof-date.
                    pd.Timestamp("2015-01-13"),
                    pd.Timestamp("2015-01-15"),
                    pd.Timestamp("2015-01-18"),
                ),
            }
        )

        # No splits for a sid before the split-adjusted-asof-date.
        sid_40_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 40,
                "ratio": (13, 14),
                "effective_date": (
                    pd.Timestamp("2015-01-20"),
                    pd.Timestamp("2015-01-22"),
                ),
            }
        )

        # No splits for a sid after the split-adjusted-asof-date.
        sid_50_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 50,
                "ratio": (15, 16),
                "effective_date": (
                    pd.Timestamp("2015-01-13"),
                    pd.Timestamp("2015-01-14"),
                ),
            }
        )

        return pd.concat(
            [
                sid_0_splits,
                sid_10_splits,
                sid_20_splits,
                sid_30_splits,
                sid_40_splits,
                sid_50_splits,
            ]
        )


class PreviousWithSplitAdjustedWindows(WithSplitAdjustedWindows, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousSplitAdjustedEarningsEstimatesLoader(
            events,
            columns,
            split_adjustments_loader=cls.adjustment_reader,
            split_adjusted_column_names=["estimate"],
            split_adjusted_asof=cls.split_adjusted_asof_date,
        )

    @classmethod
    def make_expected_timelines(cls):
        oneq_previous = pd.concat(
            [
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, np.NaN, cls.window_test_start_date),
                                (10, np.NaN, cls.window_test_start_date),
                                (20, np.NaN, cls.window_test_start_date),
                                # Undo all adjustments that haven't happened yet.
                                (30, 131 * 1 / 10, pd.Timestamp("2015-01-09")),
                                (40, 140.0, pd.Timestamp("2015-01-09")),
                                (50, 150 * 1 / 15 * 1 / 16, pd.Timestamp("2015-01-09")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-09", "2015-01-12")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                        (30, 131, pd.Timestamp("2015-01-09")),
                        (40, 140.0, pd.Timestamp("2015-01-09")),
                        (50, 150.0 * 1 / 16, pd.Timestamp("2015-01-09")),
                    ],
                    pd.Timestamp("2015-01-13"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                        (30, 131, pd.Timestamp("2015-01-09")),
                        (40, 140.0, pd.Timestamp("2015-01-09")),
                        (50, 150.0, pd.Timestamp("2015-01-09")),
                    ],
                    pd.Timestamp("2015-01-14"),
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, np.NaN, cls.window_test_start_date),
                                (10, np.NaN, cls.window_test_start_date),
                                (20, np.NaN, cls.window_test_start_date),
                                (30, 131 * 11, pd.Timestamp("2015-01-09")),
                                (40, 140.0, pd.Timestamp("2015-01-09")),
                                (50, 150.0, pd.Timestamp("2015-01-09")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-15", "2015-01-16")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 101, pd.Timestamp("2015-01-20")),
                                (10, np.NaN, cls.window_test_start_date),
                                (20, 121 * 0.7 * 0.8, pd.Timestamp("2015-01-20")),
                                (30, 231, pd.Timestamp("2015-01-20")),
                                (40, 140.0 * 13, pd.Timestamp("2015-01-09")),
                                (50, 150.0, pd.Timestamp("2015-01-09")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-20", "2015-01-21")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 101, pd.Timestamp("2015-01-20")),
                                (10, 111 * 0.3, pd.Timestamp("2015-01-22")),
                                (20, 121 * 0.7 * 0.8, pd.Timestamp("2015-01-20")),
                                (30, 231, pd.Timestamp("2015-01-20")),
                                (40, 140.0 * 13 * 14, pd.Timestamp("2015-01-09")),
                                (50, 150.0, pd.Timestamp("2015-01-09")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-22", "2015-01-29")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 101 * 7, pd.Timestamp("2015-01-20")),
                                (10, 111 * 0.3, pd.Timestamp("2015-01-22")),
                                (20, 121 * 0.7 * 0.8 * 0.9, pd.Timestamp("2015-01-20")),
                                (30, 231, pd.Timestamp("2015-01-20")),
                                (40, 140.0 * 13 * 14, pd.Timestamp("2015-01-09")),
                                (50, 150.0, pd.Timestamp("2015-01-09")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-30", "2015-02-04")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 101 * 7, pd.Timestamp("2015-01-20")),
                                (10, 311 * 0.3, pd.Timestamp("2015-02-05")),
                                (20, 121 * 0.7 * 0.8 * 0.9, pd.Timestamp("2015-01-20")),
                                (30, 231, pd.Timestamp("2015-01-20")),
                                (40, 140.0 * 13 * 14, pd.Timestamp("2015-01-09")),
                                (50, 150.0, pd.Timestamp("2015-01-09")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-02-05", "2015-02-09")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 201, pd.Timestamp("2015-02-10")),
                        (10, 311 * 0.3, pd.Timestamp("2015-02-05")),
                        (20, 221 * 0.8 * 0.9, pd.Timestamp("2015-02-10")),
                        (30, 231, pd.Timestamp("2015-01-20")),
                        (40, 240.0 * 13 * 14, pd.Timestamp("2015-02-10")),
                        (50, 250.0, pd.Timestamp("2015-02-10")),
                    ],
                    pd.Timestamp("2015-02-10"),
                ),
            ]
        )

        twoq_previous = pd.concat(
            [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                        (30, np.NaN, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-09", "2015-01-19")
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                        (30, 131 * 11 * 12, pd.Timestamp("2015-01-20")),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-20", "2015-02-09")
            ]
            # We never get estimates for S1 for 2Q ago because once Q3
            # becomes our previous quarter, 2Q ago would be Q2, and we have
            # no data on it.
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 101 * 7, pd.Timestamp("2015-02-10")),
                        (10, np.NaN, pd.Timestamp("2015-02-05")),
                        (20, 121 * 0.7 * 0.8 * 0.9, pd.Timestamp("2015-02-10")),
                        (30, 131 * 11 * 12, pd.Timestamp("2015-01-20")),
                        (40, 140.0 * 13 * 14, pd.Timestamp("2015-02-10")),
                        (50, 150.0, pd.Timestamp("2015-02-10")),
                    ],
                    pd.Timestamp("2015-02-10"),
                )
            ]
        )
        return {1: oneq_previous, 2: twoq_previous}


class NextWithSplitAdjustedWindows(WithSplitAdjustedWindows, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextSplitAdjustedEarningsEstimatesLoader(
            events,
            columns,
            split_adjustments_loader=cls.adjustment_reader,
            split_adjusted_column_names=["estimate"],
            split_adjusted_asof=cls.split_adjusted_asof_date,
        )

    @classmethod
    def make_expected_timelines(cls):
        oneq_next = pd.concat(
            [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100 * 1 / 4, cls.window_test_start_date),
                        (10, 110, pd.Timestamp("2015-01-09")),
                        (20, 120 * 5 / 3, cls.window_test_start_date),
                        (20, 121 * 5 / 3, pd.Timestamp("2015-01-07")),
                        (30, 130 * 1 / 10, cls.window_test_start_date),
                        (30, 131 * 1 / 10, pd.Timestamp("2015-01-09")),
                        (40, 140, pd.Timestamp("2015-01-09")),
                        (50, 150.0 * 1 / 15 * 1 / 16, pd.Timestamp("2015-01-09")),
                    ],
                    pd.Timestamp("2015-01-09"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100 * 1 / 4, cls.window_test_start_date),
                        (10, 110, pd.Timestamp("2015-01-09")),
                        (10, 111, pd.Timestamp("2015-01-12")),
                        (20, 120 * 5 / 3, cls.window_test_start_date),
                        (20, 121 * 5 / 3, pd.Timestamp("2015-01-07")),
                        (30, 230 * 1 / 10, cls.window_test_start_date),
                        (40, np.NaN, pd.Timestamp("2015-01-10")),
                        (50, 250.0 * 1 / 15 * 1 / 16, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-01-12"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100, cls.window_test_start_date),
                        (10, 110, pd.Timestamp("2015-01-09")),
                        (10, 111, pd.Timestamp("2015-01-12")),
                        (20, 120, cls.window_test_start_date),
                        (20, 121, pd.Timestamp("2015-01-07")),
                        (30, 230, cls.window_test_start_date),
                        (40, np.NaN, pd.Timestamp("2015-01-10")),
                        (50, 250.0 * 1 / 16, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-01-13"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100, cls.window_test_start_date),
                        (10, 110, pd.Timestamp("2015-01-09")),
                        (10, 111, pd.Timestamp("2015-01-12")),
                        (20, 120, cls.window_test_start_date),
                        (20, 121, pd.Timestamp("2015-01-07")),
                        (30, 230, cls.window_test_start_date),
                        (40, np.NaN, pd.Timestamp("2015-01-10")),
                        (50, 250.0, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-01-14"),
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 100 * 5, cls.window_test_start_date),
                                (10, 110, pd.Timestamp("2015-01-09")),
                                (10, 111, pd.Timestamp("2015-01-12")),
                                (20, 120 * 0.7, cls.window_test_start_date),
                                (20, 121 * 0.7, pd.Timestamp("2015-01-07")),
                                (30, 230 * 11, cls.window_test_start_date),
                                (40, 240, pd.Timestamp("2015-01-15")),
                                (50, 250.0, pd.Timestamp("2015-01-12")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-15", "2015-01-16")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 100 * 5 * 6, cls.window_test_start_date),
                        (0, 101, pd.Timestamp("2015-01-20")),
                        (10, 110 * 0.3, pd.Timestamp("2015-01-09")),
                        (10, 111 * 0.3, pd.Timestamp("2015-01-12")),
                        (20, 120 * 0.7 * 0.8, cls.window_test_start_date),
                        (20, 121 * 0.7 * 0.8, pd.Timestamp("2015-01-07")),
                        (30, 230 * 11 * 12, cls.window_test_start_date),
                        (30, 231, pd.Timestamp("2015-01-20")),
                        (40, 240 * 13, pd.Timestamp("2015-01-15")),
                        (50, 250.0, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-01-20"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200 * 5 * 6, pd.Timestamp("2015-01-12")),
                        (10, 110 * 0.3, pd.Timestamp("2015-01-09")),
                        (10, 111 * 0.3, pd.Timestamp("2015-01-12")),
                        (20, 220 * 0.7 * 0.8, cls.window_test_start_date),
                        (20, 221 * 0.8, pd.Timestamp("2015-01-17")),
                        (40, 240 * 13, pd.Timestamp("2015-01-15")),
                        (50, 250.0, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-01-21"),
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200 * 5 * 6, pd.Timestamp("2015-01-12")),
                        (10, 110 * 0.3, pd.Timestamp("2015-01-09")),
                        (10, 111 * 0.3, pd.Timestamp("2015-01-12")),
                        (20, 220 * 0.7 * 0.8, cls.window_test_start_date),
                        (20, 221 * 0.8, pd.Timestamp("2015-01-17")),
                        (40, 240 * 13 * 14, pd.Timestamp("2015-01-15")),
                        (50, 250.0, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-01-22"),
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 200 * 5 * 6, pd.Timestamp("2015-01-12")),
                                (10, 310 * 0.3, pd.Timestamp("2015-01-09")),
                                (10, 311 * 0.3, pd.Timestamp("2015-01-15")),
                                (20, 220 * 0.7 * 0.8, cls.window_test_start_date),
                                (20, 221 * 0.8, pd.Timestamp("2015-01-17")),
                                (40, 240 * 13 * 14, pd.Timestamp("2015-01-15")),
                                (50, 250.0, pd.Timestamp("2015-01-12")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-23", "2015-01-29")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 200 * 5 * 6 * 7, pd.Timestamp("2015-01-12")),
                                (10, 310 * 0.3, pd.Timestamp("2015-01-09")),
                                (10, 311 * 0.3, pd.Timestamp("2015-01-15")),
                                (20, 220 * 0.7 * 0.8 * 0.9, cls.window_test_start_date),
                                (20, 221 * 0.8 * 0.9, pd.Timestamp("2015-01-17")),
                                (40, 240 * 13 * 14, pd.Timestamp("2015-01-15")),
                                (50, 250.0, pd.Timestamp("2015-01-12")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-01-30", "2015-02-05")
                    ]
                ),
                pd.concat(
                    [
                        cls.create_expected_df_for_factor_compute(
                            [
                                (0, 200 * 5 * 6 * 7, pd.Timestamp("2015-01-12")),
                                (10, np.NaN, cls.window_test_start_date),
                                (20, 220 * 0.7 * 0.8 * 0.9, cls.window_test_start_date),
                                (20, 221 * 0.8 * 0.9, pd.Timestamp("2015-01-17")),
                                (40, 240 * 13 * 14, pd.Timestamp("2015-01-15")),
                                (50, 250.0, pd.Timestamp("2015-01-12")),
                            ],
                            end_date,
                        )
                        for end_date in pd.date_range("2015-02-06", "2015-02-09")
                    ]
                ),
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200 * 5 * 6 * 7, pd.Timestamp("2015-01-12")),
                        (0, 201, pd.Timestamp("2015-02-10")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220 * 0.7 * 0.8 * 0.9, cls.window_test_start_date),
                        (20, 221 * 0.8 * 0.9, pd.Timestamp("2015-01-17")),
                        (40, 240 * 13 * 14, pd.Timestamp("2015-01-15")),
                        (50, 250.0, pd.Timestamp("2015-01-12")),
                    ],
                    pd.Timestamp("2015-02-10"),
                ),
            ]
        )

        twoq_next = pd.concat(
            [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220 * 5 / 3, cls.window_test_start_date),
                        (30, 230 * 1 / 10, cls.window_test_start_date),
                        (40, np.NaN, cls.window_test_start_date),
                        (50, np.NaN, cls.window_test_start_date),
                    ],
                    pd.Timestamp("2015-01-09"),
                )
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200 * 1 / 4, pd.Timestamp("2015-01-12")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220 * 5 / 3, cls.window_test_start_date),
                        (30, np.NaN, cls.window_test_start_date),
                        (40, np.NaN, cls.window_test_start_date),
                    ],
                    pd.Timestamp("2015-01-12"),
                )
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200, pd.Timestamp("2015-01-12")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220, cls.window_test_start_date),
                        (30, np.NaN, cls.window_test_start_date),
                        (40, np.NaN, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-13", "2015-01-14")
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200 * 5, pd.Timestamp("2015-01-12")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220 * 0.7, cls.window_test_start_date),
                        (30, np.NaN, cls.window_test_start_date),
                        (40, np.NaN, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-15", "2015-01-16")
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, 200 * 5 * 6, pd.Timestamp("2015-01-12")),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, 220 * 0.7 * 0.8, cls.window_test_start_date),
                        (20, 221 * 0.8, pd.Timestamp("2015-01-17")),
                        (30, np.NaN, cls.window_test_start_date),
                        (40, np.NaN, cls.window_test_start_date),
                    ],
                    pd.Timestamp("2015-01-20"),
                )
            ]
            + [
                cls.create_expected_df_for_factor_compute(
                    [
                        (0, np.NaN, cls.window_test_start_date),
                        (10, np.NaN, cls.window_test_start_date),
                        (20, np.NaN, cls.window_test_start_date),
                        (30, np.NaN, cls.window_test_start_date),
                        (40, np.NaN, cls.window_test_start_date),
                    ],
                    end_date,
                )
                for end_date in pd.date_range("2015-01-21", "2015-02-10")
            ]
        )

        return {1: oneq_next, 2: twoq_next}


class WithSplitAdjustedMultipleEstimateColumns(WithEstimates):
    """
    ZiplineTestCase mixin for having multiple estimate columns that are
    split-adjusted to make sure that adjustments are applied correctly.

    Attributes
    ----------
    test_start_date : pd.Timestamp
        The start date of the test.
    test_end_date : pd.Timestamp
        The start date of the test.
    split_adjusted_asof : pd.Timestamp
        The split-adjusted-asof-date of the data used in the test, to be used
        to create all loaders of test classes that subclass this mixin.

    Methods
    -------
    make_expected_timelines_1q_out -> dict[pd.Timestamp -> dict[str ->
        np.array]]
        The expected array of results for each date of the date range for
        each column. Only for 1 quarter out.

    make_expected_timelines_2q_out -> dict[pd.Timestamp -> dict[str ->
        np.array]]
        The expected array of results for each date of the date range. For 2
        quarters out, so only for the column that is requested to be loaded
        with 2 quarters out.

    Tests
    -----
    test_adjustments_with_multiple_adjusted_columns
        Tests that if you have multiple columns, we still split-adjust
        correctly.

    test_multiple_datasets_different_num_announcements
        Tests that if you have multiple datasets that ask for a different
        number of quarters out, and each asks for a different estimates column,
        we still split-adjust correctly.
    """

    END_DATE = pd.Timestamp("2015-02-10")
    test_start_date = pd.Timestamp("2015-01-06")
    test_end_date = pd.Timestamp("2015-01-12")
    split_adjusted_asof = pd.Timestamp("2015-01-08")

    @classmethod
    def make_columns(cls):
        return {
            MultipleColumnsEstimates.event_date: "event_date",
            MultipleColumnsEstimates.fiscal_quarter: "fiscal_quarter",
            MultipleColumnsEstimates.fiscal_year: "fiscal_year",
            MultipleColumnsEstimates.estimate1: "estimate1",
            MultipleColumnsEstimates.estimate2: "estimate2",
        }

    @classmethod
    def make_events(cls):
        sid_0_events = pd.DataFrame(
            {
                # We only want a stale KD here so that adjustments
                # will be applied.
                TS_FIELD_NAME: [pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-05")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-09"),
                    pd.Timestamp("2015-01-12"),
                ],
                "estimate1": [1100.0, 1200.0],
                "estimate2": [2100.0, 2200.0],
                FISCAL_QUARTER_FIELD_NAME: [1, 2],
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 0,
            }
        )

        # This is just an extra sid to make sure that we apply adjustments
        # correctly for multiple columns when we have multiple sids.
        sid_1_events = pd.DataFrame(
            {
                # We only want a stale KD here so that adjustments
                # will be applied.
                TS_FIELD_NAME: [pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-05")],
                EVENT_DATE_FIELD_NAME: [
                    pd.Timestamp("2015-01-08"),
                    pd.Timestamp("2015-01-11"),
                ],
                "estimate1": [1110.0, 1210.0],
                "estimate2": [2110.0, 2210.0],
                FISCAL_QUARTER_FIELD_NAME: [1, 2],
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 1,
            }
        )
        return pd.concat([sid_0_events, sid_1_events])

    @classmethod
    def make_splits_data(cls):
        sid_0_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 0,
                "ratio": (0.3, 3.0),
                "effective_date": (
                    pd.Timestamp("2015-01-07"),
                    pd.Timestamp("2015-01-09"),
                ),
            }
        )

        sid_1_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 1,
                "ratio": (0.4, 4.0),
                "effective_date": (
                    pd.Timestamp("2015-01-07"),
                    pd.Timestamp("2015-01-09"),
                ),
            }
        )

        return pd.concat([sid_0_splits, sid_1_splits])

    @classmethod
    def make_expected_timelines_1q_out(cls):
        return {}

    @classmethod
    def make_expected_timelines_2q_out(cls):
        return {}

    @classmethod
    def init_class_fixtures(cls):
        super(WithSplitAdjustedMultipleEstimateColumns, cls).init_class_fixtures()
        cls.timelines_1q_out = cls.make_expected_timelines_1q_out()
        cls.timelines_2q_out = cls.make_expected_timelines_2q_out()

    def test_adjustments_with_multiple_adjusted_columns(self):
        dataset = MultipleColumnsQuartersEstimates(1)
        timelines = self.timelines_1q_out
        window_len = 3

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate1, dataset.estimate2]
            window_length = window_len

            def compute(self, today, assets, out, estimate1, estimate2):
                assert_almost_equal(estimate1, timelines[today]["estimate1"])
                assert_almost_equal(estimate2, timelines[today]["estimate2"])

        engine = self.make_engine()
        engine.run_pipeline(
            Pipeline({"est": SomeFactor()}),
            start_date=self.test_start_date,
            # last event date we have
            end_date=self.test_end_date,
        )

    def test_multiple_datasets_different_num_announcements(self):
        dataset1 = MultipleColumnsQuartersEstimates(1)
        dataset2 = MultipleColumnsQuartersEstimates(2)
        timelines_1q_out = self.timelines_1q_out
        timelines_2q_out = self.timelines_2q_out
        window_len = 3

        class SomeFactor1(CustomFactor):
            inputs = [dataset1.estimate1]
            window_length = window_len

            def compute(self, today, assets, out, estimate1):
                assert_almost_equal(estimate1, timelines_1q_out[today]["estimate1"])

        class SomeFactor2(CustomFactor):
            inputs = [dataset2.estimate2]
            window_length = window_len

            def compute(self, today, assets, out, estimate2):
                assert_almost_equal(estimate2, timelines_2q_out[today]["estimate2"])

        engine = self.make_engine()
        engine.run_pipeline(
            Pipeline({"est1": SomeFactor1(), "est2": SomeFactor2()}),
            start_date=self.test_start_date,
            # last event date we have
            end_date=self.test_end_date,
        )


class PreviousWithSplitAdjustedMultipleEstimateColumns(
    WithSplitAdjustedMultipleEstimateColumns, ZiplineTestCase
):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousSplitAdjustedEarningsEstimatesLoader(
            events,
            columns,
            split_adjustments_loader=cls.adjustment_reader,
            split_adjusted_column_names=["estimate1", "estimate2"],
            split_adjusted_asof=cls.split_adjusted_asof,
        )

    @classmethod
    def make_expected_timelines_1q_out(cls):
        return {
            pd.Timestamp("2015-01-06"): {
                "estimate1": np.array([[np.NaN, np.NaN]] * 3),
                "estimate2": np.array([[np.NaN, np.NaN]] * 3),
            },
            pd.Timestamp("2015-01-07"): {
                "estimate1": np.array([[np.NaN, np.NaN]] * 3),
                "estimate2": np.array([[np.NaN, np.NaN]] * 3),
            },
            pd.Timestamp("2015-01-08"): {
                "estimate1": np.array([[np.NaN, np.NaN]] * 2 + [[np.NaN, 1110.0]]),
                "estimate2": np.array([[np.NaN, np.NaN]] * 2 + [[np.NaN, 2110.0]]),
            },
            pd.Timestamp("2015-01-09"): {
                "estimate1": np.array(
                    [[np.NaN, np.NaN]]
                    + [[np.NaN, 1110.0 * 4]]
                    + [[1100 * 3.0, 1110.0 * 4]]
                ),
                "estimate2": np.array(
                    [[np.NaN, np.NaN]]
                    + [[np.NaN, 2110.0 * 4]]
                    + [[2100 * 3.0, 2110.0 * 4]]
                ),
            },
            pd.Timestamp("2015-01-12"): {
                "estimate1": np.array(
                    [[np.NaN, np.NaN]] * 2 + [[1200 * 3.0, 1210.0 * 4]]
                ),
                "estimate2": np.array(
                    [[np.NaN, np.NaN]] * 2 + [[2200 * 3.0, 2210.0 * 4]]
                ),
            },
        }

    @classmethod
    def make_expected_timelines_2q_out(cls):
        return {
            pd.Timestamp("2015-01-06"): {"estimate2": np.array([[np.NaN, np.NaN]] * 3)},
            pd.Timestamp("2015-01-07"): {"estimate2": np.array([[np.NaN, np.NaN]] * 3)},
            pd.Timestamp("2015-01-08"): {"estimate2": np.array([[np.NaN, np.NaN]] * 3)},
            pd.Timestamp("2015-01-09"): {"estimate2": np.array([[np.NaN, np.NaN]] * 3)},
            pd.Timestamp("2015-01-12"): {
                "estimate2": np.array(
                    [[np.NaN, np.NaN]] * 2 + [[2100 * 3.0, 2110.0 * 4]]
                )
            },
        }


class NextWithSplitAdjustedMultipleEstimateColumns(
    WithSplitAdjustedMultipleEstimateColumns, ZiplineTestCase
):
    @classmethod
    def make_loader(cls, events, columns):
        return NextSplitAdjustedEarningsEstimatesLoader(
            events,
            columns,
            split_adjustments_loader=cls.adjustment_reader,
            split_adjusted_column_names=["estimate1", "estimate2"],
            split_adjusted_asof=cls.split_adjusted_asof,
        )

    @classmethod
    def make_expected_timelines_1q_out(cls):
        return {
            pd.Timestamp("2015-01-06"): {
                "estimate1": np.array(
                    [[np.NaN, np.NaN]] + [[1100.0 * 1 / 0.3, 1110.0 * 1 / 0.4]] * 2
                ),
                "estimate2": np.array(
                    [[np.NaN, np.NaN]] + [[2100.0 * 1 / 0.3, 2110.0 * 1 / 0.4]] * 2
                ),
            },
            pd.Timestamp("2015-01-07"): {
                "estimate1": np.array([[1100.0, 1110.0]] * 3),
                "estimate2": np.array([[2100.0, 2110.0]] * 3),
            },
            pd.Timestamp("2015-01-08"): {
                "estimate1": np.array([[1100.0, 1110.0]] * 3),
                "estimate2": np.array([[2100.0, 2110.0]] * 3),
            },
            pd.Timestamp("2015-01-09"): {
                "estimate1": np.array([[1100 * 3.0, 1210.0 * 4]] * 3),
                "estimate2": np.array([[2100 * 3.0, 2210.0 * 4]] * 3),
            },
            pd.Timestamp("2015-01-12"): {
                "estimate1": np.array([[1200 * 3.0, np.NaN]] * 3),
                "estimate2": np.array([[2200 * 3.0, np.NaN]] * 3),
            },
        }

    @classmethod
    def make_expected_timelines_2q_out(cls):
        return {
            pd.Timestamp("2015-01-06"): {
                "estimate2": np.array(
                    [[np.NaN, np.NaN]] + [[2200 * 1 / 0.3, 2210.0 * 1 / 0.4]] * 2
                )
            },
            pd.Timestamp("2015-01-07"): {"estimate2": np.array([[2200.0, 2210.0]] * 3)},
            pd.Timestamp("2015-01-08"): {"estimate2": np.array([[2200, 2210.0]] * 3)},
            pd.Timestamp("2015-01-09"): {
                "estimate2": np.array([[2200 * 3.0, np.NaN]] * 3)
            },
            pd.Timestamp("2015-01-12"): {"estimate2": np.array([[np.NaN, np.NaN]] * 3)},
        }


class WithAdjustmentBoundaries(WithEstimates):
    """ZiplineTestCase mixin providing class-level attributes, methods,
    and a test to make sure that when the split-adjusted-asof-date is not
    strictly within the date index, we can still apply adjustments correctly.

    Attributes
    ----------
    split_adjusted_before_start : pd.Timestamp
        A split-adjusted-asof-date before the start date of the test.
    split_adjusted_after_end : pd.Timestamp
        A split-adjusted-asof-date before the end date of the test.
    split_adjusted_asof_dates : list of tuples of pd.Timestamp
        All the split-adjusted-asof-dates over which we want to parameterize
        the test.

    Methods
    -------
    make_expected_out -> dict[pd.Timestamp -> pd.DataFrame]
        A dictionary of the expected output of the pipeline at each of the
        dates of interest.
    """

    START_DATE = pd.Timestamp("2015-01-04")
    # We want to run the pipeline starting from `START_DATE`, but the
    # pipeline results will start from the next day, which is
    # `test_start_date`.
    test_start_date = pd.Timestamp("2015-01-05")
    END_DATE = test_end_date = pd.Timestamp("2015-01-12")
    split_adjusted_before_start = test_start_date - timedelta(days=1)
    split_adjusted_after_end = test_end_date + timedelta(days=1)
    # Must parametrize over this because there can only be 1 such date for
    # each set of data.
    split_adjusted_asof_dates = [
        (test_start_date,),
        (test_end_date,),
        (split_adjusted_before_start,),
        (split_adjusted_after_end,),
    ]

    @classmethod
    def init_class_fixtures(cls):
        super(WithAdjustmentBoundaries, cls).init_class_fixtures()
        cls.s0 = cls.asset_finder.retrieve_asset(0)
        cls.s1 = cls.asset_finder.retrieve_asset(1)
        cls.s2 = cls.asset_finder.retrieve_asset(2)
        cls.s3 = cls.asset_finder.retrieve_asset(3)
        cls.s4 = cls.asset_finder.retrieve_asset(4)
        cls.expected = cls.make_expected_out()

    @classmethod
    def make_events(cls):
        # We can create a sid for each configuration of dates for KDs, events,
        # and splits. For this test we don't care about overwrites so we only
        # test 1 quarter.
        sid_0_timeline = pd.DataFrame(
            {
                # KD on first date of index
                TS_FIELD_NAME: cls.test_start_date,
                EVENT_DATE_FIELD_NAME: pd.Timestamp("2015-01-09"),
                "estimate": 10.0,
                FISCAL_QUARTER_FIELD_NAME: 1,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 0,
            },
            index=[0],
        )

        sid_1_timeline = pd.DataFrame(
            {
                TS_FIELD_NAME: cls.test_start_date,
                # event date on first date of index
                EVENT_DATE_FIELD_NAME: cls.test_start_date,
                "estimate": 11.0,
                FISCAL_QUARTER_FIELD_NAME: 1,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 1,
            },
            index=[0],
        )

        sid_2_timeline = pd.DataFrame(
            {
                # KD on first date of index
                TS_FIELD_NAME: cls.test_end_date,
                EVENT_DATE_FIELD_NAME: cls.test_end_date + timedelta(days=1),
                "estimate": 12.0,
                FISCAL_QUARTER_FIELD_NAME: 1,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 2,
            },
            index=[0],
        )

        sid_3_timeline = pd.DataFrame(
            {
                TS_FIELD_NAME: cls.test_end_date - timedelta(days=1),
                EVENT_DATE_FIELD_NAME: cls.test_end_date,
                "estimate": 13.0,
                FISCAL_QUARTER_FIELD_NAME: 1,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 3,
            },
            index=[0],
        )

        # KD and event date don't fall on date index boundaries
        sid_4_timeline = pd.DataFrame(
            {
                TS_FIELD_NAME: cls.test_end_date - timedelta(days=1),
                EVENT_DATE_FIELD_NAME: cls.test_end_date - timedelta(days=1),
                "estimate": 14.0,
                FISCAL_QUARTER_FIELD_NAME: 1,
                FISCAL_YEAR_FIELD_NAME: 2015,
                SID_FIELD_NAME: 4,
            },
            index=[0],
        )

        return pd.concat(
            [
                sid_0_timeline,
                sid_1_timeline,
                sid_2_timeline,
                sid_3_timeline,
                sid_4_timeline,
            ]
        )

    @classmethod
    def make_splits_data(cls):
        # Here we want splits that collide
        sid_0_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 0,
                "ratio": 0.10,
                "effective_date": cls.test_start_date,
            },
            index=[0],
        )

        sid_1_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 1,
                "ratio": 0.11,
                "effective_date": cls.test_start_date,
            },
            index=[0],
        )

        sid_2_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 2,
                "ratio": 0.12,
                "effective_date": cls.test_end_date,
            },
            index=[0],
        )

        sid_3_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 3,
                "ratio": 0.13,
                "effective_date": cls.test_end_date,
            },
            index=[0],
        )

        # We want 2 splits here - at the starting boundary and at the end
        # boundary - while there is no collision with KD/event date for the
        # sid.
        sid_4_splits = pd.DataFrame(
            {
                SID_FIELD_NAME: 4,
                "ratio": (0.14, 0.15),
                "effective_date": (cls.test_start_date, cls.test_end_date),
            }
        )

        return pd.concat(
            [sid_0_splits, sid_1_splits, sid_2_splits, sid_3_splits, sid_4_splits]
        )

    @parameterized.expand(split_adjusted_asof_dates)
    def test_boundaries(self, split_date):
        dataset = QuartersEstimates(1)
        loader = self.loader(split_adjusted_asof=split_date)
        engine = self.make_engine(loader)
        result = engine.run_pipeline(
            Pipeline({"estimate": dataset.estimate.latest}),
            start_date=self.trading_days[0],
            # last event date we have
            end_date=self.trading_days[-1],
        )
        expected = self.expected[split_date]
        assert_frame_equal(result, expected, check_names=False)

    @classmethod
    def make_expected_out(cls):
        return {}


class PreviousWithAdjustmentBoundaries(WithAdjustmentBoundaries, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return partial(
            PreviousSplitAdjustedEarningsEstimatesLoader,
            events,
            columns,
            split_adjustments_loader=cls.adjustment_reader,
            split_adjusted_column_names=["estimate"],
        )

    @classmethod
    def make_expected_out(cls):
        split_adjusted_at_start_boundary = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s0,
                            "estimate": np.NaN,
                        },
                        index=pd.date_range(
                            cls.test_start_date,
                            pd.Timestamp("2015-01-08"),
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s0,
                            "estimate": 10.0,
                        },
                        index=pd.date_range(
                            pd.Timestamp("2015-01-09"),
                            cls.test_end_date,
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s1,
                            "estimate": 11.0,
                        },
                        index=pd.date_range(cls.test_start_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s2, "estimate": np.NaN},
                        index=pd.date_range(cls.test_start_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s3, "estimate": np.NaN},
                        index=pd.date_range(
                            cls.test_start_date,
                            cls.test_end_date - timedelta(1),
                        ),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s3, "estimate": 13.0 * 0.13},
                        index=pd.date_range(cls.test_end_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s4, "estimate": np.NaN},
                        index=pd.date_range(
                            cls.test_start_date,
                            cls.test_end_date - timedelta(2),
                        ),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s4, "estimate": 14.0 * 0.15},
                        index=pd.date_range(
                            cls.test_end_date - timedelta(1),
                            cls.test_end_date,
                        ),
                    ),
                ]
            )
            .set_index(SID_FIELD_NAME, append=True)
            .unstack(SID_FIELD_NAME)
            .reindex(cls.trading_days)
            .stack(SID_FIELD_NAME, dropna=False)
        )

        split_adjusted_at_end_boundary = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s0,
                            "estimate": np.NaN,
                        },
                        index=pd.date_range(
                            cls.test_start_date,
                            pd.Timestamp("2015-01-08"),
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s0,
                            "estimate": 10.0,
                        },
                        index=pd.date_range(
                            pd.Timestamp("2015-01-09"),
                            cls.test_end_date,
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s1,
                            "estimate": 11.0,
                        },
                        index=pd.date_range(cls.test_start_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s2, "estimate": np.NaN},
                        index=pd.date_range(cls.test_start_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s3, "estimate": np.NaN},
                        index=pd.date_range(
                            cls.test_start_date,
                            cls.test_end_date - timedelta(1),
                        ),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s3, "estimate": 13.0},
                        index=pd.date_range(cls.test_end_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s4, "estimate": np.NaN},
                        index=pd.date_range(
                            cls.test_start_date,
                            cls.test_end_date - timedelta(2),
                        ),
                    ),
                    pd.DataFrame(
                        {SID_FIELD_NAME: cls.s4, "estimate": 14.0},
                        index=pd.date_range(
                            cls.test_end_date - timedelta(1),
                            cls.test_end_date,
                        ),
                    ),
                ]
            )
            .set_index(SID_FIELD_NAME, append=True)
            .unstack(SID_FIELD_NAME)
            .reindex(cls.trading_days)
            .stack(SID_FIELD_NAME, dropna=False)
        )

        split_adjusted_before_start_boundary = split_adjusted_at_start_boundary
        split_adjusted_after_end_boundary = split_adjusted_at_end_boundary

        return {
            cls.test_start_date: split_adjusted_at_start_boundary,
            cls.split_adjusted_before_start: split_adjusted_before_start_boundary,
            cls.test_end_date: split_adjusted_at_end_boundary,
            cls.split_adjusted_after_end: split_adjusted_after_end_boundary,
        }


class NextWithAdjustmentBoundaries(WithAdjustmentBoundaries, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return partial(
            NextSplitAdjustedEarningsEstimatesLoader,
            events,
            columns,
            split_adjustments_loader=cls.adjustment_reader,
            split_adjusted_column_names=["estimate"],
        )

    @classmethod
    def make_expected_out(cls):
        split_adjusted_at_start_boundary = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s0,
                            "estimate": 10,
                        },
                        index=pd.date_range(
                            cls.test_start_date,
                            pd.Timestamp("2015-01-09"),
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s1,
                            "estimate": 11.0,
                        },
                        index=pd.date_range(cls.test_start_date, cls.test_start_date),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s2,
                            "estimate": 12.0,
                        },
                        index=pd.date_range(cls.test_end_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s3,
                            "estimate": 13.0 * 0.13,
                        },
                        index=pd.date_range(
                            cls.test_end_date - timedelta(1),
                            cls.test_end_date,
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s4,
                            "estimate": 14.0,
                        },
                        index=pd.date_range(
                            cls.test_end_date - timedelta(1),
                            cls.test_end_date - timedelta(1),
                        ),
                    ),
                ]
            )
            .set_index(SID_FIELD_NAME, append=True)
            .unstack(SID_FIELD_NAME)
            .reindex(cls.trading_days)
            .stack(SID_FIELD_NAME, dropna=False)
        )

        split_adjusted_at_end_boundary = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s0,
                            "estimate": 10,
                        },
                        index=pd.date_range(
                            cls.test_start_date,
                            pd.Timestamp("2015-01-09"),
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s1,
                            "estimate": 11.0,
                        },
                        index=pd.date_range(cls.test_start_date, cls.test_start_date),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s2,
                            "estimate": 12.0,
                        },
                        index=pd.date_range(cls.test_end_date, cls.test_end_date),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s3,
                            "estimate": 13.0,
                        },
                        index=pd.date_range(
                            cls.test_end_date - timedelta(1),
                            cls.test_end_date,
                        ),
                    ),
                    pd.DataFrame(
                        {
                            SID_FIELD_NAME: cls.s4,
                            "estimate": 14.0,
                        },
                        index=pd.date_range(
                            cls.test_end_date - timedelta(1),
                            cls.test_end_date - timedelta(1),
                        ),
                    ),
                ]
            )
            .set_index(SID_FIELD_NAME, append=True)
            .unstack(SID_FIELD_NAME)
            .reindex(cls.trading_days)
            .stack(SID_FIELD_NAME, dropna=False)
        )

        split_adjusted_before_start_boundary = split_adjusted_at_start_boundary
        split_adjusted_after_end_boundary = split_adjusted_at_end_boundary

        return {
            cls.test_start_date: split_adjusted_at_start_boundary,
            cls.split_adjusted_before_start: split_adjusted_before_start_boundary,
            cls.test_end_date: split_adjusted_at_end_boundary,
            cls.split_adjusted_after_end: split_adjusted_after_end_boundary,
        }


class TestQuarterShift:
    """This tests, in isolation, quarter calculation logic for shifting quarters
    backwards/forwards from a starting point.
    """

    def test_quarter_normalization(self):
        input_yrs = pd.Series(range(2011, 2015), dtype=np.int64)
        input_qtrs = pd.Series(range(1, 5), dtype=np.int64)
        result_years, result_quarters = split_normalized_quarters(
            normalize_quarters(input_yrs, input_qtrs)
        )
        # Can't use assert_series_equal here with check_names=False
        # because that still fails due to name differences.
        # TODO: With pandas > 1. assert_series_equal seems to work fine
        assert_equal(input_yrs, result_years)
        assert_equal(input_qtrs, result_quarters)
