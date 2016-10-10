import blaze as bz
import itertools
from nose.tools import assert_true
from nose_parameterized import parameterized
import numpy as np
from numpy.testing import assert_array_equal
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
from zipline.pipeline.loaders.blaze.estimates import (
    BlazeNextEstimatesLoader,
    BlazePreviousEstimatesLoader
)
from zipline.pipeline.loaders.earnings_estimates import (
    INVALID_NUM_QTRS_MESSAGE,
    NextEarningsEstimatesLoader,
    normalize_quarters,
    PreviousEarningsEstimatesLoader,
    split_normalized_quarters,
)
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithTradingSessions,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal, assert_raises_regex
from zipline.utils.numpy_utils import datetime64ns_dtype
from zipline.utils.numpy_utils import float64_dtype


class Estimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate = Column(dtype=float64_dtype)


def QuartersEstimates(announcements_out):
    class QtrEstimates(Estimates):
        num_announcements = announcements_out
        name = Estimates
    return QtrEstimates


def QuartersEstimatesNoNumQuartersAttr(num_qtr):
    class QtrEstimates(Estimates):
        name = Estimates
    return QtrEstimates


class WithEstimates(WithTradingSessions, WithAssetFinder):
    """
    ZiplineTestCase mixin providing cls.loader and cls.events as class
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
    """

    # Short window defined in order for test to run faster.
    START_DATE = pd.Timestamp('2014-12-28')
    END_DATE = pd.Timestamp('2015-02-04')

    @classmethod
    def make_loader(cls, events, columns):
        raise NotImplementedError('make_loader')

    @classmethod
    def make_events(cls):
        raise NotImplementedError('make_events')

    @classmethod
    def get_sids(cls):
        return cls.events[SID_FIELD_NAME].unique()

    @classmethod
    def init_class_fixtures(cls):
        cls.events = cls.make_events()
        cls.ASSET_FINDER_EQUITY_SIDS = cls.get_sids()
        cls.columns = {
            Estimates.event_date: 'event_date',
            Estimates.fiscal_quarter: 'fiscal_quarter',
            Estimates.fiscal_year: 'fiscal_year',
            Estimates.estimate: 'estimate'
        }
        cls.loader = cls.make_loader(cls.events, {column.name: val for
                                                  column, val in
                                                  cls.columns.items()})
        cls.ASSET_FINDER_EQUITY_SYMBOLS = [
            's' + str(n) for n in cls.ASSET_FINDER_EQUITY_SIDS
        ]
        super(WithEstimates, cls).init_class_fixtures()


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
        return pd.DataFrame({SID_FIELD_NAME: 0},
                            columns=[SID_FIELD_NAME,
                                     TS_FIELD_NAME,
                                     EVENT_DATE_FIELD_NAME,
                                     FISCAL_QUARTER_FIELD_NAME,
                                     FISCAL_YEAR_FIELD_NAME,
                                     'estimate'],
                            index=[0])

    def test_wrong_num_announcements_passed(self):
        bad_dataset1 = QuartersEstimates(-1)
        bad_dataset2 = QuartersEstimates(-2)
        good_dataset = QuartersEstimates(1)
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )
        columns = {c.name + str(dataset.num_announcements): c.latest
                   for dataset in (bad_dataset1,
                                   bad_dataset2,
                                   good_dataset)
                   for c in dataset.columns}
        p = Pipeline(columns)

        with self.assertRaises(ValueError) as e:
            engine.run_pipeline(
                p,
                start_date=self.trading_days[0],
                end_date=self.trading_days[-1],
            )
            assert_raises_regex(e, INVALID_NUM_QTRS_MESSAGE % "-1,-2")

    def test_no_num_announcements_attr(self):
        dataset = QuartersEstimatesNoNumQuartersAttr(1)
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )
        p = Pipeline({c.name: c.latest for c in dataset.columns})

        with self.assertRaises(AttributeError):
            engine.run_pipeline(
                p,
                start_date=self.trading_days[0],
                end_date=self.trading_days[-1],
            )


class PreviousWithWrongNumQuarters(WithWrongLoaderDefinition,
                                   ZiplineTestCase):
    """
    Tests that previous quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)


class NextWithWrongNumQuarters(WithWrongLoaderDefinition,
                               ZiplineTestCase):
    """
    Tests that next quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)


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
    END_DATE = pd.Timestamp('2015-01-28')

    q1_knowledge_dates = [pd.Timestamp('2015-01-01'),
                          pd.Timestamp('2015-01-04'),
                          pd.Timestamp('2015-01-07'),
                          pd.Timestamp('2015-01-11')]
    q2_knowledge_dates = [pd.Timestamp('2015-01-14'),
                          pd.Timestamp('2015-01-17'),
                          pd.Timestamp('2015-01-20'),
                          pd.Timestamp('2015-01-23')]
    # We want to model the possibility of an estimate predicting a release date
    # that doesn't match the actual release. This could be done by dynamically
    # generating more combinations with different release dates, but that
    # significantly increases the amount of time it takes to run the tests.
    # These hard-coded cases are sufficient to know that we can update our
    # beliefs when we get new information.
    q1_release_dates = [pd.Timestamp('2015-01-13'),
                        pd.Timestamp('2015-01-14')]  # One day late
    q2_release_dates = [pd.Timestamp('2015-01-25'),  # One day early
                        pd.Timestamp('2015-01-26')]

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
            itertools.permutations(cls.q1_knowledge_dates +
                                   cls.q2_knowledge_dates,
                                   4)
        )
        for sid, (q1e1, q1e2, q2e1, q2e2) in it:
            # We're assuming that estimates must come before the relevant
            # release.
            if (q1e1 < q1e2 and
                    q2e1 < q2e2 and
                    # All estimates are < Q2's event, so just constrain Q1
                    # estimates.
                    q1e1 < cls.q1_release_dates[0] and
                    q1e2 < cls.q1_release_dates[0]):
                sid_estimates.append(cls.create_estimates_df(q1e1,
                                                             q1e2,
                                                             q2e1,
                                                             q2e2,
                                                             sid))
                sid_releases.append(cls.create_releases_df(sid))
        return pd.concat(sid_estimates +
                         sid_releases).reset_index(drop=True)

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
        return pd.DataFrame({
            TS_FIELD_NAME: [pd.Timestamp('2015-01-13'),
                            pd.Timestamp('2015-01-26')],
            EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-13'),
                                    pd.Timestamp('2015-01-26')],
            'estimate': [0.5, 0.8],
            FISCAL_QUARTER_FIELD_NAME: [1.0, 2.0],
            FISCAL_YEAR_FIELD_NAME: [2015.0, 2015.0],
            SID_FIELD_NAME: sid
        })

    @classmethod
    def create_estimates_df(cls,
                            q1e1,
                            q1e2,
                            q2e1,
                            q2e2,
                            sid):
        return pd.DataFrame({
            EVENT_DATE_FIELD_NAME: cls.q1_release_dates + cls.q2_release_dates,
            'estimate': [.1, .2, .3, .4],
            FISCAL_QUARTER_FIELD_NAME: [1.0, 1.0, 2.0, 2.0],
            FISCAL_YEAR_FIELD_NAME: [2015.0, 2015.0, 2015.0, 2015.0],
            TS_FIELD_NAME: [q1e1, q1e2, q2e1, q2e2],
            SID_FIELD_NAME: sid,
        })

    @classmethod
    def init_class_fixtures(cls):
        super(WithEstimatesTimeZero, cls).init_class_fixtures()

    def get_expected_estimate(self,
                              q1_knowledge,
                              q2_knowledge,
                              comparable_date):
        return pd.DataFrame()

    def test_estimates(self):
        dataset = QuartersEstimates(1)
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )
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
                assert_true(sid_estimates.isnull().all().all())
            else:
                ts_sorted_estimates = self.events[
                    self.events[SID_FIELD_NAME] == sid
                ].sort(TS_FIELD_NAME)
                q1_knowledge = ts_sorted_estimates[
                    ts_sorted_estimates[FISCAL_QUARTER_FIELD_NAME] == 1
                ]
                q2_knowledge = ts_sorted_estimates[
                    ts_sorted_estimates[FISCAL_QUARTER_FIELD_NAME] == 2
                ]
                all_expected = pd.concat(
                    [self.get_expected_estimate(
                        q1_knowledge[q1_knowledge[TS_FIELD_NAME] <=
                                     date.tz_localize(None)],
                        q2_knowledge[q2_knowledge[TS_FIELD_NAME] <=
                                     date.tz_localize(None)],
                        date.tz_localize(None),
                    ).set_index([[date]]) for date in sid_estimates.index],
                    axis=0)
                assert_equal(all_expected[sid_estimates.columns],
                             sid_estimates)


class NextEstimate(WithEstimatesTimeZero, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    def get_expected_estimate(self,
                              q1_knowledge,
                              q2_knowledge,
                              comparable_date):
        # If our latest knowledge of q1 is that the release is
        # happening on this simulation date or later, then that's
        # the estimate we want to use.
        if (not q1_knowledge.empty and
            q1_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] >=
                comparable_date):
            return q1_knowledge.iloc[-1:]
        # If q1 has already happened or we don't know about it
        # yet and our latest knowledge indicates that q2 hasn't
        # happened yet, then that's the estimate we want to use.
        elif (not q2_knowledge.empty and
              q2_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] >=
                comparable_date):
            return q2_knowledge.iloc[-1:]
        return pd.DataFrame(columns=q1_knowledge.columns,
                            index=[comparable_date])


class BlazeNextEstimateLoaderTestCase(NextEstimate):
    """
    Run the same tests as EventsLoaderTestCase, but using a BlazeEventsLoader.
    """

    @classmethod
    def make_loader(cls, events, columns):
        return BlazeNextEstimatesLoader(
            bz.data(events),
            columns,
        )


class PreviousEstimate(WithEstimatesTimeZero, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    def get_expected_estimate(self,
                              q1_knowledge,
                              q2_knowledge,
                              comparable_date):

        # The expected estimate will be for q2 if the last thing
        # we've seen is that the release date already happened.
        # Otherwise, it'll be for q1, as long as the release date
        # for q1 has already happened.
        if (not q2_knowledge.empty and
            q2_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] <=
                comparable_date):
            return q2_knowledge.iloc[-1:]
        elif (not q1_knowledge.empty and
              q1_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] <=
                comparable_date):
            return q1_knowledge.iloc[-1:]
        return pd.DataFrame(columns=q1_knowledge.columns,
                            index=[comparable_date])


class BlazePreviousEstimateLoaderTestCase(PreviousEstimate):
    """
    Run the same tests as EventsLoaderTestCase, but using a BlazeEventsLoader.
    """

    @classmethod
    def make_loader(cls, events, columns):
        return BlazePreviousEstimatesLoader(
            bz.data(events),
            columns,
        )


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
        return pd.DataFrame({
            SID_FIELD_NAME: [0] * 2,
            TS_FIELD_NAME: [pd.Timestamp('2015-01-01'),
                            pd.Timestamp('2015-01-06')],
            EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-10'),
                                    pd.Timestamp('2015-01-20')],
            'estimate': [1., 2.],
            FISCAL_QUARTER_FIELD_NAME: [1, 2],
            FISCAL_YEAR_FIELD_NAME: [2015, 2015]
        })

    @classmethod
    def init_class_fixtures(cls):
        super(WithEstimateMultipleQuarters, cls).init_class_fixtures()
        cls.expected_out = cls.make_expected_out()

    @classmethod
    def make_expected_out(cls):
        expected = pd.DataFrame(columns=[cls.columns[col] + '1'
                                         for col in cls.columns] +
                                        [cls.columns[col] + '2'
                                         for col in cls.columns],
                                index=cls.trading_days)

        for (col, raw_name), suffix in itertools.product(
            cls.columns.items(), ('1', '2')
        ):
            expected_name = raw_name + suffix
            if col.dtype == datetime64ns_dtype:
                expected[expected_name] = pd.to_datetime(
                    expected[expected_name]
                )
            else:
                expected[expected_name] = expected[
                    expected_name
                ].astype(col.dtype)
        cls.fill_expected_out(expected)
        return expected.reindex(cls.trading_days)

    def test_multiple_qtrs_requested(self):
        dataset1 = QuartersEstimates(1)
        dataset2 = QuartersEstimates(2)
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )

        results = engine.run_pipeline(
            Pipeline(
                merge([{c.name + '1': c.latest for c in dataset1.columns},
                       {c.name + '2': c.latest for c in dataset2.columns}])
            ),
            start_date=self.trading_days[0],
            end_date=self.trading_days[-1],
        )
        q1_columns = [col.name + '1' for col in self.columns]
        q2_columns = [col.name + '2' for col in self.columns]

        # We now expect a column for 1 quarter out and a column for 2
        # quarters out for each of the dataset columns.
        assert_equal(sorted(np.array(q1_columns + q2_columns)),
                     sorted(results.columns.values))
        assert_equal(self.expected_out.sort(axis=1),
                     results.xs(0, level=1).sort(axis=1))


class NextEstimateMultipleQuarters(
    WithEstimateMultipleQuarters, ZiplineTestCase
):
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def fill_expected_out(cls, expected):
        # Fill columns for 1 Q out
        for raw_name in cls.columns.values():
            expected.loc[
                pd.Timestamp('2015-01-01'):pd.Timestamp('2015-01-11'),
                raw_name + '1'
            ] = cls.events[raw_name].iloc[0]
            expected.loc[
                pd.Timestamp('2015-01-11'):pd.Timestamp('2015-01-20'),
                raw_name + '1'
            ] = cls.events[raw_name].iloc[1]

        # Fill columns for 2 Q out
        # We only have an estimate and event date for 2 quarters out before
        # Q1's event happens; after Q1's event, we know 1 Q out but not 2 Qs
        # out.
        for col_name in ['estimate', 'event_date']:
            expected.loc[
                pd.Timestamp('2015-01-06'):pd.Timestamp('2015-01-10'),
                col_name + '2'
            ] = cls.events[col_name].iloc[1]
        # But we know what FQ and FY we'd need in both Q1 and Q2
        # because we know which FQ is next and can calculate from there
        expected.loc[
            pd.Timestamp('2015-01-01'):pd.Timestamp('2015-01-09'),
            FISCAL_QUARTER_FIELD_NAME + '2'
        ] = 2
        expected.loc[
            pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-20'),
            FISCAL_QUARTER_FIELD_NAME + '2'
        ] = 3
        expected.loc[
            pd.Timestamp('2015-01-01'):pd.Timestamp('2015-01-20'),
            FISCAL_YEAR_FIELD_NAME + '2'
        ] = 2015

        return expected


class PreviousEstimateMultipleQuarters(
    WithEstimateMultipleQuarters,
    ZiplineTestCase
):

    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def fill_expected_out(cls, expected):
        # Fill columns for 1 Q out
        for raw_name in cls.columns.values():
            expected[raw_name + '1'].loc[
                pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-19')
            ] = cls.events[raw_name].iloc[0]
            expected[raw_name + '1'].loc[
                pd.Timestamp('2015-01-20'):
            ] = cls.events[raw_name].iloc[1]

        # Fill columns for 2 Q out
        for col_name in ['estimate', 'event_date']:
            expected[col_name + '2'].loc[
                pd.Timestamp('2015-01-20'):
            ] = cls.events[col_name].iloc[0]
        expected[
            FISCAL_QUARTER_FIELD_NAME + '2'
        ].loc[pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-20')] = 4
        expected[
            FISCAL_YEAR_FIELD_NAME + '2'
        ].loc[pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-20')] = 2014
        expected[
            FISCAL_QUARTER_FIELD_NAME + '2'
        ].loc[pd.Timestamp('2015-01-20'):] = 1
        expected[
            FISCAL_YEAR_FIELD_NAME + '2'
        ].loc[pd.Timestamp('2015-01-20'):] = 2015
        return expected


class WithVaryingNumEstimates(WithEstimates):
    """
    ZiplineTestCase mixin providing fixtures and a test to ensure that we
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
        return pd.DataFrame({
            SID_FIELD_NAME: [0] * 3 + [1] * 3,
            TS_FIELD_NAME: [pd.Timestamp('2015-01-09'),
                            pd.Timestamp('2015-01-12'),
                            pd.Timestamp('2015-01-13')] * 2,
            EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-12'),
                                    pd.Timestamp('2015-01-13'),
                                    pd.Timestamp('2015-01-20'),
                                    pd.Timestamp('2015-01-13'),
                                    pd.Timestamp('2015-01-12'),
                                    pd.Timestamp('2015-01-20')],
            'estimate': [11., 12., 21.] * 2,
            FISCAL_QUARTER_FIELD_NAME: [1, 1, 2] * 2,
            FISCAL_YEAR_FIELD_NAME: [2015] * 6
        })

    @classmethod
    def assert_compute(cls, estimate, today):
        raise NotImplementedError('assert_compute')

    def test_windows_with_varying_num_estimates(self):
        dataset = QuartersEstimates(1)
        assert_compute = self.assert_compute

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate]
            window_length = 3

            def compute(self, today, assets, out, estimate):
                assert_compute(estimate, today)

        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )
        engine.run_pipeline(
            Pipeline({'est': SomeFactor()}),
            start_date=pd.Timestamp('2015-01-13', tz='utc'),
            # last event date we have
            end_date=pd.Timestamp('2015-01-14', tz='utc'),
        )


class PreviousVaryingNumEstimates(
    WithVaryingNumEstimates,
    ZiplineTestCase
):
    def assert_compute(self, estimate, today):
        if today == pd.Timestamp('2015-01-13', tz='utc'):
            assert_array_equal(estimate[:, 0],
                               np.array([np.NaN, np.NaN, 12]))
            assert_array_equal(estimate[:, 1],
                               np.array([np.NaN, 12, 12]))
        else:
            assert_array_equal(estimate[:, 0],
                               np.array([np.NaN, 12, 12]))
            assert_array_equal(estimate[:, 1],
                               np.array([12, 12, 12]))

    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)


class NextVaryingNumEstimates(
    WithVaryingNumEstimates,
    ZiplineTestCase
):

    def assert_compute(self, estimate, today):
        if today == pd.Timestamp('2015-01-13', tz='utc'):
            assert_array_equal(estimate[:, 0],
                               np.array([11, 12, 12]))
            assert_array_equal(estimate[:, 1],
                               np.array([np.NaN, np.NaN, 21]))
        else:
            assert_array_equal(estimate[:, 0],
                               np.array([np.NaN, 21, 21]))
            assert_array_equal(estimate[:, 1],
                               np.array([np.NaN, 21, 21]))

    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)


class WithEstimateWindows(WithEstimates):
    """
    ZiplineTestCase mixin providing fixures and a test to test running a
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

    window_test_start_date = pd.Timestamp('2015-01-05')
    critical_dates = [pd.Timestamp('2015-01-09', tz='utc'),
                      pd.Timestamp('2015-01-12', tz='utc'),
                      pd.Timestamp('2015-01-15', tz='utc'),
                      pd.Timestamp('2015-01-20', tz='utc')]
    # window length, starting date, num quarters out, timeline. Parameterizes
    # over number of quarters out.
    window_test_cases = list(itertools.product(critical_dates, (1, 2)))

    @classmethod
    def make_events(cls):
        sid_0_timeline = pd.DataFrame({
            TS_FIELD_NAME: [pd.Timestamp('2015-01-05'),
                            pd.Timestamp('2015-01-07'),
                            pd.Timestamp('2015-01-05'),
                            pd.Timestamp('2015-01-17')],
            EVENT_DATE_FIELD_NAME:
                [pd.Timestamp('2015-01-10'),
                 pd.Timestamp('2015-01-10'),
                 pd.Timestamp('2015-01-20'),
                 pd.Timestamp('2015-01-20')],
            'estimate': [100., 101.] + [200., 201.],
            FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2,
            FISCAL_YEAR_FIELD_NAME: 2015,
            SID_FIELD_NAME: 0,
        })

        sid_1_timeline = pd.DataFrame({
            TS_FIELD_NAME: [pd.Timestamp('2015-01-09'),
                            pd.Timestamp('2015-01-12'),
                            pd.Timestamp('2015-01-09'),
                            pd.Timestamp('2015-01-15')],
            EVENT_DATE_FIELD_NAME:
                [pd.Timestamp('2015-01-12'), pd.Timestamp('2015-01-12'),
                 pd.Timestamp('2015-01-15'), pd.Timestamp('2015-01-15')],
            'estimate': [110., 111.] + [310., 311.],
            FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [3] * 2,
            FISCAL_YEAR_FIELD_NAME: 2015,
            SID_FIELD_NAME: 1
        })

        # Extra sid to make sure we have correct overwrites when sid quarter
        # boundaries collide.
        sid_3_timeline = pd.DataFrame({
            TS_FIELD_NAME: [pd.Timestamp('2015-01-05'),
                            pd.Timestamp('2015-01-07'),
                            pd.Timestamp('2015-01-05'),
                            pd.Timestamp('2015-01-17')],
            EVENT_DATE_FIELD_NAME:
                [pd.Timestamp('2015-01-10'),
                 pd.Timestamp('2015-01-10'),
                 pd.Timestamp('2015-01-20'),
                 pd.Timestamp('2015-01-20')],
            'estimate': [120., 121.] + [220., 221.],
            FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2,
            FISCAL_YEAR_FIELD_NAME: 2015,
            SID_FIELD_NAME: 2
        })
        return pd.concat([sid_0_timeline, sid_1_timeline, sid_3_timeline])

    @classmethod
    def make_expected_timelines(cls):
        return {}

    @classmethod
    def init_class_fixtures(cls):
        super(WithEstimateWindows, cls).init_class_fixtures()
        cls.timelines = cls.make_expected_timelines()

    @classmethod
    def create_expected_df(cls, tuples, end_date):
        """
        Given a list of tuples of new data we get for each sid on each critical
        date (when information changes), create a DataFrame that fills that
        data through a date range ending at `end_date`.
        """
        df = pd.DataFrame(tuples,
                          columns=[SID_FIELD_NAME,
                                   'estimate',
                                   'knowledge_date'])
        df = df.pivot_table(columns=SID_FIELD_NAME,
                            values='estimate',
                            index='knowledge_date')
        df = df.reindex(
            pd.date_range(cls.window_test_start_date, end_date)
        )
        # Index name is lost during reindex.
        df.index = df.index.rename('knowledge_date')
        df['at_date'] = end_date.tz_localize('utc')
        df = df.set_index(['at_date', df.index.tz_localize('utc')]).ffill()
        return df

    @parameterized.expand(window_test_cases)
    def test_estimate_windows_at_quarter_boundaries(self,
                                                    start_idx,
                                                    num_announcements_out):
        dataset = QuartersEstimates(num_announcements_out)
        trading_days = self.trading_days
        timelines = self.timelines
        # The window length should be from the starting index back to the first
        # date on which we got data. The goal is to ensure that as we
        # progress through the timeline, all data we got, starting from that
        # first date, is correctly overwritten.
        window_len = (
            self.trading_days.get_loc(start_idx) -
            self.trading_days.get_loc(self.window_test_start_date) + 1
        )

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate]
            window_length = window_len

            def compute(self, today, assets, out, estimate):
                today_idx = trading_days.get_loc(today)
                today_timeline = timelines[
                    num_announcements_out
                ].loc[today].reindex(
                    trading_days[:today_idx + 1]
                ).values
                timeline_start_idx = (len(today_timeline) - window_len)
                assert_equal(estimate,
                             today_timeline[timeline_start_idx:])
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )
        engine.run_pipeline(
            Pipeline({'est': SomeFactor()}),
            start_date=start_idx,
            # last event date we have
            end_date=pd.Timestamp('2015-01-20', tz='utc'),
        )


class PreviousEstimateWindows(WithEstimateWindows, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_timelines(cls):
        oneq_previous = pd.concat([
            cls.create_expected_df(
                [(0, np.NaN, cls.window_test_start_date),
                 (1, np.NaN, cls.window_test_start_date),
                 (2, np.NaN, cls.window_test_start_date)],
                pd.Timestamp('2015-01-09')
            ),
            cls.create_expected_df(
                [(0, 101, pd.Timestamp('2015-01-10')),
                 (1, 111, pd.Timestamp('2015-01-12')),
                 (2, 121, pd.Timestamp('2015-01-10'))],
                pd.Timestamp('2015-01-12')
            ),
            cls.create_expected_df(
                [(0, 101, pd.Timestamp('2015-01-10')),
                 (1, 111, pd.Timestamp('2015-01-12')),
                 (2, 121, pd.Timestamp('2015-01-10'))],
                pd.Timestamp('2015-01-13')
            ),
            cls.create_expected_df(
                [(0, 101, pd.Timestamp('2015-01-10')),
                 (1, 111, pd.Timestamp('2015-01-12')),
                 (2, 121, pd.Timestamp('2015-01-10'))],
                pd.Timestamp('2015-01-14')
            ),
            cls.create_expected_df(
                [(0, 101, pd.Timestamp('2015-01-10')),
                 (1, 311, pd.Timestamp('2015-01-15')),
                 (2, 121, pd.Timestamp('2015-01-10'))],
                pd.Timestamp('2015-01-15')
            ),
            cls.create_expected_df(
                [(0, 101, pd.Timestamp('2015-01-10')),
                 (1, 311, pd.Timestamp('2015-01-15')),
                 (2, 121, pd.Timestamp('2015-01-10'))],
                pd.Timestamp('2015-01-16')
            ),
            cls.create_expected_df(
                [(0, 201, pd.Timestamp('2015-01-17')),
                 (1, 311, pd.Timestamp('2015-01-15')),
                 (2, 221, pd.Timestamp('2015-01-17'))],
                pd.Timestamp('2015-01-20')
            ),
        ])

        twoq_previous = pd.concat(
            [cls.create_expected_df(
                [(0, np.NaN, cls.window_test_start_date),
                 (1, np.NaN, cls.window_test_start_date),
                 (2, np.NaN, cls.window_test_start_date)],
                end_date
            ) for end_date in pd.date_range('2015-01-09', '2015-01-19')] +
            [cls.create_expected_df(
                [(0, 101, pd.Timestamp('2015-01-20')),
                 (1, np.NaN, cls.window_test_start_date),
                 (2, 121, pd.Timestamp('2015-01-20'))],
                pd.Timestamp('2015-01-20')
            )]
        )
        return {
            1: oneq_previous,
            2: twoq_previous
        }


class NextEstimateWindows(WithEstimateWindows, ZiplineTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_timelines(cls):
        oneq_next = pd.concat([
            cls.create_expected_df(
                [(0, 100, cls.window_test_start_date),
                 (0, 101, pd.Timestamp('2015-01-07')),
                 (1, 110, pd.Timestamp('2015-01-09')),
                 (2, 120, cls.window_test_start_date),
                 (2, 121, pd.Timestamp('2015-01-07'))],
                pd.Timestamp('2015-01-09')
            ),
            cls.create_expected_df(
                [(0, 200, cls.window_test_start_date),
                 (1, 110, pd.Timestamp('2015-01-09')),
                 (1, 111, pd.Timestamp('2015-01-12')),
                 (2, 220, cls.window_test_start_date)],
                pd.Timestamp('2015-01-12')
            ),
            cls.create_expected_df(
                [(0, 200, cls.window_test_start_date),
                 (1, 310, pd.Timestamp('2015-01-09')),
                 (2, 220, cls.window_test_start_date)],
                pd.Timestamp('2015-01-13')
            ),
            cls.create_expected_df(
                [(0, 200, cls.window_test_start_date),
                 (1, 310, pd.Timestamp('2015-01-09')),
                 (2, 220, cls.window_test_start_date)],
                pd.Timestamp('2015-01-14')
            ),
            cls.create_expected_df(
                [(0, 200, cls.window_test_start_date),
                 (1, 310, pd.Timestamp('2015-01-09')),
                 (1, 311, pd.Timestamp('2015-01-15')),
                 (2, 220, cls.window_test_start_date)],
                pd.Timestamp('2015-01-15')
            ),
            cls.create_expected_df(
                [(0, 200, cls.window_test_start_date),
                 (1, np.NaN, cls.window_test_start_date),
                 (2, 220, cls.window_test_start_date)],
                pd.Timestamp('2015-01-16')
            ),
            cls.create_expected_df(
                [(0, 200, cls.window_test_start_date),
                 (0, 201, pd.Timestamp('2015-01-17')),
                 (1, np.NaN, cls.window_test_start_date),
                 (2, 220, cls.window_test_start_date),
                 (2, 221, pd.Timestamp('2015-01-17'))],
                pd.Timestamp('2015-01-20')
            ),
        ])

        twoq_next = pd.concat(
            [cls.create_expected_df(
                [(0, 200, pd.Timestamp(cls.window_test_start_date)),
                 (1, np.NaN, pd.Timestamp(cls.window_test_start_date)),
                 (2, 220, pd.Timestamp(cls.window_test_start_date))],
                pd.Timestamp('2015-01-09')
            )] +
            [cls.create_expected_df(
                [(0, np.NaN, pd.Timestamp(cls.window_test_start_date)),
                 (1, np.NaN, pd.Timestamp(cls.window_test_start_date)),
                 (2, np.NaN, pd.Timestamp(cls.window_test_start_date))],
                end_date
            ) for end_date in pd.date_range('2015-01-12', '2015-01-20')]
        )

        return {
            1: oneq_next,
            2: twoq_next
        }


class QuarterShiftTestCase(ZiplineTestCase):
    """
    This tests, in isolation, quarter calculation logic for shifting quarters
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
        assert_equal(input_yrs, result_years)
        assert_equal(input_qtrs, result_quarters)
