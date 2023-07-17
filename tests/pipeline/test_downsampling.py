"""
Tests for Downsampled Filters/Factors/Classifiers
"""
from functools import partial

import pandas as pd
from pandas.testing import assert_frame_equal

from zipline.errors import NoFurtherDataError
from zipline.pipeline import (
    Pipeline,
    CustomFactor,
    CustomFilter,
    CustomClassifier,
    SimplePipelineEngine,
)
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.domain import (
    CA_EQUITIES,
    EquitySessionDomain,
    GB_EQUITIES,
    US_EQUITIES,
)
from zipline.pipeline.factors import SimpleMovingAverage
from zipline.pipeline.filters.smoothing import All
from zipline.testing import ZiplineTestCase, parameter_space, ExplodingObject
from zipline.testing.fixtures import (
    WithTradingSessions,
    WithSeededRandomPipelineEngine,
    WithAssetFinder,
)
from zipline.utils.classproperty import classproperty
from zipline.utils.input_validation import _qualified_name
from zipline.utils.numpy_utils import int64_dtype
import pytest
import re


class NDaysAgoFactor(CustomFactor):
    inputs = [TestingDataSet.float_col]

    def compute(self, today, assets, out, floats):
        out[:] = floats[0]


class NDaysAgoFilter(CustomFilter):
    inputs = [TestingDataSet.bool_col]

    def compute(self, today, assets, out, bools):
        out[:] = bools[0]


class NDaysAgoClassifier(CustomClassifier):
    inputs = [TestingDataSet.categorical_col]
    dtype = TestingDataSet.categorical_col.dtype

    def compute(self, today, assets, out, cats):
        out[:] = cats[0]


class ComputeExtraRowsTestCase(WithTradingSessions, ZiplineTestCase):

    DATA_MIN_DAY = pd.Timestamp("2012-06")
    DATA_MAX_DAY = pd.Timestamp("2015")
    TRADING_CALENDAR_STRS = ("NYSE", "LSE", "TSX")

    # Test with different window_lengths to ensure that window length is not
    # used when calculating exra rows for the top-level term.
    factor1 = TestingDataSet.float_col.latest
    factor11 = NDaysAgoFactor(window_length=11)
    factor91 = NDaysAgoFactor(window_length=91)

    filter1 = TestingDataSet.bool_col.latest
    filter11 = NDaysAgoFilter(window_length=11)
    filter91 = NDaysAgoFilter(window_length=91)

    classifier1 = TestingDataSet.categorical_col.latest
    classifier11 = NDaysAgoClassifier(window_length=11)
    classifier91 = NDaysAgoClassifier(window_length=91)

    all_terms = [
        factor1,
        factor11,
        factor91,
        filter1,
        filter11,
        filter91,
        classifier1,
        classifier11,
        classifier91,
    ]

    @parameter_space(
        calendar_name=TRADING_CALENDAR_STRS,
        base_terms=[
            (factor1, factor11, factor91),
            (filter1, filter11, filter91),
            (classifier1, classifier11, classifier91),
        ],
        __fail_fast=True,
    )
    def test_yearly(self, base_terms, calendar_name):
        downsampled_terms = tuple(t.downsample("year_start") for t in base_terms)
        all_terms = base_terms + downsampled_terms

        all_sessions = self.trading_sessions[calendar_name]
        end_session = all_sessions[-1]

        years = all_sessions.year
        sessions_in_2012 = all_sessions[years == 2012]
        sessions_in_2013 = all_sessions[years == 2013]
        sessions_in_2014 = all_sessions[years == 2014]

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the first date in 2014.  We shouldn't request any
        # additional rows for the regular terms or the downsampled terms.
        for i in range(0, 30, 5):
            start_session = sessions_in_2014[i]
            self.check_extra_row_calculations(
                all_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land on the second date in 2014.  We should request one more extra
        # row in the downsampled terms to push us back to the first date in
        # 2014.
        for i in range(0, 30, 5):
            start_session = sessions_in_2014[i + 1]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i + 1,
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land on the last date of 2013. The downsampled terms should request
        # enough extra rows to push us back to the start of 2013.
        for i in range(0, 30, 5):
            start_session = sessions_in_2014[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(sessions_in_2013),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land on the last date of 2012. The downsampled terms should request
        # enough extra rows to push us back to the first known date, which is
        # in the middle of 2012.
        for i in range(0, 30, 5):
            start_session = sessions_in_2013[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(sessions_in_2012),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land prior to the first date of 2012. The downsampled terms will fail
        # to request enough extra rows.
        for i in range(0, 30, 5):
            with pytest.raises(
                NoFurtherDataError, match=r"\s*Insufficient data to compute Pipeline"
            ):
                self.check_extra_row_calculations(
                    downsampled_terms,
                    all_sessions,
                    all_sessions[i],
                    end_session,
                    min_extra_rows=i + 1,
                    expected_extra_rows=i + 1,
                )

            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                all_sessions[i],
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

    @parameter_space(
        calendar_name=TRADING_CALENDAR_STRS,
        base_terms=[
            (factor1, factor11, factor91),
            (filter1, filter11, filter91),
            (classifier1, classifier11, classifier91),
        ],
        __fail_fast=True,
    )
    def test_quarterly(self, calendar_name, base_terms):
        downsampled_terms = tuple(t.downsample("quarter_start") for t in base_terms)
        all_terms = base_terms + downsampled_terms

        # This region intersects with Q4 2013, Q1 2014, and Q2 2014.
        tmp = self.trading_sessions[calendar_name]
        all_sessions = tmp[tmp.slice_indexer("2013-12-15", "2014-04-30")]
        end_session = all_sessions[-1]

        months = all_sessions.month
        Q4_2013 = all_sessions[months == 12]
        Q1_2014 = all_sessions[(months == 1) | (months == 2) | (months == 3)]
        Q2_2014 = all_sessions[months == 4]

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the first date in Q2 2014.  We shouldn't request any
        # additional rows for the regular terms or the downsampled terms.
        for i in range(0, 15, 5):
            start_session = Q2_2014[i]
            self.check_extra_row_calculations(
                all_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the second date in Q2 2014.
        # The downsampled terms should request one more extra row.
        for i in range(0, 15, 5):
            start_session = Q2_2014[i + 1]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i + 1,
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the last date in Q1 2014.  The downsampled terms
        # should request enough extra rows to push us back to the first date of
        # Q1 2014.
        for i in range(0, 15, 5):
            start_session = Q2_2014[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(Q1_2014),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the last date in Q4 2013.  The downsampled terms
        # should request enough extra rows to push us back to the first known
        # date, which is in the middle of december 2013.
        for i in range(0, 15, 5):
            start_session = Q1_2014[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(Q4_2013),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

    @parameter_space(
        calendar_name=TRADING_CALENDAR_STRS,
        base_terms=[
            (factor1, factor11, factor91),
            (filter1, filter11, filter91),
            (classifier1, classifier11, classifier91),
        ],
        __fail_fast=True,
    )
    def test_monthly(self, calendar_name, base_terms):
        downsampled_terms = tuple(t.downsample("month_start") for t in base_terms)
        all_terms = base_terms + downsampled_terms

        # This region intersects with Dec 2013, Jan 2014, and Feb 2014.
        tmp = self.trading_sessions[calendar_name]
        all_sessions = tmp[tmp.slice_indexer("2013-12-15", "2014-02-28")]
        end_session = all_sessions[-1]

        months = all_sessions.month
        dec2013 = all_sessions[months == 12]
        jan2014 = all_sessions[months == 1]
        feb2014 = all_sessions[months == 2]

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the first date in feb 2014.  We shouldn't request any
        # additional rows for the regular terms or the downsampled terms.
        for i in range(0, 10, 2):
            start_session = feb2014[i]
            self.check_extra_row_calculations(
                all_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land on the second date in feb 2014.  We should request one more
        # extra row in the downsampled terms to push us back to the first date
        # in 2014.
        for i in range(0, 10, 2):
            start_session = feb2014[i + 1]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i + 1,
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land on the last date of jan 2014. The downsampled terms should
        # request enough extra rows to push us back to the start of jan 2014.
        for i in range(0, 10, 2):
            start_session = feb2014[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(jan2014),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land on the last date of dec 2013. The downsampled terms should
        # request enough extra rows to push us back to the first known date,
        # which is in the middle of december 2013.
        for i in range(0, 10, 2):
            start_session = jan2014[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(dec2013),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

    @parameter_space(
        calendar_name=TRADING_CALENDAR_STRS,
        base_terms=[
            (factor1, factor11, factor91),
            (filter1, filter11, filter91),
            (classifier1, classifier11, classifier91),
        ],
        __fail_fast=True,
    )
    def test_weekly(self, calendar_name, base_terms):
        downsampled_terms = tuple(t.downsample("week_start") for t in base_terms)
        all_terms = base_terms + downsampled_terms

        #    December 2013
        # Mo Tu We Th Fr Sa Su
        #                    1
        #  2  3  4  5  6  7  8
        #  9 10 11 12 13 14 15
        # 16 17 18 19 20 21 22
        # 23 24 25 26 27 28 29
        # 30 31

        #     January 2014
        # Mo Tu We Th Fr Sa Su
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30 31

        # This region intersects with the last full week of 2013, the week
        # shared by 2013 and 2014, and the first full week of 2014.
        tmp = self.trading_sessions[calendar_name]
        all_sessions = tmp[tmp.slice_indexer("2013-12-27", "2014-01-12")]
        end_session = all_sessions[-1]

        week0 = all_sessions[all_sessions.slice_indexer("2013-12-27", "2013-12-29")]
        week1 = all_sessions[all_sessions.slice_indexer("2013-12-30", "2014-01-05")]
        week2 = all_sessions[all_sessions.slice_indexer("2014-01-06", "2014-01-12")]

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the first date in week 2.  We shouldn't request any
        # additional rows for the regular terms or the downsampled terms.
        for i in range(3):
            start_session = week2[i]
            self.check_extra_row_calculations(
                all_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the second date in week 2.  The downsampled terms
        # should request one more extra row.
        for i in range(3):
            start_session = week2[i + 1]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i + 1,
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i,
                expected_extra_rows=i,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the last date in week 1.  The downsampled terms
        # should request enough extra rows to push us back to the first date of
        # week 1.
        for i in range(3):
            start_session = week2[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(week1),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

        # Simulate requesting computation where the unaltered lookback would
        # land exactly on the last date in week0.  The downsampled terms
        # should request enough extra rows to push us back to the first known
        # date, which is in the middle of december 2013.
        for i in range(3):
            start_session = week1[i]
            self.check_extra_row_calculations(
                downsampled_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + len(week0),
            )
            self.check_extra_row_calculations(
                base_terms,
                all_sessions,
                start_session,
                end_session,
                min_extra_rows=i + 1,
                expected_extra_rows=i + 1,
            )

    def check_extra_row_calculations(
        self,
        terms,
        all_sessions,
        start_session,
        end_session,
        min_extra_rows,
        expected_extra_rows,
    ):
        """
        Check that each term in ``terms`` computes an expected number of extra
        rows for the given parameters.
        """
        for term in terms:
            result = term.compute_extra_rows(
                all_sessions,
                start_session,
                end_session,
                min_extra_rows,
            )
            assert (
                result == expected_extra_rows
            ), "Expected {} extra_rows from {}, but got {}.".format(
                expected_extra_rows,
                term,
                result,
            )


class DownsampledPipelineTestCase(WithSeededRandomPipelineEngine, ZiplineTestCase):

    # Extend into the last few days of 2013 to test year/quarter boundaries.
    START_DATE = pd.Timestamp("2013-12-15")

    # Extend into the first few days of 2015 to test year/quarter boundaries.
    END_DATE = pd.Timestamp("2015-01-06")

    ASSET_FINDER_EQUITY_SIDS = tuple(range(10))
    DOMAIN = US_EQUITIES

    @classproperty
    def ASSET_FINDER_COUNTRY_CODE(cls):
        return cls.DOMAIN.country_code

    @classproperty
    def SEEDED_RANDOM_PIPELINE_DEFAULT_DOMAIN(cls):
        return cls.DOMAIN

    @classproperty
    def all_sessions(cls):
        return cls.DOMAIN.sessions()

    def check_downsampled_term(self, term):

        #       June 2014
        # Mo Tu We Th Fr Sa Su
        #                    1
        #  2  3  4  5  6  7  8
        #  9 10 11 12 13 14 15
        # 16 17 18 19 20 21 22
        # 23 24 25 26 27 28 29
        # 30
        all_sessions = self.all_sessions
        compute_dates = all_sessions[
            all_sessions.slice_indexer("2014-06-05", "2015-01-06")
        ]
        start_date, end_date = compute_dates[[0, -1]]

        pipe = Pipeline(
            {
                "year": term.downsample(frequency="year_start"),
                "quarter": term.downsample(frequency="quarter_start"),
                "month": term.downsample(frequency="month_start"),
                "week": term.downsample(frequency="week_start"),
            }
        )

        # Raw values for term, computed each day from 2014 to the end of the
        # target period.
        raw_term_results = self.run_pipeline(
            Pipeline({"term": term}),
            start_date=pd.Timestamp("2014-01-02"),
            end_date=pd.Timestamp("2015-01-06"),
        )["term"].unstack()

        expected_results = {
            "year": (
                raw_term_results.groupby(pd.Grouper(freq="AS"))
                .first()
                .reindex(compute_dates, method="ffill")
            ),
            "quarter": (
                raw_term_results.groupby(pd.Grouper(freq="QS"))
                .first()
                .reindex(compute_dates, method="ffill")
            ),
            "month": (
                raw_term_results.groupby(pd.Grouper(freq="MS"))
                .first()
                .reindex(compute_dates, method="ffill")
            ),
            "week": (
                raw_term_results.groupby(pd.Grouper(freq="W", label="left"))
                .first()
                .reindex(compute_dates, method="ffill")
            ),
        }

        results = self.run_pipeline(pipe, start_date, end_date)

        for frequency in expected_results:
            result = results[frequency].unstack()
            expected = expected_results[frequency]
            assert_frame_equal(result, expected)

    def test_downsample_windowed_factor(self):
        self.check_downsampled_term(
            SimpleMovingAverage(
                inputs=[TestingDataSet.float_col],
                window_length=5,
            )
        )

    def test_downsample_non_windowed_factor(self):
        sma = SimpleMovingAverage(
            inputs=[TestingDataSet.float_col],
            window_length=5,
        )

        self.check_downsampled_term(((sma + sma) / 2).rank())

    def test_downsample_windowed_filter(self):
        sma = SimpleMovingAverage(
            inputs=[TestingDataSet.float_col],
            window_length=5,
        )
        self.check_downsampled_term(All(inputs=[sma.top(4)], window_length=5))

    def test_downsample_nonwindowed_filter(self):
        sma = SimpleMovingAverage(
            inputs=[TestingDataSet.float_col],
            window_length=5,
        )
        self.check_downsampled_term(sma > 5)

    def test_downsample_windowed_classifier(self):
        class IntSumClassifier(CustomClassifier):
            inputs = [TestingDataSet.float_col]
            window_length = 8
            dtype = int64_dtype
            missing_value = -1

            def compute(self, today, assets, out, floats):
                out[:] = floats.sum(axis=0).astype(int) % 4

        self.check_downsampled_term(IntSumClassifier())

    def test_downsample_nonwindowed_classifier(self):
        sma = SimpleMovingAverage(
            inputs=[TestingDataSet.float_col],
            window_length=5,
        )
        self.check_downsampled_term(sma.quantiles(5))

    def test_errors_on_bad_downsample_frequency(self):

        f = NDaysAgoFactor(window_length=3)
        expected = (
            "{}() expected a value in "
            "('month_start', 'quarter_start', 'week_start', 'year_start') "
            "for argument 'frequency', but got 'bad' instead."
        ).format(_qualified_name(f.downsample))
        with pytest.raises(ValueError, match=re.escape(expected)):
            f.downsample("bad")


class DownsampledGBPipelineTestCase(DownsampledPipelineTestCase):
    DOMAIN = GB_EQUITIES


class DownsampledCAPipelineTestCase(DownsampledPipelineTestCase):
    DOMAIN = CA_EQUITIES


class TestDownsampledRowwiseOperation(WithAssetFinder, ZiplineTestCase):

    START_DATE = pd.Timestamp("2014-01-01")
    END_DATE = pd.Timestamp("2014-02-01")
    HALF_WAY_POINT = pd.Timestamp("2014-01-15")

    dates = pd.date_range(START_DATE, END_DATE)

    ASSET_FINDER_COUNTRY_CODE = "??"

    class SidFactor(CustomFactor):
        inputs = ()
        window_length = 1

        def compute(self, today, assets, out):
            out[:] = assets

    factor = SidFactor()

    @classmethod
    def init_class_fixtures(cls):
        super(TestDownsampledRowwiseOperation, cls).init_class_fixtures()
        cls.pipeline_engine = SimplePipelineEngine(
            get_loader=lambda c: ExplodingObject(),
            asset_finder=cls.asset_finder,
            default_domain=EquitySessionDomain(
                cls.dates,
                country_code=cls.ASSET_FINDER_COUNTRY_CODE,
            ),
        )

    @classmethod
    def make_equity_info(cls):
        start = cls.START_DATE - pd.Timedelta(days=1)
        end = cls.END_DATE
        early_end = cls.HALF_WAY_POINT
        return pd.DataFrame(
            [
                ["A", "Ayy Inc.", start, end, "E"],
                ["B", "early end", start, early_end, "E"],
                ["C", "C Inc.", start, end, "E"],
            ],
            index=[ord("A"), ord("B"), ord("C")],
            columns=(
                "symbol",
                "asset_name",
                "start_date",
                "end_date",
                "exchange",
            ),
        )

    def test_downsampled_rank(self):
        downsampled_rank = self.factor.rank().downsample("month_start")
        pipeline = Pipeline({"rank": downsampled_rank})

        results_month_start = self.pipeline_engine.run_pipeline(
            pipeline,
            self.START_DATE,
            self.END_DATE,
        )

        half_way_start = self.HALF_WAY_POINT + pd.Timedelta(days=1)
        results_halfway_start = self.pipeline_engine.run_pipeline(
            pipeline,
            half_way_start,
            self.END_DATE,
        )

        results_month_start_aligned = results_month_start.loc[half_way_start:]

        assert_frame_equal(results_month_start_aligned, results_halfway_start)
