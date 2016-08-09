import itertools
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from zipline.pipeline import SimplePipelineEngine, Pipeline

from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.loaders.quarter_estimates import (
    NextQuartersEstimatesLoader,
    PreviousQuartersEstimatesLoader
)
from zipline.pipeline.loaders.quarter_estimates import (
    calc_forward_shift,
    calc_backward_shift
)
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithAssetFinder, WithTradingSessions
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype
import line_profiler
prof = line_profiler.LineProfiler()


class Estimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate = Column(dtype=float64_dtype)
    value = Column(dtype=float64_dtype)


def QuartersEstimates(num_qtr):
    class QtrEstimates(Estimates):
        num_quarters = num_qtr
        name = Estimates
    return QtrEstimates

# Final release dates never change. The quarters have very tight date ranges
# in order to reduce the number of dates we need to iterate through when
# testing.
releases = pd.DataFrame({
    'timestamp': [pd.Timestamp('2015-01-15'), pd.Timestamp('2015-01-31')],
    'event_date': [pd.Timestamp('2015-01-15'), pd.Timestamp('2015-01-31')],
    'estimate': [0.5, 0.8],
    'value': [0.6, 0.9],
    'fiscal_quarter': [1.0, 2.0],
    'fiscal_year': [2015.0, 2015.0]
})

q1_knowledge_dates = [pd.Timestamp('2015-01-01'), pd.Timestamp('2015-01-04'),
                      pd.Timestamp('2015-01-08'), pd.Timestamp('2015-01-12')]
q2_knowledge_dates = [pd.Timestamp('2015-01-16'), pd.Timestamp('2015-01-20'),
                      pd.Timestamp('2015-01-24'), pd.Timestamp('2015-01-28')]
# We want to model the possibility of an estimate predicting a release date
# that gets shifted forward/backward.
q1_release_dates = [pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-15')]
q2_release_dates = [pd.Timestamp('2015-01-28'), pd.Timestamp('2015-01-30')]
estimates = pd.DataFrame({
    'estimate': [.1, .2, .3, .4],
    'value': [np.NaN, np.NaN, np.NaN, np.NaN],
    'fiscal_quarter': [1.0, 1.0, 2.0, 2.0],
    'fiscal_year': [2015.0, 2015.0, 2015.0, 2015.0]
})


def gen_estimates():
    sid_estimates = []
    sid_releases = []
    release_dates = list(itertools.product(q1_release_dates, q2_release_dates))
    knowledge_permutations = list(itertools.permutations(q1_knowledge_dates +
                                                         q2_knowledge_dates,
                                                         4))
    all_permutations = itertools.product(knowledge_permutations,
                                         release_dates)
    for sid, ((q1e1, q1e2, q2e1, q2e2), (rd1, rd2)) in enumerate(
            all_permutations):
        # We're assuming that estimates must come before the relevant release.
        if q1e1 < q1e2 and q2e1 < q2e2 and q1e1 < rd1 and q1e2 < \
                rd2:
            sid_estimate = estimates.copy(True)
            sid_estimate['timestamp'] = [q1e1, q1e2, q2e1, q2e2]
            sid_estimate['event_date'] = [rd1]*2 + [rd2] * 2
            sid_estimate['sid'] = sid
            sid_estimates += [sid_estimate]
            sid_release = releases.copy(True)
            sid_release['sid'] = sid_estimate['sid']
            sid_releases += [sid_release]

    return pd.concat(sid_estimates + sid_releases).reset_index(drop=True)


class EstimateTestCase(WithAssetFinder,
                       WithTradingSessions,
                       ZiplineTestCase):
    START_DATE = pd.Timestamp('2014-12-28')
    END_DATE = pd.Timestamp('2015-02-03')

    @classmethod
    def make_loader(cls, events, columns):
        pass

    @classmethod
    def init_class_fixtures(cls):
        cls.events = gen_estimates()
        cls.sids = cls.events['sid'].unique()
        cls.columns = {
            Estimates.estimate: 'estimate',
            Estimates.event_date: 'event_date',
            Estimates.fiscal_quarter: 'fiscal_quarter',
            Estimates.fiscal_year: 'fiscal_year',
            Estimates.value: 'value',
        }
        cls.loader = cls.make_loader(
            events=cls.events,
            columns=cls.columns
        )
        cls.ASSET_FINDER_EQUITY_SIDS = list(cls.events['sid'].unique())
        cls.ASSET_FINDER_EQUITY_SYMBOLS = [
            's' + str(n) for n in cls.ASSET_FINDER_EQUITY_SIDS
        ]
        super(EstimateTestCase, cls).init_class_fixtures()


class NextEstimateTestCase(EstimateTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return NextQuartersEstimatesLoader(events, columns)

    #@profile
    def test_next_estimates(self):
        """
        The goal of this test is to make sure that we select the right
        datapoint as our 'next' w.r.t each date.
        """
        dataset = QuartersEstimates(1)
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )

        results = engine.run_pipeline(
            Pipeline({c.name: c.latest for c in dataset.columns}),
            start_date=self.trading_days[0],
            end_date=self.trading_days[-1],
        )
        for sid in self.sids:
            sid_events = results.xs(sid, level=1)
            ed_sorted_events = self.events[
                self.events['sid'] == sid
            ]
            ed_sorted_events['key'] = 1
            all_dates = pd.DataFrame({'all_dates': sid_events.index})
            all_dates['key'] = 1
            crossproduct = pd.merge(all_dates, ed_sorted_events, on='key')
            crossproduct = crossproduct[crossproduct['timestamp'] <=
                                        crossproduct['all_dates']]
            crossproduct = crossproduct[crossproduct['event_date'] >=
                                        crossproduct['all_dates']]
            final = crossproduct.sort_values(by=['all_dates',
                                                 'event_date',
                                                 'timestamp'],
                                             ascending=[True, True,
                                                        False]).groupby([
                'all_dates', 'sid']).first().reset_index()
            final = pd.merge(final, all_dates,
                             how='right').sort_values(by='all_dates').set_index(
                'all_dates')
            final.index.name = None
            for colname in sid_events.columns:
                assert_series_equal(final[colname], sid_events[colname])


class PreviousEstimateTestCase(EstimateTestCase):
    @classmethod
    def make_loader(cls, events, columns):
        return PreviousQuartersEstimatesLoader(events, columns)

    def test_previous_estimates(self):
        """
        The goal of this test is to make sure that we select the right
        datapoint as our 'previous' w.r.t each date.
        """
        dataset = QuartersEstimates(1)
        engine = SimplePipelineEngine(
            lambda x: self.loader,
            self.trading_days,
            self.asset_finder,
        )

        results = engine.run_pipeline(
            Pipeline({c.name: c.latest for c in dataset.columns}),
            start_date=self.trading_days[0],
            end_date=self.trading_days[-1],
        )
        for sid in self.sids:
            sid_events = results.xs(sid, level=1)
            ed_sorted_events = self.events[
                self.events['sid'] == sid
            ].sort_values(by=['event_date', 'timestamp'])
            for i, date in enumerate(sid_events.index):
                # Filter for events that happened on or before the simulation
                # date and that we knew about on or before the simulation date.
                ed_eligible_events = ed_sorted_events[ed_sorted_events['event_date'] <= date]
                ts_eligible_events = ed_eligible_events[ed_eligible_events['timestamp'] <= date]
                if not ts_eligible_events.empty:
                    # The expected event is the one we knew about last.
                    expected_event = ts_eligible_events.iloc[-1]
                    for colname in sid_events.columns:
                        expected_value = expected_event[colname]
                        computed_value = sid_events.iloc[i][colname]
                        assert_equal(expected_value, computed_value)
                else:
                    assert sid_events.iloc[i].isnull().all()


class QuarterShiftTestCase(ZiplineTestCase):
    """
    This tests, in isolation, quarter calculation logic for shifting quarters
    backwards/forwards from a starting point.
    """
    def test_calc_forward_shift(self):
        input_yrs = pd.Series([0] * 4)
        input_qtrs = pd.Series(range(1, 5))
        expected = pd.DataFrame(([yr, qtr] for yr in range(0, 4) for qtr
                                 in range(1, 5)))
        for i in range(0, 8):
            years, quarters = calc_forward_shift(input_yrs, input_qtrs, i)
            # Can't use assert_series_equal here with check_names=False
            # because that still fails due to name differences.
            assert years.equals(expected[i:i+4].reset_index(drop=True)[0])
            assert quarters.equals(expected[i:i+4].reset_index(drop=True)[1])


    def test_calc_backward_shift(self):
        input_yrs = pd.Series([0] * 4)
        input_qtrs = pd.Series(range(4, 0, -1))
        expected = pd.DataFrame(([yr, qtr] for yr in range(0, -4, -1) for qtr
                                 in range(4, 0, -1)))
        for i in range(0, 8):
            years, quarters = calc_backward_shift(input_yrs, input_qtrs, i)
            # Can't use assert_series_equal here with check_names=False
            # because that still fails due to name differences.
            assert years.equals(expected[i:i+4].reset_index(drop=True)[0])
            assert quarters.equals(expected[i:i+4].reset_index(drop=True)[1])
