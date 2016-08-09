from itertools import product
import numpy as np
import pandas as pd
from zipline.pipeline import SimplePipelineEngine, Pipeline

from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.loaders.quarter_estimates import \
    NextQuartersEstimatesLoader, PreviousQuartersEstimatesLoader
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithAssetFinder, WithTradingSessions
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype


class Estimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate = Column(dtype=float64_dtype)
    value = Column(dtype=float64_dtype)


def QuartersEstimates(num_qtr):
    class QtrEstimates(Estimates):
        num_quarters = num_qtr
        name=Estimates
    return QtrEstimates

# Final release dates never change
releases = pd.DataFrame({
    'sid': [1, 1],
    'timestamp': [pd.Timestamp('2015-01-20'), pd.Timestamp('2015-4-20')],
    'event_date': [pd.Timestamp('2015-01-20'), pd.Timestamp('2015-04-20')],
    'estimate': [0.5, 0.8],
    'value': [0.6, 0.9],
    'fiscal_quarter': [1, 2],
    'fiscal_year': [2015, 2015]
})

estimates = pd.DataFrame({
    'sid': [1, 1, 1, 1],
    'timestamp': [pd.Timestamp('2015-01-02'),
                  pd.Timestamp('2015-01-10'),
                  pd.Timestamp('2015-04-02'),
                  pd.Timestamp('2015-4-10')],
    'event_date': [pd.Timestamp('2015-01-20'),
                   pd.Timestamp('2015-01-20'),
                   pd.Timestamp('2015-04-20'),
                   pd.Timestamp('2015-04-20')],
    'estimate': [.1, .2, .3, .4],
    'value': [np.NaN, np.NaN, np.NaN, np.NaN],
    'fiscal_quarter': [1, 1, 2, 2],
    'fiscal_year': [2015, 2015, 2015, 2015]
})

events = pd.concat([releases, estimates])


class NextEstimateTestCase(WithAssetFinder,
                           WithTradingSessions,
                           ZiplineTestCase):
    START_DATE = pd.Timestamp('2015-01-01')
    END_DATE = pd.Timestamp('2015-04-30')

    @classmethod
    def make_loader(cls, events, columns):
        return NextQuartersEstimatesLoader(events, columns)

    @classmethod
    def init_class_fixtures(cls):
        cls.events = events
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
        super(NextEstimateTestCase, cls).init_class_fixtures()

    def test_regular(self):
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
        sid_events = results.xs(1, level=1)
        ed_sorted_events = self.events.sort(['event_date', 'timestamp'])
        for i, date in enumerate(sid_events.index):
            # Get all upcoming events that we know about on 'date'
            eligible_timestamps = ed_sorted_events[ed_sorted_events['timestamp']
                                                <= date]
            eligible_events = eligible_timestamps[eligible_timestamps['event_date'] >= date]
            if not eligible_events.empty:
                smallest_event_date = eligible_events.iloc[0]['event_date']
                expected_event = eligible_events[eligible_events['event_date'] == smallest_event_date].iloc[-1]
                for colname in sid_events.columns:
                    expected_value = expected_event[colname]
                    computed_value = sid_events.iloc[i][colname]
                    assert_equal(expected_value, computed_value)
            else:
                assert sid_events.iloc[i].isnull().all()


class PreviousEstimateTestCase(WithAssetFinder,
                               WithTradingSessions,
                               ZiplineTestCase):
    START_DATE = pd.Timestamp('2015-01-01')
    END_DATE = pd.Timestamp('2015-04-30')

    @classmethod
    def make_loader(cls, events, columns):
        return PreviousQuartersEstimatesLoader(events, columns)

    @classmethod
    def init_class_fixtures(cls):
        cls.events = events
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
        super(PreviousEstimateTestCase, cls).init_class_fixtures()

    def test_regular(self):
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
        sid_events = results.xs(1, level=1)
        ed_sorted_events = self.events.sort(['event_date', 'timestamp'])
        for i, date in enumerate(sid_events.index):
            expected_event = ed_sorted_events[ed_sorted_events['event_date'] <=
                                 date].iloc[-1]
            if not expected_event.empty:
                for colname in sid_events.columns:
                    expected_value = expected_event[colname]
                    computed_value = sid_events.iloc[i][colname]
                    assert_equal(expected_value, computed_value)
            else:
                assert sid_events.iloc[i].isnull().all()
