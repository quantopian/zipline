"""
Tests for the reference loader for EarningsCalendar.
"""
from unittest import TestCase

import blaze as bz
from contextlib2 import ExitStack
from nose_parameterized import parameterized
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from six import iteritems

from zipline.pipeline import Pipeline
from zipline.pipeline.data import EarningsCalendar
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors.events import (
    BusinessDaysUntilNextEarnings,
    BusinessDaysSincePreviousEarnings,
)
from zipline.pipeline.loaders.earnings import EarningsCalendarLoader
from zipline.pipeline.loaders.blaze import (
    ANCMT_FIELD_NAME,
    BlazeEarningsCalendarLoader,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.utils.numpy_utils import make_datetime64D, np_NaT
from zipline.utils.tradingcalendar import trading_days
from zipline.utils.test_utils import (
    make_simple_equity_info,
    powerset,
    tmp_asset_finder,
)


def _to_series(knowledge_dates, earning_dates):
    """
    Helper for converting a dict of strings to a Series of datetimes.

    This is just for making the test cases more readable.
    """
    return pd.Series(
        index=pd.to_datetime(knowledge_dates),
        data=pd.to_datetime(earning_dates),
    )


def num_days_in_range(dates, start, end):
    """
    Return the number of days in `dates` between start and end, inclusive.
    """
    start_idx, stop_idx = dates.slice_locs(start, end)
    return stop_idx - start_idx


def gen_calendars():
    """
    Generate calendars to use as inputs to test_compute_latest.
    """
    start, stop = '2014-01-01', '2014-01-31'
    all_dates = pd.date_range(start, stop, tz='utc')

    # These dates are the points where announcements or knowledge dates happen.
    # Test every combination of them being absent.
    critical_dates = pd.to_datetime([
        '2014-01-05',
        '2014-01-10',
        '2014-01-15',
        '2014-01-20',
    ])
    for to_drop in map(list, powerset(critical_dates)):
        # Have to yield tuples.
        yield (all_dates.drop(to_drop),)

    # Also test with the trading calendar.
    yield (trading_days[trading_days.slice_indexer(start, stop)],)


class EarningsCalendarLoaderTestCase(TestCase):
    """
    Tests for loading the earnings announcement data.
    """
    loader_type = EarningsCalendarLoader

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        cls.sids = A, B, C, D, E = range(5)
        equity_info = make_simple_equity_info(
            cls.sids,
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )

        cls.earnings_dates = {
            # K1--K2--E1--E2.
            A: _to_series(
                knowledge_dates=['2014-01-05', '2014-01-10'],
                earning_dates=['2014-01-15', '2014-01-20'],
            ),
            # K1--K2--E2--E1.
            B: _to_series(
                knowledge_dates=['2014-01-05', '2014-01-10'],
                earning_dates=['2014-01-20', '2014-01-15']
            ),
            # K1--E1--K2--E2.
            C: _to_series(
                knowledge_dates=['2014-01-05', '2014-01-15'],
                earning_dates=['2014-01-10', '2014-01-20']
            ),
            # K1 == K2.
            D: _to_series(
                knowledge_dates=['2014-01-05'] * 2,
                earning_dates=['2014-01-10', '2014-01-15'],
            ),
            E: pd.Series(
                data=[],
                index=pd.DatetimeIndex([]),
                dtype='datetime64[ns]',
            ),
        }

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def loader_args(self, dates):
        """Construct the base earnings announcements object to pass to the
        loader.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The dates we can serve.

        Returns
        -------
        args : tuple[any]
            The arguments to forward to the loader positionally.
        """
        return dates, self.earnings_dates

    def setup(self, dates):
        """
        Make a PipelineEngine and expectation functions for the given dates
        calendar.

        This exists to make it easy to test our various cases with critical
        dates missing from the calendar.
        """
        A, B, C, D, E = self.sids

        def num_days_between(start_date, end_date):
            return num_days_in_range(dates, start_date, end_date)

        def zip_with_dates(dts):
            return pd.Series(pd.to_datetime(dts), index=dates)

        _expected_next_announce = pd.DataFrame({
            A: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-04') +
                ['2014-01-15'] * num_days_between('2014-01-05', '2014-01-15') +
                ['2014-01-20'] * num_days_between('2014-01-16', '2014-01-20') +
                ['NaT'] * num_days_between('2014-01-21', None)
            ),
            B: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-04') +
                ['2014-01-20'] * num_days_between('2014-01-05', '2014-01-09') +
                ['2014-01-15'] * num_days_between('2014-01-10', '2014-01-15') +
                ['2014-01-20'] * num_days_between('2014-01-16', '2014-01-20') +
                ['NaT'] * num_days_between('2014-01-21', None)
            ),
            C: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-04') +
                ['2014-01-10'] * num_days_between('2014-01-05', '2014-01-10') +
                ['NaT'] * num_days_between('2014-01-11', '2014-01-14') +
                ['2014-01-20'] * num_days_between('2014-01-15', '2014-01-20') +
                ['NaT'] * num_days_between('2014-01-21', None)
            ),
            D: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-04') +
                ['2014-01-10'] * num_days_between('2014-01-05', '2014-01-10') +
                ['2014-01-15'] * num_days_between('2014-01-11', '2014-01-15') +
                ['NaT'] * num_days_between('2014-01-16', None)
            ),
            E: zip_with_dates(['NaT'] * len(dates)),
        }, index=dates)

        _expected_previous_announce = pd.DataFrame({
            A: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between('2014-01-15', '2014-01-19') +
                ['2014-01-20'] * num_days_between('2014-01-20', None)
            ),
            B: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between('2014-01-15', '2014-01-19') +
                ['2014-01-20'] * num_days_between('2014-01-20', None)
            ),
            C: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between('2014-01-10', '2014-01-19') +
                ['2014-01-20'] * num_days_between('2014-01-20', None)
            ),
            D: zip_with_dates(
                ['NaT'] * num_days_between(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between('2014-01-10', '2014-01-14') +
                ['2014-01-15'] * num_days_between('2014-01-15', None)
            ),
            E: zip_with_dates(['NaT'] * len(dates)),
        }, index=dates)

        _expected_next_busday_offsets = self._compute_busday_offsets(
            _expected_next_announce
        )
        _expected_previous_busday_offsets = self._compute_busday_offsets(
            _expected_previous_announce
        )

        def expected_next_announce(sid):
            """
            Return the expected next announcement dates for ``sid``.
            """
            return _expected_next_announce[sid]

        def expected_next_busday_offset(sid):
            """
            Return the expected number of days to the next announcement for
            ``sid``.
            """
            return _expected_next_busday_offsets[sid]

        def expected_previous_announce(sid):
            """
            Return the expected previous announcement dates for ``sid``.
            """
            return _expected_previous_announce[sid]

        def expected_previous_busday_offset(sid):
            """
            Return the expected number of days to the next announcement for
            ``sid``.
            """
            return _expected_previous_busday_offsets[sid]

        loader = self.loader_type(*self.loader_args(dates))
        engine = SimplePipelineEngine(lambda _: loader, dates, self.finder)
        return (
            engine,
            expected_next_announce,
            expected_next_busday_offset,
            expected_previous_announce,
            expected_previous_busday_offset,
        )

    @staticmethod
    def _compute_busday_offsets(announcement_dates):
        """
        Compute expected business day offsets from a DataFrame of announcement
        dates.
        """
        # Column-vector of dates on which factor `compute` will be called.
        raw_call_dates = announcement_dates.index.values.astype(
            'datetime64[D]'
        )[:, None]

        # 2D array of dates containining expected nexg announcement.
        raw_announce_dates = (
            announcement_dates.values.astype('datetime64[D]')
        )

        # Set NaTs to 0 temporarily because busday_count doesn't support NaT.
        # We fill these entries with NaNs later.
        whereNaT = raw_announce_dates == np_NaT
        raw_announce_dates[whereNaT] = make_datetime64D(0)

        # The abs call here makes it so that we can use this function to
        # compute offsets for both next and previous earnings (previous
        # earnings offsets come back negative).
        expected = abs(np.busday_count(
            raw_call_dates,
            raw_announce_dates
        ).astype(float))

        expected[whereNaT] = np.nan
        return pd.DataFrame(
            data=expected,
            columns=announcement_dates.columns,
            index=announcement_dates.index,
        )

    @parameterized.expand(gen_calendars())
    def test_compute_earnings(self, dates):

        (
            engine,
            expected_next,
            expected_next_busday_offset,
            expected_previous,
            expected_previous_busday_offset,
        ) = self.setup(dates)

        pipe = Pipeline(
            columns={
                'next': EarningsCalendar.next_announcement.latest,
                'previous': EarningsCalendar.previous_announcement.latest,
                'days_to_next': BusinessDaysUntilNextEarnings(),
                'days_since_prev': BusinessDaysSincePreviousEarnings(),
            }
        )

        result = engine.run_pipeline(
            pipe,
            start_date=dates[0],
            end_date=dates[-1],
        )

        computed_next = result['next']
        computed_previous = result['previous']
        computed_next_busday_offset = result['days_to_next']
        computed_previous_busday_offset = result['days_since_prev']

        # NaTs in next/prev should correspond to NaNs in offsets.
        assert_series_equal(
            computed_next.isnull(),
            computed_next_busday_offset.isnull(),
        )
        assert_series_equal(
            computed_previous.isnull(),
            computed_previous_busday_offset.isnull(),
        )

        for sid in self.sids:

            assert_series_equal(
                computed_next.xs(sid, level=1),
                expected_next(sid),
                sid,
            )

            assert_series_equal(
                computed_previous.xs(sid, level=1),
                expected_previous(sid),
                sid,
            )

            assert_series_equal(
                computed_next_busday_offset.xs(sid, level=1),
                expected_next_busday_offset(sid),
                sid,
            )

            assert_series_equal(
                computed_previous_busday_offset.xs(sid, level=1),
                expected_previous_busday_offset(sid),
                sid,
            )


class BlazeEarningsCalendarLoaderTestCase(EarningsCalendarLoaderTestCase):
    loader_type = BlazeEarningsCalendarLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeEarningsCalendarLoaderTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                ANCMT_FIELD_NAME: earning_dates,
                TS_FIELD_NAME: earning_dates.index,
                SID_FIELD_NAME: sid,
            })
            for sid, earning_dates in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeEarningsCalendarLoaderNotInteractiveTestCase(
        BlazeEarningsCalendarLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeEarningsCalendarLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return (
            bz.symbol('expr', bound_expr.dshape),
            bound_expr._resources()[bound_expr],
        )
