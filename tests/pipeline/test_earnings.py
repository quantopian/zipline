"""
Tests for the reference loader for EarningsCalendar.
"""
from unittest import TestCase

import blaze as bz
from blaze.compute.core import swap_resources_into_scope
from contextlib2 import ExitStack
from nose_parameterized import parameterized
import pandas as pd
from six import iteritems
from tests.pipeline.test_events import EventLoaderCommonTest, param_dates

from zipline.pipeline.common import (
    ANNOUNCEMENT_FIELD_NAME,
    DAYS_SINCE_PREV,
    DAYS_TO_NEXT,
    NEXT_ANNOUNCEMENT,
    PREVIOUS_ANNOUNCEMENT,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)
from zipline.pipeline.data import EarningsCalendar
from zipline.pipeline.factors.events import (
    BusinessDaysSincePreviousEarnings,
    BusinessDaysUntilNextEarnings,
)
from zipline.pipeline.loaders.earnings import EarningsCalendarLoader
from zipline.pipeline.loaders.blaze import (
    BlazeEarningsCalendarLoader,
)
from zipline.utils.test_utils import (
    make_simple_equity_info,
    tmp_asset_finder,
)

earnings_dates = [
            # K1--K2--E1--E2.
            pd.DataFrame({
                TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
                ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-15',
                                                         '2014-01-20'])
            }),
            # K1--K2--E2--E1.
            pd.DataFrame({
                TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
                ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-20',
                                                         '2014-01-15'])
            }),
            # K1--E1--K2--E2.
            pd.DataFrame({
                TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
                ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-10',
                                                         '2014-01-20'])
            }),
            # K1 == K2.
            pd.DataFrame({
                TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
                ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-10',
                                                         '2014-01-15'])
            }),
            pd.DataFrame({
                TS_FIELD_NAME: pd.to_datetime([]),
                ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([])
            })
        ]


class EarningsCalendarLoaderTestCase(TestCase, EventLoaderCommonTest):
    """
    Tests for loading the earnings announcement data.
    """
    pipeline_columns = {
        NEXT_ANNOUNCEMENT: EarningsCalendar.next_announcement.latest,
        PREVIOUS_ANNOUNCEMENT: EarningsCalendar.previous_announcement.latest,
        DAYS_SINCE_PREV: BusinessDaysSincePreviousEarnings(),
        DAYS_TO_NEXT: BusinessDaysUntilNextEarnings(),
    }

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        equity_info = make_simple_equity_info(
            cls.sids,
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )
        cls.cols = {}
        cls.dataset = {sid: df for sid, df in enumerate(earnings_dates)}
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )

        cls.loader_type = EarningsCalendarLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()


    def setup(self, dates):
        _expected_next_announce = self.get_expected_next_event_dates(dates)

        _expected_previous_announce = self.get_expected_previous_event_dates(dates)

        _expected_next_busday_offsets = self._compute_busday_offsets(
            _expected_next_announce
        )
        _expected_previous_busday_offsets = self._compute_busday_offsets(
            _expected_previous_announce
        )
        self.cols[PREVIOUS_ANNOUNCEMENT] = _expected_previous_announce
        self.cols[NEXT_ANNOUNCEMENT] = _expected_next_announce
        self.cols[DAYS_TO_NEXT] = _expected_next_busday_offsets
        self.cols[DAYS_SINCE_PREV] = _expected_previous_busday_offsets

    @parameterized.expand(param_dates)
    def test_compute_earnings(self, dates):
        self._test_compute(dates)


class BlazeEarningsCalendarLoaderTestCase(EarningsCalendarLoaderTestCase):
    @classmethod
    def setUpClass(cls):
        super(BlazeEarningsCalendarLoaderTestCase, cls).setUpClass()
        cls.loader_type = BlazeEarningsCalendarLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeEarningsCalendarLoaderTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                ANNOUNCEMENT_FIELD_NAME: df[ANNOUNCEMENT_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeEarningsCalendarLoaderNotInteractiveTestCase(
        BlazeEarningsCalendarLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    @classmethod
    def setUpClass(cls):
        super(BlazeEarningsCalendarLoaderNotInteractiveTestCase, cls).setUpClass()
        cls.loader_type = BlazeEarningsCalendarLoader

    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeEarningsCalendarLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
