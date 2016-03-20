"""
Tests for the reference loader for EarningsCalendar.
"""
from functools import partial
from unittest import TestCase

import blaze as bz
from blaze.compute.core import swap_resources_into_scope
from contextlib2 import ExitStack
import pandas as pd
from six import iteritems
from .base import EventLoaderCommonMixin

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

from zipline.testing import tmp_asset_finder

earnings_cases = [
    # K1--K2--A1--A2.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-15', '2014-01-20'])
    }),
    # K1--K2--A2--A1.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-20', '2014-01-15'])
    }),
    # K1--A1--K2--A2.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-20'])
    }),
    # K1 == K2.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-15'])
    }),
    pd.DataFrame(
        columns=[ANNOUNCEMENT_FIELD_NAME,
                 TS_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]


class EarningsCalendarLoaderTestCase(TestCase, EventLoaderCommonMixin):
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
    def get_sids(cls):
        return range(5)

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        cls.cols = {}
        cls.dataset = {sid: df for sid, df in enumerate(earnings_cases)}
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=cls.get_equity_info()),
        )

        cls.loader_type = EarningsCalendarLoader

    def get_expected_next_event_dates(self, dates):
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)
        return pd.DataFrame({
            0: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-15'] *
                num_days_between_for_dates('2014-01-05', '2014-01-15') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-16', '2014-01-20') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-21', None)
            ),
            1: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-05', '2014-01-09') +
                ['2014-01-15'] *
                num_days_between_for_dates('2014-01-10', '2014-01-15') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-16', '2014-01-20') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-21', None)
            ),
            2: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-10'] *
                num_days_between_for_dates('2014-01-05', '2014-01-10') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-11', '2014-01-14') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-15', '2014-01-20') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-21', None)
            ),
            3: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-10'] *
                num_days_between_for_dates('2014-01-05', '2014-01-10') +
                ['2014-01-15'] *
                num_days_between_for_dates('2014-01-11', '2014-01-15') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-16', None)
            ),
            4: zip_with_dates_for_dates(['NaT'] *
                                        len(dates)),
        }, index=dates)

    def get_expected_previous_event_dates(self, dates):
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)
        return pd.DataFrame({
            0: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            1: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            2: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between_for_dates('2014-01-10',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            3: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between_for_dates('2014-01-10',
                                                            '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            None),
            ),
            4: zip_with_dates_for_dates(['NaT'] * len(dates)),
        }, index=dates)

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        _expected_next_announce = self.get_expected_next_event_dates(dates)

        _expected_previous_announce = self.get_expected_previous_event_dates(
            dates
        )

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
        return (bz.data(pd.concat(
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
        super(BlazeEarningsCalendarLoaderNotInteractiveTestCase,
              cls).setUpClass()
        cls.loader_type = BlazeEarningsCalendarLoader

    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeEarningsCalendarLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
