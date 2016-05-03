"""
Tests for the reference loader for EarningsCalendar.
"""
import blaze as bz
from blaze.compute.core import swap_resources_into_scope
import pandas as pd
from six import iteritems

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
from zipline.pipeline.loaders.blaze import BlazeEarningsCalendarLoader
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithNextAndPreviousEventDataLoader
)


class EarningsCalendarLoaderTestCase(WithNextAndPreviousEventDataLoader,
                                     ZiplineTestCase):
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
    def get_dataset(cls):
        return {sid: df.rename(
            columns={'other_date': ANNOUNCEMENT_FIELD_NAME}
        ) for sid, df in enumerate(cls.base_cases)}

    loader_type = EarningsCalendarLoader

    def setup(self, dates):
        cols = {
            PREVIOUS_ANNOUNCEMENT: self.get_expected_previous_event_dates(
                dates,
                'datetime64[ns]', 'NaN'
            ),
            NEXT_ANNOUNCEMENT: self.get_expected_next_event_dates(
                dates, 'datetime64[ns]', 'NaN'),
        }
        cols[DAYS_TO_NEXT] = self._compute_busday_offsets(
            cols[NEXT_ANNOUNCEMENT]
        )
        cols[DAYS_SINCE_PREV] = self._compute_busday_offsets(
            cols[PREVIOUS_ANNOUNCEMENT]
        )
        return cols


class BlazeEarningsCalendarLoaderTestCase(EarningsCalendarLoaderTestCase):
    loader_type = BlazeEarningsCalendarLoader

    def pipeline_event_loader_args(self, dates):
        _, mapping = super(
            BlazeEarningsCalendarLoaderTestCase,
            self,
        ).pipeline_event_loader_args(dates)
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

    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeEarningsCalendarLoaderNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
