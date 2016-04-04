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
from zipline.pipeline.loaders.utils import (
    get_values_for_date_ranges,
    zip_with_dates
)
from zipline.testing.fixtures import (
    WithPipelineEventDataLoader,
    ZiplineTestCase
)

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

next_date_intervals = [
    [[None, '2014-01-04'],
     ['2014-01-05', '2014-01-15'],
     ['2014-01-16', '2014-01-20'],
     ['2014-01-21', None]],
    [[None, '2014-01-04'],
     ['2014-01-05', '2014-01-09'],
     ['2014-01-10', '2014-01-15'],
     ['2014-01-16', '2014-01-20'],
     ['2014-01-21', None]],
    [[None, '2014-01-04'],
     ['2014-01-05', '2014-01-10'],
     ['2014-01-11', '2014-01-14'],
     ['2014-01-15', '2014-01-20'],
     ['2014-01-21', None]],
    [[None, '2014-01-04'],
     ['2014-01-05', '2014-01-10'],
     ['2014-01-11', '2014-01-15'],
     ['2014-01-16', None]]
]

next_dates = [
    ['NaT', '2014-01-15', '2014-01-20', 'NaT'],
    ['NaT', '2014-01-20', '2014-01-15', '2014-01-20', 'NaT'],
    ['NaT', '2014-01-10', 'NaT', '2014-01-20', 'NaT'],
    ['NaT', '2014-01-10', '2014-01-15', 'NaT'],
    ['NaT']
]

prev_date_intervals = [
    [[None, '2014-01-14'],
     ['2014-01-15', '2014-01-19'],
     ['2014-01-20', None]],
    [[None, '2014-01-14'],
     ['2014-01-15', '2014-01-19'],
     ['2014-01-20', None]],
    [[None, '2014-01-09'],
     ['2014-01-10', '2014-01-19'],
     ['2014-01-20', None]],
    [[None, '2014-01-09'],
     ['2014-01-10', '2014-01-14'],
     ['2014-01-15', None]]
]

prev_dates = [
    ['NaT', '2014-01-15', '2014-01-20'],
    ['NaT', '2014-01-15', '2014-01-20'],
    ['NaT', '2014-01-10', '2014-01-20'],
    ['NaT', '2014-01-10', '2014-01-15'],
    ['NaT']
]


class EarningsCalendarLoaderTestCase(WithPipelineEventDataLoader,
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
        return {sid: df for sid, df in enumerate(earnings_cases)}

    loader_type = EarningsCalendarLoader

    def get_expected_next_event_dates(self, dates):
        return pd.DataFrame({
            0: get_values_for_date_ranges(zip_with_dates,
                                          next_dates[0],
                                          next_date_intervals[0],
                                          dates),
            1: get_values_for_date_ranges(zip_with_dates,
                                          next_dates[1],
                                          next_date_intervals[1],
                                          dates),
            2: get_values_for_date_ranges(zip_with_dates,
                                          next_dates[2],
                                          next_date_intervals[2],
                                          dates),
            3: get_values_for_date_ranges(zip_with_dates,
                                          next_dates[3],
                                          next_date_intervals[3],
                                          dates),
            4: zip_with_dates(dates, ['NaT'] * len(dates)),
        }, index=dates)

    def get_expected_previous_event_dates(self, dates):
        return pd.DataFrame({
            0: get_values_for_date_ranges(zip_with_dates,
                                          prev_dates[0],
                                          prev_date_intervals[0],
                                          dates),
            1: get_values_for_date_ranges(zip_with_dates,
                                          prev_dates[1],
                                          prev_date_intervals[1],
                                          dates),
            2: get_values_for_date_ranges(zip_with_dates,
                                          prev_dates[2],
                                          prev_date_intervals[2],
                                          dates),
            3: get_values_for_date_ranges(zip_with_dates,
                                          prev_dates[3],
                                          prev_date_intervals[3],
                                          dates),
            4: zip_with_dates(dates, ['NaT'] * len(dates)),
        }, index=dates)

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
        cols = {}
        cols[PREVIOUS_ANNOUNCEMENT] = _expected_previous_announce
        cols[NEXT_ANNOUNCEMENT] = _expected_next_announce
        cols[DAYS_TO_NEXT] = _expected_next_busday_offsets
        cols[DAYS_SINCE_PREV] = _expected_previous_busday_offsets
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
