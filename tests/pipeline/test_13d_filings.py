"""
Tests for the reference loader for 13d filings.
"""
import pandas as pd

from zipline.pipeline.common import(
    DAYS_SINCE_PREV_DISCLOSURE,
    DISCLOSURE_DATE,
    NUM_SHARES,
    PERCENT_SHARES,
    PREVIOUS_NUM_SHARES,
    PREVIOUS_PERCENT_SHARES,
    PREVIOUS_DISCLOSURE_DATE,
    TS_FIELD_NAME,
)
from zipline.pipeline.data import _13DFilings
from zipline.pipeline.factors.events import BusinessDaysSince13DFilingsDate
from zipline.pipeline.loaders._13d_filings import _13DFilingsLoader
from zipline.pipeline.loaders.utils import (
    get_values_for_date_ranges,
    zip_with_floats,
    zip_with_dates
)
from zipline.testing.fixtures import WithPipelineEventDataLoader
from zipline.testing.fixtures import ZiplineTestCase

date_intervals = [[None, '2014-01-04'],
                  ['2014-01-05', '2014-01-09'],
                  ['2014-01-10', None]]

empty_df = pd.DataFrame(
    columns=[NUM_SHARES,
             PERCENT_SHARES,
             DISCLOSURE_DATE,
             TS_FIELD_NAME],
)

empty_df[NUM_SHARES] = empty_df[NUM_SHARES].astype('float')
empty_df[PERCENT_SHARES] = empty_df[PERCENT_SHARES].astype('float')
empty_df[TS_FIELD_NAME] = empty_df[TS_FIELD_NAME].astype('datetime64[ns]')
empty_df[DISCLOSURE_DATE] = empty_df[DISCLOSURE_DATE].astype('datetime64[ns]')

_13d_filngs_cases = [
    pd.DataFrame({
        NUM_SHARES: [1, 15],
        PERCENT_SHARES: [10, 20],
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        DISCLOSURE_DATE: pd.to_datetime(['2014-01-04', '2014-01-09'])
    }),
    empty_df
]


def get_expected_previous_values(zip_date_index_with_vals,
                                 vals,
                                 date_intervals,
                                 dates):
    return pd.DataFrame({
        0: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals,
                                      date_intervals,
                                      dates),
        1: zip_date_index_with_vals(dates, ['NaN'] * len(dates)),
    }, index=dates)


class _13DFilingsLoaderTestCase(WithPipelineEventDataLoader,
                                ZiplineTestCase):
    """
    Test for _13_filings dataset.
    """
    pipeline_columns = {
        PREVIOUS_NUM_SHARES:
            _13DFilings.number_shares.latest,
        PREVIOUS_PERCENT_SHARES:
            _13DFilings.percent_shares.latest,
        PREVIOUS_DISCLOSURE_DATE:
            _13DFilings.disclosure_date.latest,
        DAYS_SINCE_PREV_DISCLOSURE:
            BusinessDaysSince13DFilingsDate(),
    }

    @classmethod
    def get_sids(cls):
        return range(2)

    @classmethod
    def get_dataset(cls):
        return {sid: frame
                for sid, frame
                in enumerate(_13d_filngs_cases)}

    loader_type = _13DFilingsLoader

    def setup(self, dates):
        cols = {}
        cols[
            PREVIOUS_DISCLOSURE_DATE
        ] = get_expected_previous_values(zip_with_dates,
                                         ['NaT', '2014-01-04', '2014-01-09'],
                                         date_intervals, dates)
        cols[PREVIOUS_NUM_SHARES] = get_expected_previous_values(
            zip_with_floats, ['NaN', 1, 15], date_intervals, dates
        )
        cols[PREVIOUS_PERCENT_SHARES] = get_expected_previous_values(
            zip_with_floats, ['NaN', 10, 20], date_intervals, dates
        )
        cols[DAYS_SINCE_PREV_DISCLOSURE] = self._compute_busday_offsets(
            cols[PREVIOUS_DISCLOSURE_DATE]
        )
        return cols
