"""
Tests for the reference loader for 13d filings.
"""
from unittest import TestCase

from contextlib2 import ExitStack
import pandas as pd

from .base import EventLoaderCommonMixin
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
from zipline.testing import tmp_asset_finder

date_intervals = [[None, '2014-01-04'], ['2014-01-05', '2014-01-09'],
                  ['2014-01-10', None]]

_13d_filngs_cases = [
    pd.DataFrame({
        NUM_SHARES: [1, 15],
        PERCENT_SHARES: [10, 20],
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        DISCLOSURE_DATE: pd.to_datetime(['2014-01-04', '2014-01-09'])
    }),
    pd.DataFrame(
        columns=[NUM_SHARES,
                 PERCENT_SHARES,
                 DISCLOSURE_DATE,
                 TS_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]


def get_expected_previous_values(zip_date_index_with_vals,
                                 dates,
                                 vals_for_date_intervals):
    return pd.DataFrame({
        0: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals_for_date_intervals,
                                      date_intervals,
                                      dates),
        1: zip_date_index_with_vals(dates, ['NaN'] * len(dates)),
    }, index=dates)


class _13DFilingsLoaderTestCase(TestCase, EventLoaderCommonMixin):
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
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=cls.get_equity_info()),
        )
        cls.cols = {}
        cls.dataset = {sid:
                       frame
                       for sid, frame
                       in enumerate(_13d_filngs_cases)}
        cls.loader_type = _13DFilingsLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        _expected_previous_num_shares = get_expected_previous_values(
            zip_with_floats, dates,
            ['NaN', 1, 15]
        )
        _expected_previous_percent_shares = get_expected_previous_values(
            zip_with_floats, dates,
            ['NaN', 10, 20]
        )
        self.cols[
            PREVIOUS_DISCLOSURE_DATE
        ] = get_expected_previous_values(zip_with_dates, dates,
                                         ['NaT', '2014-01-04', '2014-01-09'])
        self.cols[PREVIOUS_NUM_SHARES] = _expected_previous_num_shares
        self.cols[PREVIOUS_PERCENT_SHARES] = _expected_previous_percent_shares
        self.cols[DAYS_SINCE_PREV_DISCLOSURE] = self._compute_busday_offsets(
            self.cols[PREVIOUS_DISCLOSURE_DATE]
        )
