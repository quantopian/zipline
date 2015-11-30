"""
Tests for the reference loader for EarningsCalendar.
"""
from unittest import TestCase

from contextlib2 import ExitStack
import pandas as pd

from zipline.pipeline import Pipeline
from zipline.pipeline.data import EarningsCalendar
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import EarningsCalendarLoader
from zipline.utils.test_utils import tmp_asset_finder


class DateLoaderTestCase(TestCase):
    """
    Tests for loading adjusted_arrays with datetime data.
    """

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        finder = stack.enter_context(tmp_asset_finder())
        A, B, C = finder.sids
        calendar = pd.date_range('2014', '2014-01-31')
        announcement_dates = {
            A: pd.DatetimeIndex(['2014-01-01', '2014-01-15', '2014-02-01']),
            # Last 10 entries for B are trailing NaTs.
            B: pd.DatetimeIndex(['2014-01-10', '2014-01-15', '2014-01-20']),
            C: pd.DatetimeIndex(['2014-01-10', '2014-01-20', '2014-02-01']),
        }
        loader = EarningsCalendarLoader(calendar, announcement_dates)
        cls.engine = SimplePipelineEngine(lambda _: loader, calendar, finder)

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def test_compute_latest(self):
        p = Pipeline(
            columns={
                'next': EarningsCalendar.next_announcement,
                'previous': EarningsCalendar.previous_announcement,
            }
        )
