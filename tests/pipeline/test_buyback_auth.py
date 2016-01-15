"""
Tests for the reference loader for EarningsCalendar.
"""
from unittest import TestCase

import blaze as bz
from blaze.compute.core import swap_resources_into_scope
from contextlib2 import ExitStack
from nose_parameterized import parameterized
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from six import iteritems

from zipline.pipeline import Pipeline
from zipline.pipeline.data import (CashBuybackAuthorizations,
                                   ShareBuybackAuthorizations)
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors.events import (
    BusinessDaysSincePreviousCashBuybackAuth,
    BusinessDaysSincePreviousShareBuybackAuth
)
from zipline.pipeline.loaders.buyback_auth import \
    CashBuybackAuthorizationsLoader, ShareBuybackAuthorizationsLoader
from zipline.pipeline.loaders.blaze import (
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    CashBuybackAuthorizationsLoader,
    SHARE_COUNT_FIELD_NAME,
    SID_FIELD_NAME,
    ShareBuybackAuthorizationsLoader,
    TS_FIELD_NAME,
    VALUE_FIELD_NAME
)
from zipline.utils.numpy_utils import make_datetime64D, np_NaT
from zipline.utils.test_utils import (
    make_simple_equity_info,
    tmp_asset_finder,
    gen_calendars,
    num_days_in_range,
)


sids = A, B, C, D, E = range(5)

equity_info = make_simple_equity_info(
            sids,
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )

buyback_authorizations = {
            # K1--K2--A1--A2--SC1--SC2--V1--V2.
            A: pd.DataFrame({
                "timestamp": pd.to_datetime(['2014-01-05', '2014-01-10']),
                BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-15',
                                                     '2014-01-20']),
                SHARE_COUNT_FIELD_NAME: [1, 15],
                VALUE_FIELD_NAME: [10, 20]
            }),
            # K1--K2--E2--E1.
            B: pd.DataFrame({
                "timestamp": pd.to_datetime(['2014-01-05', '2014-01-10']),
                BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
                    '2014-01-20', '2014-01-15']),
                SHARE_COUNT_FIELD_NAME: [7, 13], VALUE_FIELD_NAME: [10, 22]
            }),
            # K1--E1--K2--E2.
            C: pd.DataFrame({
                "timestamp": pd.to_datetime(['2014-01-05', '2014-01-15']),
                BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
                    '2014-01-10', '2014-01-20']),
                SHARE_COUNT_FIELD_NAME: [3, 1],
                VALUE_FIELD_NAME: [4, 7]
            }),
            # K1 == K2.
            D: pd.DataFrame({
                "timestamp": pd.to_datetime(['2014-01-05'] * 2),
                BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
                    '2014-01-10', '2014-01-15']),
                SHARE_COUNT_FIELD_NAME: [6, 23],
                VALUE_FIELD_NAME: [1, 2]
            }),
            E: pd.DataFrame(
                columns=["timestamp",
                         BUYBACK_ANNOUNCEMENT_FIELD_NAME,
                         SHARE_COUNT_FIELD_NAME,
                         VALUE_FIELD_NAME],
                dtype='datetime64[ns]'
            ),
        }

param_dates = gen_calendars(
        '2014-01-01',
        '2014-01-31',
        critical_dates=pd.to_datetime([
            '2014-01-05',
            '2014-01-10',
            '2014-01-15',
            '2014-01-20',
        ]),
    )


def zip_with_floats(flts, dates):
    return pd.Series(flts, index=dates).astype('float')


def num_days_between(dates, start_date, end_date):
    return num_days_in_range(dates, start_date, end_date)


def zip_with_dates(dts, dates):
    return pd.Series(pd.to_datetime(dts), index=dates)


class BuybackAuthLoaderTestCase(TestCase):
    """
    Tests for loading the earnings announcement data.
    """

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()

        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )
        cls.cols = {}
        cls.buyback_authorizations = None


    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def loader_args(self, dates):
        """Construct the base buyback authorizations object to pass to the
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
        return dates, self.buyback_authorizations

    def setup(self, dates):
        """
        Make a PipelineEngine and expectation functions for the given dates
        calendar.

        This exists to make it easy to test our various cases with critical
        dates missing from the calendar.
        """

        _expected_previous_buyback_announcement = pd.DataFrame({
            A: zip_with_dates(
                ['NaT'] * num_days_between(dates, None, '2014-01-14') +
                ['2014-01-15'] * num_days_between(dates, '2014-01-15', '2014-01-19') +
                ['2014-01-20'] * num_days_between(dates, '2014-01-20', None),
                dates
            ),
            B: zip_with_dates(
                ['NaT'] * num_days_between(dates, None, '2014-01-14') +
                ['2014-01-15'] * num_days_between(dates, '2014-01-15', '2014-01-19') +
                ['2014-01-20'] * num_days_between(dates, '2014-01-20', None),
                dates
            ),
            C: zip_with_dates(
                ['NaT'] * num_days_between(dates, None, '2014-01-09') +
                ['2014-01-10'] * num_days_between(dates, '2014-01-10', '2014-01-19') +
                ['2014-01-20'] * num_days_between(dates, '2014-01-20', None),
                dates
            ),
            D: zip_with_dates(
                ['NaT'] * num_days_between(dates, None, '2014-01-09') +
                ['2014-01-10'] * num_days_between(dates, '2014-01-10', '2014-01-14') +
                ['2014-01-15'] * num_days_between(dates, '2014-01-15', None),
                dates
            ),
            E: zip_with_dates(['NaT'] * len(dates), dates),
        }, index=dates)

        _expected_previous_busday_offsets = self._compute_busday_offsets(
            _expected_previous_buyback_announcement
        )

        self.cols['previous_buyback_announcement'] = _expected_previous_buyback_announcement
        self.cols['days_since_prev'] = _expected_previous_busday_offsets

        loader = self.loader_type(*self.loader_args(dates))
        engine = SimplePipelineEngine(lambda _: loader, dates, self.finder)
        return engine

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

    def _test_compute_buyback_auth(self, dates):
        engine = self.setup(dates)

        pipe = Pipeline(
            columns=self.pipeline_columns
        )

        result = engine.run_pipeline(
            pipe,
            start_date=dates[0],
            end_date=dates[-1],
        )

        for sid in sids:
            for col_name in self.cols.keys():
                assert_series_equal(result[col_name].xs(sid, level=1),
                                    self.cols[col_name][sid],
                                    sid)


class ShareBuybackAuthLoaderTestCase(BuybackAuthLoaderTestCase):
    buyback_authorizations = {sid: df.drop(VALUE_FIELD_NAME, 1)
                              for sid, df in iteritems(buyback_authorizations)}
    pipeline_columns = {
                'previous_buyback_share_count':
                    ShareBuybackAuthorizations.previous_share_count.latest,
                'previous_buyback_announcement':
                    ShareBuybackAuthorizations.previous_announcement_date.latest,
                'days_since_prev':
                    BusinessDaysSincePreviousShareBuybackAuth(),
            }

    @classmethod
    def setUpClass(cls):
        super(ShareBuybackAuthLoaderTestCase, cls).setUpClass()
        cls.buyback_authorizations = buyback_authorizations
        cls.loader_type = ShareBuybackAuthorizationsLoader

    def setup(self, dates):
        engine = super(ShareBuybackAuthLoaderTestCase, self).setup(dates)
        _expected_previous_buyback_share_count = pd.DataFrame({
                A: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-14') +
                   [1] * num_days_between(dates, '2014-01-15', '2014-01-19') +
                   [15] * num_days_between(dates, '2014-01-20', None), dates),
                B: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-14') +
                   [13] * num_days_between(dates, '2014-01-15', '2014-01-19') +
                   [7] * num_days_between(dates, '2014-01-20', None), dates),
                C: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-09') +
                   [3] * num_days_between(dates, '2014-01-10', '2014-01-19') +
                   [1] * num_days_between(dates, '2014-01-20', None), dates),
                D: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-09') +
                   [6] * num_days_between(dates, '2014-01-10', '2014-01-14') +
                   [23] * num_days_between(dates, '2014-01-15', None), dates),
                E: zip_with_floats(['NaN'] * len(dates), dates),
            }, index=dates)
        self.cols['previous_buyback_share_count'] = _expected_previous_buyback_share_count
        return engine

    @parameterized.expand(param_dates)
    def test_compute_buyback_auth(self, dates):
        self._test_compute_buyback_auth(dates)


class CashBuybackAuthLoaderTestCase(BuybackAuthLoaderTestCase):
    buyback_authorizations = {sid: df.drop(SHARE_COUNT_FIELD_NAME, 1)
                     for sid, df in iteritems(buyback_authorizations)}
    pipeline_columns = {
                'previous_buyback_value':
                    CashBuybackAuthorizations.previous_value.latest,
                'previous_buyback_announcement':
                    CashBuybackAuthorizations.previous_announcement_date.latest,
                'days_since_prev':
                    BusinessDaysSincePreviousCashBuybackAuth(),
            }

    @classmethod
    def setUpClass(cls):
        super(CashBuybackAuthLoaderTestCase, cls).setUpClass()
        cls.buyback_authorizations = buyback_authorizations
        cls.loader_type = CashBuybackAuthLoaderTestCase

    def setup(self, dates):
        engine = super(ShareBuybackAuthLoaderTestCase, self).setup(dates)
        _expected_previous_value = pd.DataFrame({
            # TODO if the next knowledge date is 10, why is the range
            #  until 15?
            A: zip_with_floats(
                ['NaN'] * num_days_between(dates, None, '2014-01-14') +
               [10] * num_days_between(dates, '2014-01-15', '2014-01-19') +
               [20] * num_days_between(dates, '2014-01-20', None), dates),
            B: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-14') +
               [22] * num_days_between(dates, '2014-01-15', '2014-01-19') +
               [10] * num_days_between(dates, '2014-01-20', None), dates),
            C: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-09') +
               [4] * num_days_between(dates, '2014-01-10', '2014-01-19') +
               [7] * num_days_between(dates, '2014-01-20', None), dates),
            D: zip_with_floats(['NaN'] * num_days_between(dates, None, '2014-01-09') +
               [1] * num_days_between(dates, '2014-01-10', '2014-01-14') +
               [2] * num_days_between(dates, '2014-01-15', None), dates),
            E: zip_with_floats(['NaN'] * len(dates), dates),
        }, index=dates)
        self.cols['previous_buyback_value'] = _expected_previous_value
        return engine

    @parameterized.expand(param_dates)
    def test_compute_buyback_auth(self, dates):
        self._test_compute_buyback_auth(dates)


# class BlazeBuybackAuthLoaderTestCase(BuybackAuthLoaderTestCase):
#     loader_type = BlazeBuybackAuthorizationsLoader
#
#     def loader_args(self, dates):
#         _, mapping = super(
#             BlazeBuybackAuthLoaderTestCase,
#             self,
#         ).loader_args(dates)
#         return (bz.Data(pd.concat(
#             pd.DataFrame({
#                 BUYBACK_ANNOUNCEMENT_FIELD_NAME:
#                     frame[BUYBACK_ANNOUNCEMENT_FIELD_NAME],
#                 SHARE_COUNT_FIELD_NAME: frame[SHARE_COUNT_FIELD_NAME],
#                 VALUE_FIELD_NAME: frame[VALUE_FIELD_NAME],
#                 TS_FIELD_NAME: frame.index,
#                 SID_FIELD_NAME: sid,
#             })
#             for sid, frame in iteritems(mapping)
#         ).reset_index(drop=True)),)
#
#
# class BlazeEarningsCalendarLoaderNotInteractiveTestCase(
#         BlazeBuybackAuthLoaderTestCase):
#     """Test case for passing a non-interactive symbol and a dict of resources.
#     """
#     def loader_args(self, dates):
#         (bound_expr,) = super(
#             BlazeEarningsCalendarLoaderNotInteractiveTestCase,
#             self,
#         ).loader_args(dates)
#         return swap_resources_into_scope(bound_expr, {})
#
#
# class BuybackAuthLoaderInferTimestampTestCase(TestCase):
#     def test_infer_timestamp(self):
#         dtx = pd.date_range('2014-01-01', '2014-01-10')
#         events_by_sid = {
#             0: pd.DataFrame({BUYBACK_ANNOUNCEMENT_FIELD_NAME: dtx}),
#             1: pd.DataFrame(
#                 {BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.Series(dtx, dtx)},
#                 index=dtx
#             )
#         }
#         loader = BuybackAuthorizationsLoader(
#             dtx,
#             events_by_sid,
#             infer_timestamps=True,
#         )
#         self.assertEqual(
#             loader.events_by_sid.keys(),
#             events_by_sid.keys(),
#         )
#         assert_series_equal(
#             loader.events_by_sid[0][BUYBACK_ANNOUNCEMENT_FIELD_NAME],
#             pd.Series(index=[dtx[0]] * 10, data=dtx),
#         )
#         assert_series_equal(
#             loader.events_by_sid[1][BUYBACK_ANNOUNCEMENT_FIELD_NAME],
#             events_by_sid[1][BUYBACK_ANNOUNCEMENT_FIELD_NAME],
#         )
