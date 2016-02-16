"""
Tests for the reference loader for EarningsCalendar.
"""
from functools import partial
from unittest import TestCase

import blaze as bz
from blaze.compute.core import swap_resources_into_scope
from contextlib2 import ExitStack
from nose_parameterized import parameterized
import numpy as np
import pandas as pd
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
    BlazeCashBuybackAuthorizationsLoader,
    BlazeShareBuybackAuthorizationsLoader,
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    SHARE_COUNT_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
    CASH_FIELD_NAME
)
from zipline.utils.numpy_utils import make_datetime64D, NaTD
from zipline.utils.test_utils import (
    gen_calendars,
    make_simple_equity_info,
    num_days_in_range,
    tmp_asset_finder,
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
        CASH_FIELD_NAME: [10, 20]
    }),
    # K1--K2--E2--E1.
    B: pd.DataFrame({
        "timestamp": pd.to_datetime(['2014-01-05', '2014-01-10']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
            '2014-01-20', '2014-01-15'
        ]),
        SHARE_COUNT_FIELD_NAME: [7, 13], CASH_FIELD_NAME: [10, 22]
    }),
    # K1--E1--K2--E2.
    C: pd.DataFrame({
        "timestamp": pd.to_datetime(['2014-01-05', '2014-01-15']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
            '2014-01-10', '2014-01-20'
        ]),
        SHARE_COUNT_FIELD_NAME: [3, 1],
        CASH_FIELD_NAME: [4, 7]
    }),
    # K1 == K2.
    D: pd.DataFrame({
        "timestamp": pd.to_datetime(['2014-01-05'] * 2),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
            '2014-01-10', '2014-01-15'
        ]),
        SHARE_COUNT_FIELD_NAME: [6, 23],
        CASH_FIELD_NAME: [1, 2]
    }),
    E: pd.DataFrame(
        columns=["timestamp",
                 BUYBACK_ANNOUNCEMENT_FIELD_NAME,
                 SHARE_COUNT_FIELD_NAME,
                 CASH_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
}

# Must be a list - can't use generator since this needs to be used more than
# once.
param_dates = list(gen_calendars(
    '2014-01-01',
    '2014-01-31',
    critical_dates=pd.to_datetime([
        '2014-01-05',
        '2014-01-10',
        '2014-01-15',
        '2014-01-20',
    ]),
))


def zip_with_floats(dates, flts):
    return pd.Series(flts, index=dates).astype('float')


def num_days_between(dates, start_date, end_date):
    return num_days_in_range(dates, start_date, end_date)


def zip_with_dates(index_dates, dts):
    return pd.Series(pd.to_datetime(dts), index=index_dates)


class BuybackAuthLoaderCommonTest(object):
    """
    Tests for loading the buyback authorization announcement data.
    """

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

    def setup_engine(self, dates):
        """
        Make a Pipeline Enigne object based on the given dates.
        """
        loader = self.loader_type(*self.loader_args(dates))
        return SimplePipelineEngine(lambda _: loader, dates, self.finder)

    def setup_expected_cols(self, dates):
        """
        Make expectation functions for the given dates calendar.

        This exists to make it easy to test our various cases with critical
        dates missing from the calendar.
        """
        num_days_between_for_dates = partial(num_days_between, dates)
        zip_with_dates_for_dates = partial(zip_with_dates, dates)
        _expected_previous_buyback_announcement = pd.DataFrame({
            A: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            B: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            C: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between_for_dates('2014-01-10',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            D: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between_for_dates('2014-01-10',
                                                            '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            None),
            ),
            E: zip_with_dates_for_dates(['NaT'] * len(dates)),
        }, index=dates)

        _expected_previous_busday_offsets = self._compute_busday_offsets(
            _expected_previous_buyback_announcement
        )

        # Common cols for buyback authorization datasets are announcement
        # date and days since previous.
        self.cols[
            'previous_buyback_announcement'
        ] = _expected_previous_buyback_announcement
        self.cols['days_since_prev'] = _expected_previous_busday_offsets

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
        whereNaT = raw_announce_dates == NaTD
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
        engine = self.setup_engine(dates)
        self.setup_expected_cols(dates)

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


class CashBuybackAuthLoaderTestCase(TestCase, BuybackAuthLoaderCommonTest):
    """
    Test for cash buyback authorizations dataset.
    """
    pipeline_columns = {
        'previous_buyback_cash':
            CashBuybackAuthorizations.previous_value.latest,
        'previous_buyback_announcement':
            CashBuybackAuthorizations.previous_announcement_date.latest,
        'days_since_prev':
            BusinessDaysSincePreviousCashBuybackAuth(),
    }

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )
        cls.cols = {}
        cls.buyback_authorizations = {sid: df.drop(SHARE_COUNT_FIELD_NAME, 1)
                                      for sid, df in
                                      iteritems(buyback_authorizations)}
        cls.loader_type = CashBuybackAuthorizationsLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(zip_with_floats, dates)
        num_days_between_dates = partial(num_days_between, dates)
        super(CashBuybackAuthLoaderTestCase, self).setup_expected_cols(dates)
        _expected_previous_cash = pd.DataFrame({
            # TODO if the next knowledge date is 10, why is the range
            #  until 15?
            A: zip_with_floats_dates(
                ['NaN'] * num_days_between(dates, None, '2014-01-14') +
                [10] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [20] * num_days_between_dates('2014-01-20', None)
            ),
            B: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [22] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [10] * num_days_between_dates('2014-01-20', None)
            ),
            C: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [4] * num_days_between_dates('2014-01-10', '2014-01-19') +
                [7] * num_days_between_dates('2014-01-20', None)
            ),
            D: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [1] * num_days_between_dates('2014-01-10', '2014-01-14') +
                [2] * num_days_between_dates('2014-01-15', None)
            ),
            E: zip_with_floats_dates(['NaN'] * len(dates)),
        }, index=dates)
        self.cols['previous_buyback_cash'] = _expected_previous_cash

    @parameterized.expand(param_dates)
    def test_compute_cash_buyback_auth(self, dates):
        self._test_compute_buyback_auth(dates)


class ShareBuybackAuthLoaderTestCase(BuybackAuthLoaderCommonTest, TestCase):
    """
    Test for share buyback authorizations dataset.
    """
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
        cls._cleanup_stack = stack = ExitStack()
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )
        cls.cols = {}
        cls.buyback_authorizations = {sid: df.drop(CASH_FIELD_NAME, 1)
                                      for sid, df in
                                      iteritems(buyback_authorizations)}
        cls.loader_type = ShareBuybackAuthorizationsLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(zip_with_floats, dates)
        num_days_between_dates = partial(num_days_between, dates)
        super(ShareBuybackAuthLoaderTestCase, self).setup_expected_cols(dates)
        _expected_previous_buyback_share_count = pd.DataFrame({
            A: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [1] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [15] * num_days_between_dates('2014-01-20', None)
            ),
            B: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [13] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [7] * num_days_between_dates('2014-01-20', None)
            ),
            C: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [3] * num_days_between_dates('2014-01-10', '2014-01-19') +
                [1] * num_days_between_dates('2014-01-20', None)
            ),
            D: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [6] * num_days_between_dates('2014-01-10', '2014-01-14') +
                [23] * num_days_between_dates('2014-01-15', None)
            ),
            E: zip_with_floats_dates(['NaN'] * len(dates)),
        }, index=dates)
        self.cols[
            'previous_buyback_share_count'
        ] = _expected_previous_buyback_share_count

    @parameterized.expand(param_dates)
    def test_compute_share_buyback_auth(self, dates):
        self._test_compute_buyback_auth(dates)


class BlazeCashBuybackAuthLoaderTestCase(CashBuybackAuthLoaderTestCase):
    """ Test case for loading via blaze.
    """
    @classmethod
    def setUpClass(cls):
        super(BlazeCashBuybackAuthLoaderTestCase, cls).setUpClass()
        cls.loader_type = BlazeCashBuybackAuthorizationsLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeCashBuybackAuthLoaderTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                BUYBACK_ANNOUNCEMENT_FIELD_NAME:
                    frame[BUYBACK_ANNOUNCEMENT_FIELD_NAME],
                CASH_FIELD_NAME:
                    frame[CASH_FIELD_NAME],
                TS_FIELD_NAME:
                    frame[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
            })
            for sid, frame in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeShareBuybackAuthLoaderTestCase(ShareBuybackAuthLoaderTestCase):
    """ Test case for loading via blaze.
    """
    @classmethod
    def setUpClass(cls):
        super(BlazeShareBuybackAuthLoaderTestCase, cls).setUpClass()
        cls.loader_type = BlazeShareBuybackAuthorizationsLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeShareBuybackAuthLoaderTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                BUYBACK_ANNOUNCEMENT_FIELD_NAME:
                    frame[BUYBACK_ANNOUNCEMENT_FIELD_NAME],
                SHARE_COUNT_FIELD_NAME:
                    frame[SHARE_COUNT_FIELD_NAME],
                TS_FIELD_NAME:
                    frame[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
            })
            for sid, frame in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeShareBuybackAuthLoaderNotInteractiveTestCase(
        BlazeShareBuybackAuthLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeShareBuybackAuthLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})


class BlazeCashBuybackAuthLoaderNotInteractiveTestCase(
        BlazeCashBuybackAuthLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeCashBuybackAuthLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})

dtx = pd.date_range('2014-01-01', '2014-01-10')


class BuybackAuthLoaderInferTimestampTestCase(TestCase):
    # 'fields' needs to match expected fields for the given loader to
    # satisfy column check in constructor.
    @parameterized.expand([[CashBuybackAuthorizationsLoader,
                            {BUYBACK_ANNOUNCEMENT_FIELD_NAME: dtx,
                             CASH_FIELD_NAME: [0] * 10}],
                           [ShareBuybackAuthorizationsLoader,
                            {BUYBACK_ANNOUNCEMENT_FIELD_NAME: dtx,
                             SHARE_COUNT_FIELD_NAME: [0] * 10}]])
    def test_infer_timestamp(self, loader, fields):
        events_by_sid = {
            # No timestamp column - should index by first given date
            0: pd.DataFrame(fields),
            # timestamp column exists - should index by it
            1: pd.DataFrame(dict(fields, **{TS_FIELD_NAME: dtx}))
        }
        loader = loader(
            dtx,
            events_by_sid,
            infer_timestamps=True,
        )
        self.assertEqual(
            loader.events_by_sid.keys(),
            events_by_sid.keys(),
        )

        # Check that index by first given date has been added
        assert_series_equal(
            loader.events_by_sid[0][BUYBACK_ANNOUNCEMENT_FIELD_NAME],
            pd.Series(index=[dtx[0]] * 10,
                      data=dtx,
                      name=BUYBACK_ANNOUNCEMENT_FIELD_NAME),
        )

        # Check that timestamp column was turned into index
        modified_events_by_sid_date_col = pd.Series(data=np.array(
            events_by_sid[1][BUYBACK_ANNOUNCEMENT_FIELD_NAME]),
            index=events_by_sid[1][TS_FIELD_NAME],
            name=BUYBACK_ANNOUNCEMENT_FIELD_NAME)
        assert_series_equal(
            loader.events_by_sid[1][BUYBACK_ANNOUNCEMENT_FIELD_NAME],
            modified_events_by_sid_date_col,
        )
