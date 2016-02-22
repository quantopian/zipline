"""
Tests for the reference loader for Buyback Authorizations.
"""
from functools import partial
from unittest import TestCase

import blaze as bz
from blaze.compute.core import swap_resources_into_scope
from contextlib2 import ExitStack
from nose_parameterized import parameterized
import pandas as pd
from six import iteritems
from tests.pipeline.test_events import param_dates, EventLoaderCommonTest

from zipline.pipeline.common import(
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    CASH_FIELD_NAME,
    DAYS_SINCE_PREV,
    PREVIOUS_BUYBACK_ANNOUNCEMENT,
    PREVIOUS_BUYBACK_CASH,
    PREVIOUS_BUYBACK_SHARE_COUNT,
    SHARE_COUNT_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME)
from zipline.pipeline.data import (CashBuybackAuthorizations,
                                   ShareBuybackAuthorizations)
from zipline.pipeline.factors.events import (
    BusinessDaysSincePreviousCashBuybackAuth,
    BusinessDaysSincePreviousShareBuybackAuth
)
from zipline.pipeline.loaders.buyback_auth import \
    CashBuybackAuthorizationsLoader, ShareBuybackAuthorizationsLoader
from zipline.pipeline.loaders.blaze import (
    BlazeCashBuybackAuthorizationsLoader,
    BlazeShareBuybackAuthorizationsLoader,
)
from zipline.utils.test_utils import (
    tmp_asset_finder,
)


buyback_authorizations = [
    # K1--K2--A1--A2.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-15',
                                                         '2014-01-20']),
        SHARE_COUNT_FIELD_NAME: [1, 15],
        CASH_FIELD_NAME: [10, 20]
    }),
    # K1--K2--A2--A1.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
            '2014-01-20', '2014-01-15'
        ]),
        SHARE_COUNT_FIELD_NAME: [7, 13], CASH_FIELD_NAME: [10, 22]
    }),
    # K1--A1--K2--A2.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
            '2014-01-10', '2014-01-20'
        ]),
        SHARE_COUNT_FIELD_NAME: [3, 1],
        CASH_FIELD_NAME: [4, 7]
    }),
    # K1 == K2.
    pd.DataFrame({
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime([
            '2014-01-10', '2014-01-15'
        ]),
        SHARE_COUNT_FIELD_NAME: [6, 23],
        CASH_FIELD_NAME: [1, 2]
    }),
    pd.DataFrame(
        columns=[TS_FIELD_NAME,
                 BUYBACK_ANNOUNCEMENT_FIELD_NAME,
                 SHARE_COUNT_FIELD_NAME,
                 CASH_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]


class CashBuybackAuthLoaderTestCase(TestCase, EventLoaderCommonTest):
    """
    Test for cash buyback authorizations dataset.
    """
    pipeline_columns = {
        (PREVIOUS_BUYBACK_CASH):
            CashBuybackAuthorizations.previous_value.latest,
        PREVIOUS_BUYBACK_ANNOUNCEMENT:
            CashBuybackAuthorizations.previous_announcement_date.latest,
        DAYS_SINCE_PREV:
            BusinessDaysSincePreviousCashBuybackAuth(),
    }

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=cls.equity_info),
        )
        cls.cols = {}
        cls.dataset = {sid: df.drop(SHARE_COUNT_FIELD_NAME, 1)
                       for sid, df in enumerate(buyback_authorizations)}
        cls.loader_type = CashBuybackAuthorizationsLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(self.zip_with_floats, dates)
        num_days_between_dates = partial(self.num_days_between, dates)
        _expected_previous_cash = pd.DataFrame({
            0: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [10] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [20] * num_days_between_dates('2014-01-20', None)
            ),
            1: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [22] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [10] * num_days_between_dates('2014-01-20', None)
            ),
            2: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [4] * num_days_between_dates('2014-01-10', '2014-01-19') +
                [7] * num_days_between_dates('2014-01-20', None)
            ),
            3: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [1] * num_days_between_dates('2014-01-10', '2014-01-14') +
                [2] * num_days_between_dates('2014-01-15', None)
            ),
            4: zip_with_floats_dates(['NaN'] * len(dates)),
        }, index=dates)
        self.cols[PREVIOUS_BUYBACK_ANNOUNCEMENT] = \
            self.get_expected_previous_event_dates(dates)
        self.cols[PREVIOUS_BUYBACK_CASH] = _expected_previous_cash
        self.cols[DAYS_SINCE_PREV] = self._compute_busday_offsets(
            self.cols[PREVIOUS_BUYBACK_ANNOUNCEMENT]
        )

    @parameterized.expand(param_dates)
    def test_compute_cash_buyback_auth(self, dates):
        self._test_compute(dates)


class ShareBuybackAuthLoaderTestCase(EventLoaderCommonTest, TestCase):
    """
    Test for share buyback authorizations dataset.
    """
    pipeline_columns = {
        PREVIOUS_BUYBACK_SHARE_COUNT:
            ShareBuybackAuthorizations.previous_share_count.latest,
        PREVIOUS_BUYBACK_ANNOUNCEMENT:
            ShareBuybackAuthorizations.previous_announcement_date.latest,
        DAYS_SINCE_PREV:
            BusinessDaysSincePreviousShareBuybackAuth(),
    }

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=cls.equity_info),
        )
        cls.cols = {}
        cls.dataset = {sid: df.drop(CASH_FIELD_NAME, 1)
                       for sid, df in
                       enumerate(buyback_authorizations)}
        cls.loader_type = ShareBuybackAuthorizationsLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(self.zip_with_floats, dates)
        num_days_between_dates = partial(self.num_days_between, dates)
        _expected_previous_buyback_share_count = pd.DataFrame({
            0: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [1] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [15] * num_days_between_dates('2014-01-20', None)
            ),
            1: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-14') +
                [13] * num_days_between_dates('2014-01-15', '2014-01-19') +
                [7] * num_days_between_dates('2014-01-20', None)
            ),
            2: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [3] * num_days_between_dates('2014-01-10', '2014-01-19') +
                [1] * num_days_between_dates('2014-01-20', None)
            ),
            3: zip_with_floats_dates(
                ['NaN'] * num_days_between_dates(None, '2014-01-09') +
                [6] * num_days_between_dates('2014-01-10', '2014-01-14') +
                [23] * num_days_between_dates('2014-01-15', None)
            ),
            4: zip_with_floats_dates(['NaN'] * len(dates)),
        }, index=dates)
        self.cols[
            PREVIOUS_BUYBACK_SHARE_COUNT
        ] = _expected_previous_buyback_share_count
        self.cols[PREVIOUS_BUYBACK_ANNOUNCEMENT] = \
            self.get_expected_previous_event_dates(dates)
        self.cols[DAYS_SINCE_PREV] = self._compute_busday_offsets(
            self.cols[PREVIOUS_BUYBACK_ANNOUNCEMENT]
        )

    @parameterized.expand(param_dates)
    def test_compute_share_buyback_auth(self, dates):
        self._test_compute(dates)


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
