"""
Tests for the reference loader for Dividends datasets.
"""
from functools import partial
from unittest import TestCase

import blaze as bz
from blaze.compute.core import swap_resources_into_scope
from contextlib2 import ExitStack
import itertools
import pandas as pd
from six import iteritems
from tests.pipeline.base import EventLoaderCommonMixin

from zipline.pipeline.common import (
    ANNOUNCEMENT_FIELD_NAME,
    DAYS_SINCE_PREV_DIVIDEND_ANNOUNCEMENT,
    DAYS_SINCE_PREV_EX_DATE,
    DAYS_TO_NEXT_EX_DATE,
    NEXT_AMOUNT,
    NEXT_EX_DATE,
    NEXT_PAY_DATE,
    PREVIOUS_ANNOUNCEMENT,
    PREVIOUS_EX_DATE,
    PREVIOUS_PAY_DATE,
    PREVIOUS_AMOUNT,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
    CASH_AMOUNT_FIELD_NAME,
    EX_DATE_FIELD_NAME,
    PAY_DATE_FIELD_NAME
)
from zipline.pipeline.data.dividends import DividendsByAnnouncementDate, \
    DividendsByExDate, DividendsByPayDate
from zipline.pipeline.factors.events import (
    BusinessDaysSinceDividendAnnouncement,
    BusinessDaysSincePreviousExDate,
    BusinessDaysUntilNextExDate
)
from zipline.pipeline.loaders.blaze.dividends import \
    BlazeDividendsByAnnouncementDateLoader, BlazeDividendsByPayDateLoader, \
    BlazeDividendsByExDateLoader
from zipline.pipeline.loaders.dividends import DividendsByAnnouncementDateLoader, \
    DividendsByExDateLoader, DividendsByPayDateLoader
from zipline.utils.test_utils import (
    make_simple_equity_info,
    tmp_asset_finder,
)


dividends_cases = [
    # K1--K2--A1--A2.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [1, 15],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-15', '2014-01-20']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-15', '2014-01-20']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-09'])
    }),
    # K1--K2--A2--A1.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [7, 13],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-20', '2014-01-15']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-20', '2014-01-15']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-09'])
    }),
    # K1--A1--K2--A2.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [3, 1],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-20']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-20']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-14'])
    }),
    # K1 == K2.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [6, 23],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-15']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-15']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-04'])
    }),
    pd.DataFrame(
        columns=[CASH_AMOUNT_FIELD_NAME,
                 EX_DATE_FIELD_NAME,
                 PAY_DATE_FIELD_NAME,
                 TS_FIELD_NAME,
                 ANNOUNCEMENT_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]

prev_date_intervals = [
            [
                [None, '2014-01-14'], ['2014-01-15', '2014-01-19'],
                ['2014-01-20', None]
            ],
            [
                [None, '2014-01-14'], ['2014-01-15', '2014-01-19'],
                ['2014-01-20', None]
            ],
            [
               [None, '2014-01-09'], ['2014-01-10', '2014-01-19'],
               ['2014-01-20', None]
            ],
            [
               [None, '2014-01-09'], ['2014-01-10', '2014-01-14'],
               ['2014-01-15', None]
            ]
        ]

next_date_intervals = [
    [
        [None, '2014-01-04'], ['2014-01-05', '2014-01-15'],
        ['2014-01-16', '2014-01-20'], ['2014-01-21', None]
    ],
    [
        [None, '2014-01-04'], ['2014-01-05', '2014-01-09'],
        ['2014-01-10', '2014-01-15'], ['2014-01-16', '2014-01-20'],
        ['2014-01-21', None]
    ],
    [
        [None, '2014-01-04'], ['2014-01-05', '2014-01-10'],
        ['2014-01-11', '2014-01-14'], ['2014-01-15', '2014-01-20'],
        ['2014-01-21', None]
    ],
    [
        [None, '2014-01-04'], ['2014-01-05', '2014-01-10'],
        ['2014-01-11', '2014-01-15'], ['2014-01-16', None]
    ]
]

next_ex_and_pay_dates = [['NaT', '2014-01-15', '2014-01-20', 'NaT'],
                         ['NaT', '2014-01-20', '2014-01-15', '2014-01-20',
                          'NaT'],
                         ['NaT', '2014-01-10', 'NaT', '2014-01-20', 'NaT'],
                         ['NaT', '2014-01-10', '2014-01-15', 'NaT']]

prev_ex_and_pay_dates = [['NaT', '2014-01-15', '2014-01-20'],
                           ['NaT', '2014-01-15', '2014-01-20'],
                           ['NaT', '2014-01-10', '2014-01-20'],
                           ['NaT', '2014-01-10', '2014-01-15']]

prev_amounts = [['NaN', 1, 15],
                ['NaN', 13, 7],
                ['NaN', 3, 1],
                ['NaN', 6, 23]]

next_amounts = [['NaN', 1, 15, 'NaN'],
                ['NaN', 7, 13, 7, 'NaN'],
                ['NaN', 3, 'NaN', 1, 'NaN'],
                ['NaN', 6, 23, 'NaN']]


def get_values_for_date_ranges(zip_vals_dates,
                               num_days_between_dates,
                               vals_for_date_intervals,
                               date_intervals):
    # Fill in given values for given date ranges.
    return zip_vals_dates(
        list(
            itertools.chain(*[
                [val] * num_days_between_dates(*date_intervals[i])
                for i, val in enumerate(vals_for_date_intervals)
            ])
        )
    )


def get_vals_for_dates(zip_with_floats_dates,
                                   num_days_between_dates,
                                   dates,
                                   date_invervals,
                                   vals):
    return pd.DataFrame({
            0: get_values_for_date_ranges(zip_with_floats_dates,
                                          num_days_between_dates,
                                          vals[0],
                                          date_invervals[0]),
            1: get_values_for_date_ranges(zip_with_floats_dates,
                                          num_days_between_dates,
                                          vals[1],
                                          date_invervals[1]),
            2: get_values_for_date_ranges(zip_with_floats_dates,
                                          num_days_between_dates,
                                          vals[2],
                                          date_invervals[2]),
            # Assume the latest of 2 cash values is used if we find out about 2
            # announcements that happened on the same day for the same sid.
            3: get_values_for_date_ranges(zip_with_floats_dates,
                                          num_days_between_dates,
                                          vals[3],
                                          date_invervals[3]),
            4: zip_with_floats_dates(['NaN'] * len(dates)),
        }, index=dates)


class DividendsByAnnouncementDateTestCase(TestCase, EventLoaderCommonMixin):
    """
    Tests for loading the dividends by announcement date data.
    """
    pipeline_columns = {
        PREVIOUS_ANNOUNCEMENT:
            DividendsByAnnouncementDate.previous_announcement_date.latest,
        PREVIOUS_AMOUNT: DividendsByAnnouncementDate.previous_amount.latest,
        DAYS_SINCE_PREV_DIVIDEND_ANNOUNCEMENT:
            BusinessDaysSinceDividendAnnouncement(),
    }

    @classmethod
    def get_sids(cls):
        return range(0, 5)

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        equity_info = make_simple_equity_info(
            cls.get_sids(),
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )
        cls.cols = {}
        cls.dataset = {sid:
                       frame.drop([EX_DATE_FIELD_NAME,
                                   PAY_DATE_FIELD_NAME], axis=1)
                       for sid, frame
                       in enumerate(dividends_cases)}
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )

        cls.loader_type = DividendsByAnnouncementDateLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(self.zip_with_floats, dates)
        num_days_between_dates = partial(self.num_days_between, dates)
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)
        date_intervals = [
            [
                [None, '2014-01-04'], ['2014-01-05', '2014-01-09'],
                ['2014-01-10', None]
            ],
            [
                [None, '2014-01-04'], ['2014-01-05', '2014-01-09'],
                ['2014-01-10', None]
            ],
            [
               [None, '2014-01-04'], ['2014-01-05', '2014-01-14'],
               ['2014-01-15', None]
            ],
            [
               [None, '2014-01-04'], ['2014-01-05', None]
            ]
        ]
        announcement_dates = [['NaT', '2014-01-04', '2014-01-09'],
                 ['NaT', '2014-01-04', '2014-01-09'],
                 ['NaT', '2014-01-04', '2014-01-14'],
                 ['NaT', '2014-01-04']]
        amounts = [['NaN', 1, 15], ['NaN', 7, 13], ['NaN', 3, 1], ['NaN', 23]]

        self.cols[PREVIOUS_ANNOUNCEMENT] = get_vals_for_dates(
            zip_with_dates_for_dates, num_days_between_for_dates, dates,
            date_intervals, announcement_dates
        )

        self.cols[PREVIOUS_AMOUNT] = get_vals_for_dates(
            zip_with_floats_dates, num_days_between_dates, dates,
            date_intervals, amounts
        )

        self.cols[
            DAYS_SINCE_PREV_DIVIDEND_ANNOUNCEMENT
        ] = self._compute_busday_offsets(self.cols[PREVIOUS_ANNOUNCEMENT])


class BlazeDividendsByAnnouncementDateTestCase(
    DividendsByAnnouncementDateTestCase
):
    @classmethod
    def setUpClass(cls):
        super(BlazeDividendsByAnnouncementDateTestCase, cls).setUpClass()
        cls.loader_type = BlazeDividendsByAnnouncementDateLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeDividendsByAnnouncementDateTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                ANNOUNCEMENT_FIELD_NAME: df[ANNOUNCEMENT_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME]
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeDividendsByAnnouncementDateNotInteractiveTestCase(
        BlazeDividendsByAnnouncementDateTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    @classmethod
    def setUpClass(cls):
        super(BlazeDividendsByAnnouncementDateNotInteractiveTestCase,
              cls).setUpClass()
        cls.loader_type = BlazeDividendsByAnnouncementDateLoader

    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeDividendsByAnnouncementDateNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})


class DividendsByExDateTestCase(TestCase, EventLoaderCommonMixin):
    """
    Tests for loading the dividends by ex date data.
    """
    pipeline_columns = {
        NEXT_EX_DATE: DividendsByExDate.previous_ex_date.latest,
        PREVIOUS_EX_DATE: DividendsByExDate.next_ex_date.latest,
        NEXT_AMOUNT: DividendsByExDate.next_amount.latest,
        PREVIOUS_AMOUNT: DividendsByExDate.previous_amount.latest,
        DAYS_TO_NEXT_EX_DATE: BusinessDaysUntilNextExDate(),
        DAYS_SINCE_PREV_EX_DATE: BusinessDaysSincePreviousExDate()
    }

    @classmethod
    def get_sids(cls):
        return range(0, 5)

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        equity_info = make_simple_equity_info(
            cls.get_sids(),
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )
        cls.cols = {}
        cls.dataset = {sid:
                       frame.drop([ANNOUNCEMENT_FIELD_NAME,
                                   PAY_DATE_FIELD_NAME], axis=1)
                       for sid, frame
                       in enumerate(dividends_cases)}
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )

        cls.loader_type = DividendsByExDateLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(self.zip_with_floats, dates)
        num_days_between_dates = partial(self.num_days_between, dates)
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)

        self.cols[NEXT_EX_DATE] = get_vals_for_dates(
            zip_with_dates_for_dates, num_days_between_for_dates, dates,
            next_date_intervals, next_ex_and_pay_dates
        )

        self.cols[PREVIOUS_EX_DATE] = get_vals_for_dates(
            zip_with_dates_for_dates, num_days_between_for_dates, dates,
            prev_date_intervals, prev_ex_and_pay_dates
        )

        self.cols[NEXT_AMOUNT] = get_vals_for_dates(
            zip_with_floats_dates, num_days_between_dates,
            dates, next_date_intervals, next_amounts
        )

        self.cols[PREVIOUS_AMOUNT] = get_vals_for_dates(
            zip_with_floats_dates, num_days_between_dates,
            dates, prev_date_intervals, prev_amounts
        )

        self.cols[DAYS_TO_NEXT_EX_DATE] = self._compute_busday_offsets(
                self.cols[NEXT_EX_DATE]
        )

        self.cols[DAYS_SINCE_PREV_EX_DATE] = self._compute_busday_offsets(
                self.cols[PREVIOUS_EX_DATE]
        )


class BlazeDividendsByExDateLoaderTestCase(DividendsByExDateTestCase):
    @classmethod
    def setUpClass(cls):
        super(BlazeDividendsByExDateLoaderTestCase, cls).setUpClass()
        cls.loader_type = BlazeDividendsByExDateLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeDividendsByExDateLoaderTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                EX_DATE_FIELD_NAME: df[EX_DATE_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME]
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeDividendsByExDateLoaderNotInteractiveTestCase(
        BlazeDividendsByExDateLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    @classmethod
    def setUpClass(cls):
        super(BlazeDividendsByExDateLoaderNotInteractiveTestCase,
              cls).setUpClass()
        cls.loader_type = DividendsByExDateLoader

    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeDividendsByExDateLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})


class DividendsByPayDateTestCase(TestCase, EventLoaderCommonMixin):
    """
    Tests for loading the dividends by pay date data.
    """
    pipeline_columns = {
        NEXT_PAY_DATE: DividendsByPayDate.next_pay_date.latest,
        PREVIOUS_PAY_DATE: DividendsByPayDate.previous_pay_date.latest,
        NEXT_AMOUNT: DividendsByPayDate.next_amount.latest,
        PREVIOUS_AMOUNT: DividendsByPayDate.previous_amount.latest,
    }

    @classmethod
    def get_sids(cls):
        return range(0, 5)

    @classmethod
    def setUpClass(cls):
        cls._cleanup_stack = stack = ExitStack()
        equity_info = make_simple_equity_info(
            cls.get_sids(),
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )
        cls.cols = {}
        cls.dataset = {sid:
                       frame.drop([ANNOUNCEMENT_FIELD_NAME,
                                   EX_DATE_FIELD_NAME], axis=1)
                       for sid, frame
                       in enumerate(dividends_cases)}
        cls.finder = stack.enter_context(
            tmp_asset_finder(equities=equity_info),
        )

        cls.loader_type = DividendsByPayDateLoader

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_stack.close()

    def setup(self, dates):
        zip_with_floats_dates = partial(self.zip_with_floats, dates)
        num_days_between_dates = partial(self.num_days_between, dates)
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)

        self.cols[NEXT_PAY_DATE] = get_vals_for_dates(
            zip_with_dates_for_dates, num_days_between_for_dates, dates,
            next_date_intervals, next_ex_and_pay_dates
        )
        self.cols[PREVIOUS_PAY_DATE] = get_vals_for_dates(
            zip_with_dates_for_dates, num_days_between_for_dates, dates,
            prev_date_intervals, prev_ex_and_pay_dates
        )

        self.cols[NEXT_AMOUNT] = get_vals_for_dates(
            zip_with_floats_dates, num_days_between_dates,
            dates, next_date_intervals, next_amounts
        )

        self.cols[PREVIOUS_AMOUNT] = get_vals_for_dates(
            zip_with_floats_dates, num_days_between_dates,
            dates, prev_date_intervals, prev_amounts
        )


class BlazeDividendsByPayDateLoaderTestCase(DividendsByPayDateTestCase):
    @classmethod
    def setUpClass(cls):
        super(BlazeDividendsByPayDateLoaderTestCase, cls).setUpClass()
        cls.loader_type = BlazeDividendsByPayDateLoader

    def loader_args(self, dates):
        _, mapping = super(
            BlazeDividendsByPayDateLoaderTestCase,
            self,
        ).loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                PAY_DATE_FIELD_NAME: df[PAY_DATE_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME]
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeDividendsByPayDateLoaderNotInteractiveTestCase(
        BlazeDividendsByPayDateLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    @classmethod
    def setUpClass(cls):
        super(BlazeDividendsByPayDateLoaderNotInteractiveTestCase,
              cls).setUpClass()
        cls.loader_type = BlazeDividendsByPayDateLoader

    def loader_args(self, dates):
        (bound_expr,) = super(
            BlazeDividendsByPayDateLoaderNotInteractiveTestCase,
            self,
        ).loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
