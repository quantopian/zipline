"""
Tests for the reference loader for Dividends datasets.
"""
import blaze as bz
from blaze.compute.core import swap_resources_into_scope
import pandas as pd
from six import iteritems

from zipline.pipeline.common import (
    ANNOUNCEMENT_FIELD_NAME,
    PREVIOUS_AMOUNT,
    PREVIOUS_ANNOUNCEMENT,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.data.dividends import (
    DividendsByAnnouncementDate,
    DividendsByExDate,
    DividendsByPayDate
)
from zipline.pipeline.factors.events import (
    BusinessDaysSinceDividendAnnouncement,
    BusinessDaysSincePreviousExDate,
    BusinessDaysUntilNextExDate
)
from zipline.pipeline.loaders.blaze.dividends import (
    BlazeDividendsByAnnouncementDateLoader,
    BlazeDividendsByPayDateLoader,
    BlazeDividendsByExDateLoader
)
from zipline.pipeline.loaders.dividends import (
    CASH_AMOUNT_FIELD_NAME,
    CURRENCY_FIELD_NAME,
    DIVIDEND_TYPE_FIELD_NAME,
    DividendsByAnnouncementDateLoader,
    DividendsByExDateLoader,
    DividendsByPayDateLoader,
    EX_DATE_FIELD_NAME,
    PAY_DATE_FIELD_NAME,
)
from zipline.pipeline.loaders.utils import (
    zip_with_dates,
    zip_with_floats,
    zip_with_strs,
)
from zipline.testing.fixtures import (
    WithPipelineEventDataLoader,
    ZiplineTestCase
)

DAYS_SINCE_PREV_DIVIDEND_ANNOUNCEMENT = 'days_since_prev_dividend_announcement'
DAYS_SINCE_PREV_EX_DATE = 'days_since_prev_ex_date'
DAYS_TO_NEXT_EX_DATE = 'days_to_next_ex_date'
NEXT_AMOUNT = 'next_amount'
NEXT_CURRENCY_TYPE = 'next_currency_type'
NEXT_DIVIDEND_TYPE = 'next_dividend_type'
NEXT_EX_DATE = 'next_ex_date'
NEXT_PAY_DATE = 'next_pay_date'
PREVIOUS_CURRENCY_TYPE = 'previous_currency_type'
PREVIOUS_DIVIDEND_TYPE = 'previous_dividend_type'
PREVIOUS_EX_DATE = 'previous_ex_date'
PREVIOUS_PAY_DATE = 'previous_pay_date'


dividends_cases = [
    # K1--K2--A1--A2.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [1, 15],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-15', '2014-01-20']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-15', '2014-01-20']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-09']),
        CURRENCY_FIELD_NAME: ["$", "EUR"],
        DIVIDEND_TYPE_FIELD_NAME: ["Stock", "Mixed"]
    }),
    # K1--K2--A2--A1.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [7, 13],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-20', '2014-01-15']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-20', '2014-01-15']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-09']),
        CURRENCY_FIELD_NAME: ["EUR", "$"],
        DIVIDEND_TYPE_FIELD_NAME: ["Mixed", "Stock"]
    }),
    # K1--A1--K2--A2.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [3, 1],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-20']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-20']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-14']),
        CURRENCY_FIELD_NAME: ["$", "EUR"],
        DIVIDEND_TYPE_FIELD_NAME: ["Stock", "Mixed"]
    }),
    # K1 == K2.
    pd.DataFrame({
        CASH_AMOUNT_FIELD_NAME: [6, 23],
        EX_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-15']),
        PAY_DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-15']),
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
        ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04', '2014-01-04']),
        CURRENCY_FIELD_NAME: ["$", "EUR"],
        DIVIDEND_TYPE_FIELD_NAME: ["Stock", "Mixed"]
    }),
    pd.DataFrame(
        columns=[CASH_AMOUNT_FIELD_NAME,
                 EX_DATE_FIELD_NAME,
                 PAY_DATE_FIELD_NAME,
                 TS_FIELD_NAME,
                 ANNOUNCEMENT_FIELD_NAME,
                 CURRENCY_FIELD_NAME,
                 DIVIDEND_TYPE_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]

prev_date_intervals = [
    [
        ['2014-01-01', '2014-01-14'], ['2014-01-15', '2014-01-19'],
        ['2014-01-20', '2014-01-31']
    ],
    [
        ['2014-01-01', '2014-01-14'], ['2014-01-15', '2014-01-19'],
        ['2014-01-20', '2014-01-31']
    ],
    [
        ['2014-01-01', '2014-01-09'], ['2014-01-10', '2014-01-19'],
        ['2014-01-20', '2014-01-31']
    ],
    [
        ['2014-01-01', '2014-01-09'], ['2014-01-10', '2014-01-14'],
        ['2014-01-15', '2014-01-31']
    ]
]

next_date_intervals = [
    [
        ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-15'],
        ['2014-01-16', '2014-01-20'], ['2014-01-21', '2014-01-31']
    ],
    [
        ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-09'],
        ['2014-01-10', '2014-01-15'], ['2014-01-16', '2014-01-20'],
        ['2014-01-21', '2014-01-31']
    ],
    [
        ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-10'],
        ['2014-01-11', '2014-01-14'], ['2014-01-15', '2014-01-20'],
        ['2014-01-21', '2014-01-31']
    ],
    [
        ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-10'],
        ['2014-01-11', '2014-01-15'], ['2014-01-16', '2014-01-31']
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

prev_currency_types = [[None, "$", "EUR"],
                       [None, "$", "EUR"],
                       [None, "$", "EUR"],
                       [None, "$", "EUR"]]

next_currency_types = [[None, "$", "EUR", None],
                       [None, "EUR", "$", "EUR", None],
                       [None, "$", None, "EUR", None],
                       [None, "$", "EUR", None]]

prev_dividend_types = [[None, "Stock", "Mixed"],
                       [None, "Stock", "Mixed"],
                       [None, "Stock", "Mixed"],
                       [None, "Stock", "Mixed"]]

next_dividend_types = [[None, "Stock", "Mixed", None],
                       [None, "Mixed", "Stock", "Mixed", None],
                       [None, "Stock", None, "Mixed", None],
                       [None, "Stock", "Mixed", None]]


class DividendsByAnnouncementDateTestCase(WithPipelineEventDataLoader,
                                          ZiplineTestCase):
    """
    Tests for loading the dividends by announcement date data.
    """
    pipeline_columns = {
        PREVIOUS_ANNOUNCEMENT:
            DividendsByAnnouncementDate.previous_announcement_date.latest,
        PREVIOUS_AMOUNT: DividendsByAnnouncementDate.previous_amount.latest,
        DAYS_SINCE_PREV_DIVIDEND_ANNOUNCEMENT:
            BusinessDaysSinceDividendAnnouncement(),
        PREVIOUS_CURRENCY_TYPE:
            DividendsByAnnouncementDate.previous_currency.latest,
        PREVIOUS_DIVIDEND_TYPE:
            DividendsByAnnouncementDate.previous_type.latest,
    }

    @classmethod
    def get_dataset(cls):
        return {sid:
                frame.drop([EX_DATE_FIELD_NAME,
                            PAY_DATE_FIELD_NAME], axis=1)
                for sid, frame
                in enumerate(dividends_cases)}

    loader_type = DividendsByAnnouncementDateLoader

    def setup(self, dates):
        date_intervals = [
            [
                ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-09'],
                ['2014-01-10', '2014-01-31']
            ],
            [
                ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-09'],
                ['2014-01-10', '2014-01-31']
            ],
            [
                ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-14'],
                ['2014-01-15', '2014-01-31']
            ],
            [
                ['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-31']
            ]
        ]
        announcement_dates = [['NaT', '2014-01-04', '2014-01-09'],
                              ['NaT', '2014-01-04', '2014-01-09'],
                              ['NaT', '2014-01-04', '2014-01-14'],
                              ['NaT', '2014-01-04']]
        amounts = [['NaN', 1, 15], ['NaN', 7, 13], ['NaN', 3, 1], ['NaN', 23]]
        currency_types = [[None, "$", "EUR"], [None, "EUR", "$"],
                          [None, "$", "EUR"], [None, "EUR"]]
        dividend_types = [[None, "Stock", "Mixed"], [None, "Mixed", "Stock"],
                          [None, "Stock", "Mixed"], [None, "Mixed"]]
        cols = {
            PREVIOUS_ANNOUNCEMENT: self.get_sids_to_frames(
                zip_with_dates, announcement_dates, date_intervals, dates,
                'datetime64[ns]', 'NaN'
            ),
            PREVIOUS_AMOUNT: self.get_sids_to_frames(
                zip_with_floats, amounts, date_intervals, dates, 'float', 'NaN'
            ),
            PREVIOUS_CURRENCY_TYPE: self.get_sids_to_frames(
                zip_with_strs, currency_types, date_intervals, dates,
                'category', None
            ),
            PREVIOUS_DIVIDEND_TYPE: self.get_sids_to_frames(
                zip_with_strs, dividend_types, date_intervals, dates,
                'category', None
            ),
        }

        cols[
            DAYS_SINCE_PREV_DIVIDEND_ANNOUNCEMENT
        ] = self._compute_busday_offsets(cols[PREVIOUS_ANNOUNCEMENT])
        return cols


class BlazeDividendsByAnnouncementDateTestCase(
    DividendsByAnnouncementDateTestCase
):
    loader_type = BlazeDividendsByAnnouncementDateLoader

    def pipeline_event_loader_args(self, dates):
        _, mapping = super(
            BlazeDividendsByAnnouncementDateTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                ANNOUNCEMENT_FIELD_NAME: df[ANNOUNCEMENT_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME],
                CURRENCY_FIELD_NAME: df[CURRENCY_FIELD_NAME],
                DIVIDEND_TYPE_FIELD_NAME: df[DIVIDEND_TYPE_FIELD_NAME],
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeDividendsByAnnouncementDateNotInteractiveTestCase(
        BlazeDividendsByAnnouncementDateTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """

    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeDividendsByAnnouncementDateNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})


class DividendsByExDateTestCase(WithPipelineEventDataLoader, ZiplineTestCase):
    """
    Tests for loading the dividends by ex date data.
    """
    pipeline_columns = {
        NEXT_EX_DATE: DividendsByExDate.next_date.latest,
        PREVIOUS_EX_DATE: DividendsByExDate.previous_date.latest,
        NEXT_AMOUNT: DividendsByExDate.next_amount.latest,
        PREVIOUS_AMOUNT: DividendsByExDate.previous_amount.latest,
        PREVIOUS_CURRENCY_TYPE: DividendsByExDate.previous_currency.latest,
        NEXT_CURRENCY_TYPE: DividendsByExDate.next_currency.latest,
        PREVIOUS_DIVIDEND_TYPE: DividendsByExDate.previous_type.latest,
        NEXT_DIVIDEND_TYPE: DividendsByExDate.next_type.latest,
        DAYS_TO_NEXT_EX_DATE: BusinessDaysUntilNextExDate(),
        DAYS_SINCE_PREV_EX_DATE: BusinessDaysSincePreviousExDate()
    }

    @classmethod
    def get_dataset(cls):
        return {sid:
                frame.drop([ANNOUNCEMENT_FIELD_NAME,
                            PAY_DATE_FIELD_NAME], axis=1)
                for sid, frame
                in enumerate(dividends_cases)}

    loader_type = DividendsByExDateLoader

    def setup(self, dates):
        cols = {
            NEXT_EX_DATE: self.get_sids_to_frames(
                zip_with_dates, next_ex_and_pay_dates, next_date_intervals,
                dates,
                'datetime64[ns]', 'NaN'
            ),
            PREVIOUS_EX_DATE: self.get_sids_to_frames(
                zip_with_dates, prev_ex_and_pay_dates, prev_date_intervals,
                dates,
                'datetime64[ns]', 'NaN'
            ),
            NEXT_AMOUNT: self.get_sids_to_frames(
                zip_with_floats, next_amounts, next_date_intervals, dates,
                'float', 'NaN'
            ),
            PREVIOUS_AMOUNT: self.get_sids_to_frames(
                zip_with_floats, prev_amounts, prev_date_intervals, dates,
                'float', 'NaN'
            ),
            PREVIOUS_CURRENCY_TYPE: self.get_sids_to_frames(
                zip_with_strs, prev_currency_types, prev_date_intervals, dates,
                'category', None
            ),
            NEXT_CURRENCY_TYPE: self.get_sids_to_frames(
                zip_with_strs, next_currency_types, next_date_intervals, dates,
                'category', None
            ),
            PREVIOUS_DIVIDEND_TYPE: self.get_sids_to_frames(
                zip_with_strs, prev_dividend_types, prev_date_intervals, dates,
                'category', None
            ),
            NEXT_DIVIDEND_TYPE: self.get_sids_to_frames(
                zip_with_strs, next_dividend_types, next_date_intervals, dates,
                'category', None
            ),
        }

        cols[DAYS_TO_NEXT_EX_DATE] = self._compute_busday_offsets(
            cols[NEXT_EX_DATE]
        )
        cols[DAYS_SINCE_PREV_EX_DATE] = self._compute_busday_offsets(
            cols[PREVIOUS_EX_DATE]
        )
        return cols


class BlazeDividendsByExDateLoaderTestCase(DividendsByExDateTestCase):
    loader_type = BlazeDividendsByExDateLoader

    def pipeline_event_loader_args(self, dates):
        _, mapping = super(
            BlazeDividendsByExDateLoaderTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                EX_DATE_FIELD_NAME: df[EX_DATE_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME],
                CURRENCY_FIELD_NAME: df[CURRENCY_FIELD_NAME],
                DIVIDEND_TYPE_FIELD_NAME: df[DIVIDEND_TYPE_FIELD_NAME],
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeDividendsByExDateLoaderNotInteractiveTestCase(
        BlazeDividendsByExDateLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """

    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeDividendsByExDateLoaderNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})


class DividendsByPayDateTestCase(WithPipelineEventDataLoader, ZiplineTestCase):
    """
    Tests for loading the dividends by pay date data.
    """
    pipeline_columns = {
        NEXT_PAY_DATE: DividendsByPayDate.next_date.latest,
        PREVIOUS_PAY_DATE: DividendsByPayDate.previous_date.latest,
        NEXT_AMOUNT: DividendsByPayDate.next_amount.latest,
        PREVIOUS_AMOUNT: DividendsByPayDate.previous_amount.latest,
        PREVIOUS_CURRENCY_TYPE: DividendsByPayDate.previous_currency.latest,
        NEXT_CURRENCY_TYPE: DividendsByPayDate.next_currency.latest,
        PREVIOUS_DIVIDEND_TYPE: DividendsByPayDate.previous_type.latest,
        NEXT_DIVIDEND_TYPE: DividendsByPayDate.next_type.latest,
    }

    @classmethod
    def get_dataset(cls):
        return {sid:
                frame.drop([ANNOUNCEMENT_FIELD_NAME,
                            EX_DATE_FIELD_NAME], axis=1)
                for sid, frame
                in enumerate(dividends_cases)}

    loader_type = DividendsByPayDateLoader

    def setup(self, dates):
        return {
            NEXT_PAY_DATE: self.get_sids_to_frames(
                zip_with_dates, next_ex_and_pay_dates, next_date_intervals,
                dates,
                'datetime64[ns]', 'NaN'
            ),
            PREVIOUS_PAY_DATE: self.get_sids_to_frames(
                zip_with_dates, prev_ex_and_pay_dates, prev_date_intervals,
                dates,
                'datetime64[ns]', 'NaN'
            ),
            NEXT_AMOUNT: self.get_sids_to_frames(
                zip_with_floats, next_amounts, next_date_intervals, dates,
                'float', 'NaN'
            ),
            PREVIOUS_AMOUNT: self.get_sids_to_frames(
                zip_with_floats, prev_amounts, prev_date_intervals, dates,
                'float', 'NaN'
            ),
            PREVIOUS_CURRENCY_TYPE: self.get_sids_to_frames(
                zip_with_strs, prev_currency_types, prev_date_intervals, dates,
                'category', None
            ),
            NEXT_CURRENCY_TYPE: self.get_sids_to_frames(
                zip_with_strs, next_currency_types, next_date_intervals, dates,
                'category', None
            ),
            PREVIOUS_DIVIDEND_TYPE: self.get_sids_to_frames(
                zip_with_strs, prev_dividend_types, prev_date_intervals, dates,
                'category', None
            ),
            NEXT_DIVIDEND_TYPE: self.get_sids_to_frames(
                zip_with_strs, next_dividend_types, next_date_intervals, dates,
                'category', None
            ),
        }


class BlazeDividendsByPayDateLoaderTestCase(DividendsByPayDateTestCase):
    loader_type = BlazeDividendsByPayDateLoader

    def pipeline_event_loader_args(self, dates):
        _, mapping = super(
            BlazeDividendsByPayDateLoaderTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return (bz.Data(pd.concat(
            pd.DataFrame({
                PAY_DATE_FIELD_NAME: df[PAY_DATE_FIELD_NAME],
                TS_FIELD_NAME: df[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME],
                CURRENCY_FIELD_NAME: df[CURRENCY_FIELD_NAME],
                DIVIDEND_TYPE_FIELD_NAME: df[DIVIDEND_TYPE_FIELD_NAME],
            })
            for sid, df in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeDividendsByPayDateLoaderNotInteractiveTestCase(
        BlazeDividendsByPayDateLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """

    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeDividendsByPayDateLoaderNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
