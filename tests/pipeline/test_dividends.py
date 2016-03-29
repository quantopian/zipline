"""
Tests for the reference loader for Dividends datasets.
"""
import blaze as bz
from blaze.compute.core import swap_resources_into_scope
import pandas as pd
from six import iteritems

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
    DividendsByAnnouncementDateLoader,
    DividendsByExDateLoader,
    DividendsByPayDateLoader
)
from zipline.pipeline.loaders.utils import (
    get_values_for_date_ranges,
    zip_with_dates,
    zip_with_floats
)
from zipline.testing.fixtures import (
    WithPipelineEventDataLoader,
    ZiplineTestCase
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


def get_vals_for_dates(zip_date_index_with_vals,
                       vals,
                       date_invervals,
                       dates):
    return pd.DataFrame({
        0: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals[0],
                                      date_invervals[0],
                                      dates),
        1: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals[1],
                                      date_invervals[1],
                                      dates),
        2: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals[2],
                                      date_invervals[2],
                                      dates),
        # Assume the latest of 2 cash values is used if we find out about 2
        # announcements that happened on the same day for the same sid.
        3: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals[3],
                                      date_invervals[3],
                                      dates),
        4: zip_date_index_with_vals(dates, ['NaN'] * len(dates)),
    }, index=dates)


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
        cols = {}
        cols[PREVIOUS_ANNOUNCEMENT] = get_vals_for_dates(
            zip_with_dates, announcement_dates, date_intervals, dates
        )

        cols[PREVIOUS_AMOUNT] = get_vals_for_dates(
            zip_with_floats, amounts, date_intervals, dates
        )

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
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME]
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
        cols = {}
        cols[NEXT_EX_DATE] = get_vals_for_dates(
            zip_with_dates, next_ex_and_pay_dates, next_date_intervals, dates,
        )

        cols[PREVIOUS_EX_DATE] = get_vals_for_dates(
            zip_with_dates, prev_ex_and_pay_dates, prev_date_intervals, dates
        )

        cols[NEXT_AMOUNT] = get_vals_for_dates(
            zip_with_floats, next_amounts, next_date_intervals, dates
        )

        cols[PREVIOUS_AMOUNT] = get_vals_for_dates(
            zip_with_floats, prev_amounts, prev_date_intervals, dates
        )

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
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME]
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
        cols = {}
        cols[NEXT_PAY_DATE] = get_vals_for_dates(
            zip_with_dates, next_ex_and_pay_dates, next_date_intervals, dates
        )
        cols[PREVIOUS_PAY_DATE] = get_vals_for_dates(
            zip_with_dates, prev_ex_and_pay_dates, prev_date_intervals, dates
        )

        cols[NEXT_AMOUNT] = get_vals_for_dates(
            zip_with_floats, next_amounts, next_date_intervals, dates
        )

        cols[PREVIOUS_AMOUNT] = get_vals_for_dates(
            zip_with_floats, prev_amounts, prev_date_intervals, dates
        )
        return cols


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
                CASH_AMOUNT_FIELD_NAME: df[CASH_AMOUNT_FIELD_NAME]
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
