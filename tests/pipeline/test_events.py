"""
Tests for setting up an EventsLoader and a BlazeEventsLoader.
"""
from functools import partial
from nose_parameterized import parameterized
import re
from unittest import TestCase

import blaze as bz
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from zipline.pipeline import SimplePipelineEngine, Pipeline

from zipline.pipeline.common import (
    ANNOUNCEMENT_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)
from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.loaders.blaze.events import BlazeEventsLoader
from zipline.pipeline.loaders.events import (
    DF_NO_TS_NOT_INFER_TS_ERROR,
    DTINDEX_NOT_INFER_TS_ERROR,
    EventsLoader,
    SERIES_NO_DTINDEX_ERROR,
    WRONG_COLS_ERROR,
    WRONG_MANY_COL_DATA_FORMAT_ERROR,
    WRONG_SINGLE_COL_DATA_FORMAT_ERROR
)
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    NaTD,
    make_datetime64D
)
from zipline.utils.test_utils import (
    gen_calendars,
    num_days_in_range,
    make_simple_equity_info
)

ABSTRACT_CONCRETE_LOADER_ERROR = 'abstract methods concrete_loader'
ABSTRACT_EXPECTED_COLS_ERROR = 'abstract methods expected_cols'
DATE_FIELD_NAME = "event_date"


class EventDataSet(DataSet):
    previous_announcement = Column(datetime64ns_dtype)


class EventDataSetLoader(EventsLoader):
    expected_cols = frozenset([ANNOUNCEMENT_FIELD_NAME])

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=EventDataSet):
        super(EventDataSetLoader, self).__init__(
            all_dates,
            events_by_sid,
            infer_timestamps=infer_timestamps,
            dataset=dataset,
        )

    @lazyval
    def previous_announcement_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement,
            ANNOUNCEMENT_FIELD_NAME,
        )

    @lazyval
    def next_announcement_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement,
            ANNOUNCEMENT_FIELD_NAME,
        )


# Test case just for catching an error when multiple columns are in the wrong
#  data format, so no loader defined.
class EventDataSetLoaderMultipleExpectedCols(EventsLoader):
    expected_cols = frozenset([ANNOUNCEMENT_FIELD_NAME, "other_field"])


class EventDataSetLoaderNoExpectedCols(EventsLoader):

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=EventDataSet):
        super(EventDataSetLoaderNoExpectedCols, self).__init__(
            all_dates,
            events_by_sid,
            infer_timestamps=infer_timestamps,
            dataset=dataset,
        )


dtx = pd.date_range('2014-01-01', '2014-01-10')


class EventLoaderTestCase(TestCase):
    def assert_loader_error(self, events_by_sid, error, msg,
                            infer_timestamps, loader):
        with self.assertRaisesRegexp(error, re.escape(msg)):
            loader(
                dtx, events_by_sid, infer_timestamps=infer_timestamps,
            )

    def test_no_expected_cols_defined(self):
        events_by_sid = {0: pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx})}
        self.assert_loader_error(events_by_sid, TypeError,
                                 ABSTRACT_EXPECTED_COLS_ERROR,
                                 True, EventDataSetLoaderNoExpectedCols)

    def test_wrong_cols(self):
        wrong_col_name = 'some_other_col'
        # Test wrong cols (cols != expected)
        events_by_sid = {0: pd.DataFrame({wrong_col_name: dtx})}
        self.assert_loader_error(
            events_by_sid, ValueError, WRONG_COLS_ERROR.format(
                expected_columns=list(EventDataSetLoader.expected_cols),
                sid=0,
                resulting_columns=[wrong_col_name],
            ),
            True,
            EventDataSetLoader
        )

    @parameterized.expand([
        # DataFrame without timestamp column and infer_timestamps = True
        [pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx}), True],
        # DataFrame with timestamp column
        [pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx,
                       TS_FIELD_NAME: dtx}), False],
        # DatetimeIndex with infer_timestamps = True
        [pd.DatetimeIndex(dtx), True],
        # Series with DatetimeIndex as index and infer_timestamps = False
        [pd.Series(dtx, index=dtx), False]
    ])
    def test_conversion_to_df(self, df, infer_timestamps):

        events_by_sid = {0: df}
        loader = EventDataSetLoader(
            dtx,
            events_by_sid,
            infer_timestamps=infer_timestamps,
        )
        self.assertEqual(
            loader.events_by_sid.keys(),
            events_by_sid.keys(),
        )

        if infer_timestamps:
            expected = pd.Series(index=[dtx[0]] * 10, data=dtx,
                                 name=ANNOUNCEMENT_FIELD_NAME)
        else:
            expected = pd.Series(index=dtx, data=dtx,
                                 name=ANNOUNCEMENT_FIELD_NAME)
            expected.index.name = TS_FIELD_NAME
        # Check that index by first given date has been added
        assert_series_equal(
            loader.events_by_sid[0][ANNOUNCEMENT_FIELD_NAME],
            expected,
        )

    @parameterized.expand(
        [
            # DataFrame without timestamp column and infer_timestamps = True
            [
                pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx}),
                False,
                DF_NO_TS_NOT_INFER_TS_ERROR.format(
                    timestamp_column_name=TS_FIELD_NAME,
                    sid=0
                ),
                EventDataSetLoader
            ],
            # DatetimeIndex with infer_timestamps = False
            [
                pd.DatetimeIndex(dtx, name=ANNOUNCEMENT_FIELD_NAME),
                False,
                DTINDEX_NOT_INFER_TS_ERROR.format(sid=0),
                EventDataSetLoader
            ],
            # Series with DatetimeIndex as index and infer_timestamps = False
            [
                pd.Series(dtx, name=ANNOUNCEMENT_FIELD_NAME),
                False,
                SERIES_NO_DTINDEX_ERROR.format(sid=0),
                EventDataSetLoader
            ],
            # Below, 2 cases repeated for infer_timestamps = True and False.
            # Shouldn't make a difference in the outcome.
            # We expected 1 column but got a data structure other than a
            # DataFrame, Series, or DatetimeIndex
            [
                [dtx],
                True,
                WRONG_SINGLE_COL_DATA_FORMAT_ERROR.format(sid=0),
                EventDataSetLoader
            ],
            # We expected multiple columns but got a data structure other
            # than a DataFrame
            [
                [dtx, dtx],
                True,
                WRONG_MANY_COL_DATA_FORMAT_ERROR.format(sid=0),
                EventDataSetLoaderMultipleExpectedCols
            ],
            [
                [dtx],
                False,
                WRONG_SINGLE_COL_DATA_FORMAT_ERROR.format(sid=0),
                EventDataSetLoader
            ],
            # We expected multiple columns but got a data structure other
            # than a DataFrame
            [
                [dtx, dtx],
                False,
                WRONG_MANY_COL_DATA_FORMAT_ERROR.format(sid=0),
                EventDataSetLoaderMultipleExpectedCols
            ]
        ]
    )
    def test_bad_conversion_to_df(self, df, infer_timestamps, msg, loader):
        events_by_sid = {0: df}
        self.assert_loader_error(events_by_sid, ValueError, msg,
                                 infer_timestamps, loader)


class BlazeEventDataSetLoaderNoConcreteLoader(BlazeEventsLoader):
    def __init__(self,
                 expr,
                 dataset=EventDataSet,
                 **kwargs):
        super(
            BlazeEventDataSetLoaderNoConcreteLoader, self
        ).__init__(expr,
                   dataset=dataset,
                   **kwargs)


class BlazeEventLoaderTestCase(TestCase):
    # Blaze loader: need to test failure if no concrete loader
    def test_no_concrete_loader_defined(self):
        with self.assertRaisesRegexp(
                TypeError, re.escape(ABSTRACT_CONCRETE_LOADER_ERROR)
        ):
            BlazeEventDataSetLoaderNoConcreteLoader(
                bz.Data(
                    pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx,
                                  SID_FIELD_NAME: 0
                                  })
                )
            )


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
    ], utc=True),
))


class EventLoaderCommonMixin(object):
    sids = A, B, C, D, E = range(5)
    equity_info = make_simple_equity_info(
        sids,
        start_date=pd.Timestamp('2013-01-01', tz='UTC'),
        end_date=pd.Timestamp('2015-01-01', tz='UTC'),
    )

    event_dates_cases = [
        # K1--K2--E1--E2.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
            DATE_FIELD_NAME: pd.to_datetime(['2014-01-15', '2014-01-20'])
        }),
        # K1--K2--E2--E1.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
            DATE_FIELD_NAME: pd.to_datetime(['2014-01-20', '2014-01-15'])
        }),
        # K1--E1--K2--E2.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
            DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-20'])
        }),
        # K1 == K2.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
            DATE_FIELD_NAME: pd.to_datetime(['2014-01-10', '2014-01-15'])
        }),
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime([]),
            DATE_FIELD_NAME: pd.to_datetime([])
        })
    ]

    def zip_with_floats(self, dates, flts):
        return pd.Series(flts, index=dates).astype('float')

    def num_days_between(self, dates, start_date, end_date):
        return num_days_in_range(dates, start_date, end_date)

    def zip_with_dates(self, index_dates, dts):
        return pd.Series(pd.to_datetime(dts), index=index_dates)

    def loader_args(self, dates):
        """Construct the base  object to pass to the loader.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The dates we can serve.

        Returns
        -------
        args : tuple[any]
            The arguments to forward to the loader positionally.
        """
        return dates, self.dataset

    def setup_engine(self, dates):
        """
        Make a Pipeline Enigne object based on the given dates.
        """
        loader = self.loader_type(*self.loader_args(dates))
        return SimplePipelineEngine(lambda _: loader, dates, self.finder)

    def get_expected_next_event_dates(self, dates):
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)
        return pd.DataFrame({
            0: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-15'] *
                num_days_between_for_dates('2014-01-05', '2014-01-15') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-16', '2014-01-20') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-21', None)
            ),
            1: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-05', '2014-01-09') +
                ['2014-01-15'] *
                num_days_between_for_dates('2014-01-10', '2014-01-15') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-16', '2014-01-20') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-21', None)
            ),
            2: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-10'] *
                num_days_between_for_dates('2014-01-05', '2014-01-10') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-11', '2014-01-14') +
                ['2014-01-20'] *
                num_days_between_for_dates('2014-01-15', '2014-01-20') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-21', None)
            ),
            3: zip_with_dates_for_dates(
                ['NaT'] *
                num_days_between_for_dates(None, '2014-01-04') +
                ['2014-01-10'] *
                num_days_between_for_dates('2014-01-05', '2014-01-10') +
                ['2014-01-15'] *
                num_days_between_for_dates('2014-01-11', '2014-01-15') +
                ['NaT'] *
                num_days_between_for_dates('2014-01-16', None)
            ),
            4: zip_with_dates_for_dates(['NaT'] *
                                        len(dates)),
        }, index=dates)

    def get_expected_previous_event_dates(self, dates):
        num_days_between_for_dates = partial(self.num_days_between, dates)
        zip_with_dates_for_dates = partial(self.zip_with_dates, dates)
        return pd.DataFrame({
            0: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            1: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            2: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between_for_dates('2014-01-10',
                                                            '2014-01-19') +
                ['2014-01-20'] * num_days_between_for_dates('2014-01-20',
                                                            None),
            ),
            3: zip_with_dates_for_dates(
                ['NaT'] * num_days_between_for_dates(None, '2014-01-09') +
                ['2014-01-10'] * num_days_between_for_dates('2014-01-10',
                                                            '2014-01-14') +
                ['2014-01-15'] * num_days_between_for_dates('2014-01-15',
                                                            None),
            ),
            4: zip_with_dates_for_dates(['NaT'] * len(dates)),
        }, index=dates)

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

    @parameterized.expand(param_dates)
    def test_compute(self, dates):
        engine = self.setup_engine(dates)
        self.setup(dates)

        pipe = Pipeline(
            columns=self.pipeline_columns
        )

        result = engine.run_pipeline(
            pipe,
            start_date=dates[0],
            end_date=dates[-1],
        )

        for sid in self.sids:
            for col_name in self.cols.keys():
                assert_series_equal(result[col_name].xs(sid, level=1),
                                    self.cols[col_name][sid],
                                    check_names=False)
