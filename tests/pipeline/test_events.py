"""
Tests for setting up an EventsLoader and a BlazeEventsLoader.
"""
from functools import partial
from nose_parameterized import parameterized

import blaze as bz
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal, TestCase
from zipline.pipeline import SimplePipelineEngine, Pipeline

from zipline.pipeline.common import (
    ANNOUNCEMENT_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)
from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.loaders.blaze.events import BlazeEventsLoader
from zipline.pipeline.loaders.events import (
    BAD_DATA_FORMAT_ERROR,
    DF_NO_TS_NOT_INFER_TS_ERROR,
    DTINDEX_NOT_INFER_TS_ERROR,
    EventsLoader,
    SERIES_NO_DTINDEX_ERROR,
    WRONG_COLS_ERROR,
)
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import datetime64ns_dtype, NaTD, make_datetime64D
from zipline.utils.test_utils import gen_calendars, num_days_in_range, \
    make_simple_equity_info

ABSTRACT_METHODS_ERROR = 'abstract methods concrete_loader'


class EventDataSet(DataSet):
    previous_announcement = Column(datetime64ns_dtype)


class EventDataSetLoader(EventsLoader):

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

    @property
    def expected_cols(self):
        return frozenset([ANNOUNCEMENT_FIELD_NAME])

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


def assert_loader_error(events_by_sid, error, msg, infer_timestamps):
    with TestCase.assertRaises(error) as context:
        EventDataSetLoader(
            dtx, events_by_sid, infer_timestamps=infer_timestamps,
        )
        TestCase.assertTrue(msg in context.exception)


class EventLoaderTestCase(TestCase):

    def test_no_expected_cols_defined(self):
        events_by_sid = {0: pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx})}
        assert_loader_error(events_by_sid, TypeError, ABSTRACT_METHODS_ERROR,
                            True)

    def test_wrong_cols(self):
        wrong_col_name = 'some_other_col'
        # Test wrong cols (cols != expected)
        events_by_sid = {0: pd.DataFrame({wrong_col_name: dtx})}
        assert_loader_error(
            events_by_sid, ValueError, WRONG_COLS_ERROR.format(
                expected_columns=EventDataSetLoader.expected_cols,
                sid=0,
                resulting_columns=wrong_col_name,
            ),
            True
        )

    @parameterized.expand([
        # DataFrame without timestamp column and infer_timestamps = True
        [pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx}), True],
        # DataFrame with timestamp column
        [pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx,
                       TS_FIELD_NAME: dtx}), False],
        # DatetimeIndex with infer_timestamps = True
        [pd.DatetimeIndex(dtx, name=ANNOUNCEMENT_FIELD_NAME), True],
        # Series with DatetimeIndex as index and infer_timestamps = False
        [pd.Series(dtx, index=dtx, name=ANNOUNCEMENT_FIELD_NAME), False]
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

    @parameterized.expand([
        # DataFrame without timestamp column and infer_timestamps = True
        [pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx}),
         False,
         DF_NO_TS_NOT_INFER_TS_ERROR.format(
             timestamp_column_name=TS_FIELD_NAME,
             sid=0
         )
         ],
        # DatetimeIndex with infer_timestamps = False
        [pd.DatetimeIndex(dtx, name=ANNOUNCEMENT_FIELD_NAME), False,
         DTINDEX_NOT_INFER_TS_ERROR.format(sid=0)],
        # Series with DatetimeIndex as index and infer_timestamps = False
        [pd.Series(dtx, name=ANNOUNCEMENT_FIELD_NAME), False,
         SERIES_NO_DTINDEX_ERROR.format(sid=0)],
        # Some other data structure that is not expected
        [dtx, False, BAD_DATA_FORMAT_ERROR.format(sid=0)],
        [dtx, True, BAD_DATA_FORMAT_ERROR.format(sid=0)]
    ])
    def test_bad_conversion_to_df(self, df, infer_timestamps, msg):
        events_by_sid = {0: df}
        assert_loader_error(events_by_sid, ValueError, msg,
                            infer_timestamps=infer_timestamps)


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
        with TestCase.assertRaises(TypeError) as context:
            BlazeEventDataSetLoaderNoConcreteLoader(
                bz.Data(
                    pd.DataFrame({ANNOUNCEMENT_FIELD_NAME: dtx,
                                  SID_FIELD_NAME: 0
                                  })
                )
            )
            TestCase.assertTrue(ABSTRACT_METHODS_ERROR in context.exception)







##########################


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


class EventLoaderCommonTest(object):
    sids = A, B, C, D, E = range(5)
    equity_info = make_simple_equity_info(
        sids,
        start_date=pd.Timestamp('2013-01-01', tz='UTC'),
        end_date=pd.Timestamp('2015-01-01', tz='UTC'),
    )

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

    def get_expected_previous(self, dates):
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

    def _test_compute_buyback_auth(self, dates):
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




