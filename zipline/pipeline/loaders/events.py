import abc
import pandas as pd
from six import iteritems
from toolz import merge

from .base import PipelineLoader
from .frame import DataFrameLoader
from .utils import previous_event_frame, next_event_frame
from zipline.pipeline.common import TS_FIELD_NAME
from zipline.utils.numpy_utils import NaTD

WRONG_COLS_ERROR = "Expected columns {expected_columns} for sid {sid} but " \
                   "got columns {resulting_columns}."

WRONG_SINGLE_COL_DATA_FORMAT_ERROR = ("Data for sid {sid} is expected to have "
                                      "1 column and to be in a DataFrame, "
                                      "Series, or DatetimeIndex.")

WRONG_MANY_COL_DATA_FORMAT_ERROR = ("Data for sid {sid} is expected to have "
                                    "more than 1 column and to be in a "
                                    "DataFrame.")

SERIES_NO_DTINDEX_ERROR = ("Got Series for sid {sid}, but index was not "
                           "DatetimeIndex.")

DTINDEX_NOT_INFER_TS_ERROR = ("Got DatetimeIndex for sid {sid}.\n"
                              "Pass `infer_timestamps=True` to use the first "
                              "date in `all_dates` as implicit timestamp.")

DF_NO_TS_NOT_INFER_TS_ERROR = ("Got DataFrame without a '{"
                               "timestamp_column_name}' column for sid {sid}."
                               "\nPass `infer_timestamps=True` to use the "
                               "first date in `all_dates` as implicit "
                               "timestamp.")


class EventsLoader(PipelineLoader):
    """
    Abstract loader.

    Does not currently support adjustments to the dates of known events.

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Index of dates for which we can serve queries.
    events_by_sid : dict[int -> pd.DataFrame or pd.Series or pd.DatetimeIndex]
        Dict mapping sids to objects representing dates on which earnings
        occurred.

        If a dict value is a Series, it's interpreted as a mapping from the
        date on which we learned an announcement was coming to the date on
        which the announcement was made.

        If a dict value is a DatetimeIndex, it's interpreted as just containing
        the dates that announcements were made, and we assume we knew about the
        announcement on all prior dates.  This mode is only supported if
        ``infer_timestamp`` is explicitly passed as a truthy value.
        Dict mapping sids to DataFrames, Series, or DatetimeIndexes.

        If the value is a DataFrame, it then represents dates on which events
        occurred along with other associated values. If the DataFrame
        contains a "timestamp" column, that column is interpreted as the date
        on which we learned about the event. If the DataFrames do not contain a
         "timestamp" column, we assume we knew about the event on all prior
         dates.  This mode is only supported if ``infer_timestamp`` is
         explicitly passed as a truthy value.

    infer_timestamps : bool, optional
        Whether to allow omitting the "timestamp" column.
    dataset : DataSet
        The DataSet object for which this loader loads data.

    """

    @abc.abstractproperty
    def expected_cols(self):
        raise NotImplemented('expected_cols')

    @abc.abstractproperty
    def event_date_col(self):
        raise NotImplemented('event_date_col')

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=None):
        self.all_dates = all_dates
        # Do not modify the original in place, since it may be used for other
        #  purposes.
        self.events_by_sid = (
            events_by_sid.copy()
        )
        dates = self.all_dates.values

        for k, v in iteritems(events_by_sid):
            # Already a DataFrame
            if isinstance(v, pd.DataFrame):
                if TS_FIELD_NAME not in v.columns:
                    if not infer_timestamps:
                        raise ValueError(
                            DF_NO_TS_NOT_INFER_TS_ERROR.format(
                                timestamp_column_name=TS_FIELD_NAME,
                                sid=k
                            )
                        )
                    self.events_by_sid[k] = v = v.copy()
                    v.index = [dates[0]] * len(v)
                else:
                    self.events_by_sid[k] = v.set_index(TS_FIELD_NAME)
                # Once data is in a DF, make sure columns are correct.
                cols_except_ts = (set(v.columns) -
                                  {TS_FIELD_NAME})

                # Check that all columns other than timestamp are as expected.
                if cols_except_ts != self.expected_cols:
                    raise ValueError(
                        WRONG_COLS_ERROR.format(
                            expected_columns=list(self.expected_cols),
                            sid=k,
                            resulting_columns=v.columns.values
                        )
                    )
            # Not a DataFrame and we only expect 1 column
            elif len(self.expected_cols) == 1:
                # First, must convert to DataFrame.
                if isinstance(v, pd.Series):
                    if not isinstance(v.index, pd.DatetimeIndex):
                        raise ValueError(
                            SERIES_NO_DTINDEX_ERROR.format(sid=k)
                        )
                    self.events_by_sid[k] = pd.DataFrame({
                        list(self.expected_cols)[0]: v})
                elif isinstance(v, pd.DatetimeIndex):
                    if not infer_timestamps:
                        raise ValueError(
                            DTINDEX_NOT_INFER_TS_ERROR.format(sid=k)
                        )
                    self.events_by_sid[k] = pd.DataFrame({
                        list(self.expected_cols)[0]: v
                    }, index=[dates[0]] * len(v))
                else:
                    # We expect 1 column, but we got something other than a
                    # Series, DatetimeIndex, or DataFrame.
                    raise ValueError(
                        WRONG_SINGLE_COL_DATA_FORMAT_ERROR.format(sid=k)
                    )
            else:
                # We expected multiple columns, but we got something other
                # than a DataFrame.
                raise ValueError(
                    WRONG_MANY_COL_DATA_FORMAT_ERROR.format(sid=k)
                )
        self.events_by_sid = {sid: df.dropna(subset=[self.event_date_col]) for
                              sid, df in self.events_by_sid.items()}
        self.dataset = dataset

    def get_loader(self, column):
        if column in self.dataset.columns:
            return getattr(self, "%s_loader" % column.name)
        raise ValueError("Don't know how to load column '%s'." % column)

    def load_adjusted_array(self, columns, dates, assets, mask):
        return merge(
            self.get_loader(column).load_adjusted_array(
                [column], dates, assets, mask
            )
            for column in columns
        )

    def _next_event_date_loader(self, next_date_field):
        return DataFrameLoader(
            next_date_field,
            next_event_frame(
                self.events_by_sid,
                self.all_dates,
                next_date_field.missing_value,
                next_date_field.dtype,
                self.event_date_col,
                self.event_date_col
            ),
            adjustments=None,
        )

    def _next_event_value_loader(self,
                                 next_value_field,
                                 value_field_name):
        return DataFrameLoader(
            next_value_field,
            next_event_frame(
                self.events_by_sid,
                self.all_dates,
                next_value_field.missing_value,
                next_value_field.dtype,
                self.event_date_col,
                value_field_name
            ),
            adjustments=None,
        )

    def _previous_event_date_loader(self,
                                    prev_date_field):
        return DataFrameLoader(
            prev_date_field,
            previous_event_frame(
                self.events_by_sid,
                self.all_dates,
                NaTD,
                'datetime64[ns]',
                self.event_date_col,
                self.event_date_col
            ),
            adjustments=None,
        )

    def _previous_event_value_loader(self,
                                     previous_value_field,
                                     value_field_name):
        return DataFrameLoader(
            previous_value_field,
            previous_event_frame(
                self.events_by_sid,
                self.all_dates,
                previous_value_field.missing_value,
                previous_value_field.dtype,
                self.event_date_col,
                value_field_name
            ),
            adjustments=None,
        )
