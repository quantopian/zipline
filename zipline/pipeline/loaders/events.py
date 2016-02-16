import numpy as np
import pandas as pd
from six import iteritems
from toolz import merge

from .base import PipelineLoader
from .frame import DataFrameLoader
from .utils import next_date_frame, previous_date_frame, previous_value

TS_FIELD_NAME = "timestamp"
SID_FIELD_NAME = "sid"


class EventsLoader(PipelineLoader):
    """
    Abstract loader.

    Does not currently support adjustments to the dates of known events.

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Index of dates for which we can serve queries.
    events_by_sid : dict[int -> pd.DataFrame]
        Dict mapping sids to DataFrames representing dates on which events
        occurred along with other associated values.

        If the DataFrames contain a "timestamp" column, that column is
        interpreted as the date on which we learned about the event.

        If the DataFrames do not contain a "timestamp" column, we assume we
        knew about the event on all prior dates.  This mode is only supported
        if ``infer_timestamp`` is explicitly passed as a truthy value.
    infer_timestamps : bool, optional
        Whether to allow omitting the "timestamp" column.
    dataset : DataSet
        The DataSet object for which this loader loads data.
    expected_cols : frozenset
        Set of expected columns for the dataset, without timestamp.
    """

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=None,
                 expected_cols=frozenset()):
        self.all_dates = all_dates
        # Do not modify the original in place, since it may be used for other
        #  purposes.
        self.events_by_sid = (
            events_by_sid.copy()
        )
        dates = self.all_dates.values

        for k, v in iteritems(events_by_sid):
            # First, must convert to DataFrame.
            if isinstance(v, pd.Series):
                # If Series was passed, DateTime index is assumed.
                self.events_by_sid[k] = pd.DataFrame(v)
            elif isinstance(v, pd.DatetimeIndex):
                if not infer_timestamps:
                    raise ValueError(
                        "Got DatetimeIndex for sid %d.\n"
                        "Pass `infer_timestamps=True` to use the first date in"
                        " `all_dates` as implicit timestamp." % k
                    )
                self.events_by_sid[k] = pd.DataFrame(v)
                v.index = [dates[0]] * len(v)
            # Already a DataFrame
            elif isinstance(v, pd.DataFrame):
                if TS_FIELD_NAME not in v.columns:
                    if not infer_timestamps:
                        raise ValueError(
                            "Got DataFrame without a '%s' column for sid %d.\n"
                            "Pass `infer_timestamps=True` to use the first "
                            "date in `all_dates` as implicit timestamp." %
                            (TS_FIELD_NAME, k)
                        )
                    self.events_by_sid[k] = v = v.copy()
                    v.index = [dates[0]] * len(v)
                else:
                    self.events_by_sid[k] = v.set_index(TS_FIELD_NAME)
            else:
                raise ValueError("Data for sid %s must be in DataFrame, "
                                 "Series, or DatetimeIndex." % k)
            # Once data is in a DF, make sure columns are correct.
            cols_except_ts = (set(v.columns.values) -
                              {TS_FIELD_NAME} -
                              {SID_FIELD_NAME})
            # Check that all columns other than timestamp are as expected.
            if cols_except_ts != expected_cols:
                raise ValueError(
                    "Expected columns %s for sid %s but got columns %s." %
                    (expected_cols, k, v.columns.values)
                )
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

    def mk_date_series(self, date_field_name):
        return {sid: pd.Series(index=event.index,
                               data=np.array(event[date_field_name]))
                for sid, event in iteritems(self.events_by_sid)}

    def _next_event_date_loader(self, next_date_field, event_date_field_name):
        return DataFrameLoader(
            next_date_field,
            next_date_frame(
                self.all_dates,
                self.mk_date_series(event_date_field_name),
            ),
            adjustments=None,
        )

    def _previous_event_date_loader(self,
                                    prev_date_field,
                                    event_date_field_name):
        return DataFrameLoader(
            prev_date_field,
            previous_date_frame(
                self.all_dates,
                self.mk_date_series(event_date_field_name),
            ),
            adjustments=None,
        )

    def _previous_event_value_loader(self,
                                     previous_value_field,
                                     event_date_field_name,
                                     value_field_name):
        return DataFrameLoader(
            previous_value_field,
            previous_value(
                self.all_dates,
                self.events_by_sid,
                event_date_field_name,
                value_field_name,
                previous_value_field.dtype,
                previous_value_field.missing_value
            ),
            adjustments=None,
        )
