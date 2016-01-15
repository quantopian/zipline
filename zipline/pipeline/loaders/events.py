from abc import abstractmethod
from itertools import repeat

import pandas as pd
from six import iteritems
from toolz import merge

from .base import PipelineLoader
from .frame import DataFrameLoader
from zipline.utils.memoize import lazyval


class EventsLoader(PipelineLoader):
    def __init__(self,
                 all_dates,
                 announcement_dates,
                 infer_timestamps=False,
                 dataset=None):
        self.all_dates = all_dates
        self.announcement_dates = (
            announcement_dates.copy()
        )
        dates = self.all_dates.values
        for k, v in iteritems(announcement_dates):
            if isinstance(v, pd.DatetimeIndex):
                if not infer_timestamps:
                    raise ValueError(
                        "Got DatetimeIndex of announcement dates for sid %d.\n"
                        "Pass `infer_timestamps=True` to use the first date in"
                        " `all_dates` as implicit timestamp."
                    )
                # If we are passed a DatetimeIndex, we always have
                # knowledge of the announcements.
                announcement_dates[k] = pd.Series(
                    v, index=repeat(dates[0], len(v)),
                )
        self.dataset = dataset


    @abstractmethod
    def get_loader(self):
        raise NotImplementedError("EventsLoader must implement 'get_loader'.")


    def load_adjusted_array(self, columns, dates, assets, mask):
        return merge(
            self.get_loader(column).load_adjusted_array(
                [column], dates, assets, mask
            )
            for column in columns
        )

    @lazyval
    def date_frame_loader(self, col_name, next_or_prev):
        return DataFrameLoader(
            col_name,
            next_or_prev(
                self.all_dates,
                self.announcement_dates,
            ),
            adjustments=None,
        )
