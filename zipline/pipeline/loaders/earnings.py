"""
Reference implementation for EarningsCalendar loaders.
"""
from itertools import repeat

import pandas as pd
from six import iteritems
from toolz import merge

from .base import PipelineLoader
from .frame import DataFrameLoader
from .utils import next_date_frame, previous_date_frame
from ..data.earnings import EarningsCalendar
from zipline.utils.memoize import lazyval


class EarningsCalendarLoader(PipelineLoader):
    """
    Reference loader for
    :class:`zipline.pipeline.data.earnings.EarningsCalendar`.

    Does not currently support adjustments to the dates of known earnings.

    Parameters
    ----------
    all_dates : pd.DatetimeIndex
        Index of dates for which we can serve queries.
    announcement_dates : dict[int -> pd.Series or pd.DatetimeIndex]
        Dict mapping sids to objects representing dates on which earnings
        occurred.

        If a dict value is a Series, it's interpreted as a mapping from the
        date on which we learned an announcement was coming to the date on
        which the announcement was made.

        If a dict value is a DatetimeIndex, it's interpreted as just containing
        the dates that announcements were made, and we assume we knew about the
        announcement on all prior dates.  This mode is only supported if
        ``infer_timestamp`` is explicitly passed as a truthy value.

    infer_timestamps : bool, optional
        Whether to allow passing ``DatetimeIndex`` values in
        ``announcement_dates``.
    """
    def __init__(self,
                 all_dates,
                 announcement_dates,
                 infer_timestamps=False,
                 dataset=EarningsCalendar):
        self.all_dates = all_dates
        self.announcement_dates = announcement_dates = (
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

    def get_loader(self, column):
        """Dispatch to the loader for ``column``.
        """
        if column is self.dataset.next_announcement:
            return self.next_announcement_loader
        elif column is self.dataset.previous_announcement:
            return self.previous_announcement_loader
        else:
            raise ValueError("Don't know how to load column '%s'." % column)

    @lazyval
    def next_announcement_loader(self):
        return DataFrameLoader(
            self.dataset.next_announcement,
            next_date_frame(
                self.all_dates,
                self.announcement_dates,
            ),
            adjustments=None,
        )

    @lazyval
    def previous_announcement_loader(self):
        return DataFrameLoader(
            self.dataset.previous_announcement,
            previous_date_frame(
                self.all_dates,
                self.announcement_dates,
            ),
            adjustments=None,
        )

    def load_adjusted_array(self, columns, dates, assets, mask):
        return merge(
            self.get_loader(column).load_adjusted_array(
                [column], dates, assets, mask
            )
            for column in columns
        )
