"""
Reference implementation for EarningsCalendar loaders.
"""

from ..data import EarningsCalendar
from .events import EventsLoader
from zipline.pipeline.common import ANNOUNCEMENT_FIELD_NAME
from zipline.utils.memoize import lazyval


class EarningsCalendarLoader(EventsLoader):

    expected_cols = frozenset([ANNOUNCEMENT_FIELD_NAME])

    event_date_col = ANNOUNCEMENT_FIELD_NAME

    def __init__(self, all_dates, events_by_sid,
                 infer_timestamps=False,
                 dataset=EarningsCalendar):
        super(EarningsCalendarLoader, self).__init__(
            all_dates, events_by_sid, infer_timestamps, dataset=dataset,
        )

    @lazyval
    def next_announcement_loader(self):
        return self._next_event_date_loader(self.dataset.next_announcement)

    @lazyval
    def previous_announcement_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement,
        )
