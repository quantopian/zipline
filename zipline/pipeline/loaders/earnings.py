"""
Reference implementation for EarningsCalendar loaders.
"""

from ..data.earnings import EarningsCalendar
from .events import EventsLoader
from zipline.utils.memoize import lazyval

ANNOUNCEMENT_FIELD_NAME = "announcement_date"


class EarningsCalendarLoader(EventsLoader):
    expected_cols = frozenset([ANNOUNCEMENT_FIELD_NAME])

    def __init__(self, all_dates, events_by_sid,
                 infer_timestamps=False,
                 dataset=EarningsCalendar,
                 expected_cols=expected_cols):
        super(EarningsCalendarLoader, self).__init__(
            all_dates, events_by_sid, infer_timestamps, dataset=dataset,
            expected_cols=expected_cols
        )

    @lazyval
    def next_announcement_loader(self):
        return self._next_event_date_loader(self.dataset.next_announcement,
                                            ANNOUNCEMENT_FIELD_NAME)

    @lazyval
    def previous_announcement_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement,
            ANNOUNCEMENT_FIELD_NAME
        )
