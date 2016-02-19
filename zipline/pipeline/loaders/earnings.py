"""
Reference implementation for EarningsCalendar loaders.
"""

from ..data.earnings import EarningsCalendar
from .events import EventsLoader
from zipline.utils.memoize import lazyval

ANNOUNCEMENT_FIELD_NAME = "announcement_date"


class EarningsCalendarLoader(EventsLoader):

    def __init__(self, all_dates, events_by_sid,
                 infer_timestamps=False,
                 dataset=EarningsCalendar):
        super(EarningsCalendarLoader, self).__init__(
            all_dates, events_by_sid, infer_timestamps, dataset=dataset,
        )

    @property
    def expected_cols(self):
        return frozenset([ANNOUNCEMENT_FIELD_NAME])

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
