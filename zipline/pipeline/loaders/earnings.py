"""
Reference implementation for EarningsCalendar loaders.
"""

from events import EventsLoader
from ..data.earnings import EarningsCalendar
from zipline.utils.memoize import lazyval

ANNOUNCEMENT_FIELD_NAME = "announcement_date"


class EarningsCalendarLoader(EventsLoader):
    def __init__(self, all_dates, events_by_sid, infer_timestamps=False,
                 dataset=EarningsCalendar):
        super(EarningsCalendarLoader, self).__init__(all_dates,
                                                     events_by_sid,
                                                     infer_timestamps,
                                                     dataset=dataset)

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
        return self._next_event_date_loader(self.dataset.next_announcement,
                                            ANNOUNCEMENT_FIELD_NAME)

    @lazyval
    def previous_announcement_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement,
            ANNOUNCEMENT_FIELD_NAME
        )
