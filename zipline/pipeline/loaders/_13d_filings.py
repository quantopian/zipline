"""
Reference implementation for 13d filings loaders.
"""

from zipline.pipeline.common import (
    DISCLOSURE_DATE,
    PERCENT_SHARES,
    NUM_SHARES
)
from zipline.pipeline.data import _13DFilings
from zipline.pipeline.loaders.events import EventsLoader
from zipline.utils.memoize import lazyval


class _13DFilingsLoader(EventsLoader):
    """
    Reference loader for
    :class:`zipline.pipeline.data._13DFilings`.

    events_by_sid: dict[sid -> pd.DataFrame(knowledge date,
    disclosure date, percent shares, number of shares)]

    """
    expected_cols = frozenset([DISCLOSURE_DATE,
                               PERCENT_SHARES,
                               NUM_SHARES])
    event_date_col = DISCLOSURE_DATE

    def __init__(self, all_dates, events_by_sid,
                 infer_timestamps=False,
                 dataset=_13DFilings):
        super(_13DFilingsLoader, self).__init__(
            all_dates, events_by_sid, infer_timestamps, dataset=dataset,
        )

    @lazyval
    def disclosure_date_loader(self):
        return self._previous_event_date_loader(
            self.dataset.disclosure_date,
        )

    @lazyval
    def percent_shares_loader(self):
        return self._previous_event_value_loader(
            self.dataset.percent_shares,
            PERCENT_SHARES
        )

    @lazyval
    def number_shares_loader(self):
        return self._previous_event_value_loader(
            self.dataset.number_shares,
            NUM_SHARES
        )
